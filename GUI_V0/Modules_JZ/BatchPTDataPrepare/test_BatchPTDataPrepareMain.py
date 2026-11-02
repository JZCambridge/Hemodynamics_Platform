# -*- coding: utf-8 -*-
"""Focused regression tests for Batch PT Data Prepare."""

import csv
from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import SimpleITK as sitk

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from . import BatchPTDataPrepareMain as module
except ImportError:
    import BatchPTDataPrepareMain as module

try:
    from PySide2.QtUiTools import QUiLoader
    from PySide2.QtWidgets import QApplication
    QT_TESTS_AVAILABLE = True
except ImportError:
    QUiLoader = None
    QApplication = None
    QT_TESTS_AVAILABLE = False

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
GUI_ROOT = Path(__file__).resolve().parents[2]


class _FlushRecordingBuffer(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        super().flush()


def _touch(path, payload=b"nifti"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_nifti(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(
        sitk.GetImageFromArray(np.asarray(array)),
        str(path),
    )
    return path


def _config(input_folder, output_parent, **overrides):
    values = {
        "input_folder": str(input_folder),
        "output_parent": str(output_parent),
        "manifest_path": "",
        "layout": "auto",
        "selected_arteries": module.ARTERIES,
        "write_vit": True,
        "write_patch": False,
        "max_workers": 2,
        "save_qc": False,
        "qc_count": 3,
        "checkpoint_cases": 2,
    }
    values.update(overrides)
    return module.PTPrepareConfig(**values)


class FilenameAndDiscoveryTests(unittest.TestCase):
    def test_right_suffix_preserves_case_underscores(self):
        self.assertEqual(
            module.parse_prepared_filename(
                "110029_from_pacs_LCX.nii.gz"
            ),
            ("110029_from_pacs", "LCX"),
        )
        self.assertEqual(
            module.parse_prepared_filename("case_RCA_LAD.nii.gz"),
            ("case_RCA", "LAD"),
        )
        self.assertIsNone(
            module.parse_prepared_filename(
                "case_LCX_resampled.csv"
            )
        )

    def test_nested_pipeline_and_partial_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            output = Path(temporary) / "output"
            _touch(
                root / "batch_0" / "case10" / "Preproc"
                / "Xsection" / "mRCACTALI.nii.gz"
            )
            _touch(
                root / "batch_0" / "case10" / "Preproc"
                / "Xsection" / "mLADCTALI.nii.gz"
            )
            _touch(
                root / "batch_2" / "case2" / "Preproc"
                / "Xsection" / "LCxCTALI.nii.gz"
            )
            records = module.discover_xsection_inputs(
                _config(root, output)
            )
            self.assertEqual(len(records), 3)
            self.assertEqual(
                [(item.batch_name, item.case_id, item.artery)
                 for item in records],
                [
                    ("batch_0", "case10", "RCA"),
                    ("batch_0", "case10", "LAD"),
                    ("batch_2", "case2", "LCX"),
                ],
            )
            self.assertTrue(
                all(item.layout == "pipeline" for item in records)
            )

    def test_single_batch_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch_5"
            _touch(
                root / "case1" / "Preproc" / "Xsection"
                / "LCxCTALI.nii.gz"
            )
            records = module.discover_xsection_inputs(
                _config(root, Path(temporary) / "out")
            )
            self.assertEqual(records[0].batch_name, "batch_5")
            self.assertEqual(records[0].case_id, "case1")

    def test_empty_pipeline_case_is_a_valid_missing_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "batch_5"
            (
                root / "empty_case" / "Preproc" / "Xsection"
            ).mkdir(parents=True)
            records = module.discover_xsection_inputs(
                _config(root, Path(temporary) / "out")
            )
            self.assertEqual(records, ())

    def test_prepared_case_folder_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "prepared"
            _touch(
                root / "case_from_pacs"
                / "case_from_pacs_RCA.nii.gz"
            )
            _touch(
                root / "case_from_pacs"
                / "case_from_pacs_LAD.nii.gz"
            )
            _touch(
                root / "case_from_pacs"
                / "case_from_pacs_LAD_resampled.csv"
            )
            records = module.discover_xsection_inputs(
                _config(root, Path(temporary) / "out")
            )
            self.assertEqual(
                [record.artery for record in records],
                ["RCA", "LAD"],
            )
            self.assertTrue(
                all(record.layout == "prepared" for record in records)
            )

    def test_flat_layout_and_artery_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "flat"
            _touch(root / "case_1_RCA.nii.gz")
            _touch(root / "case_1_LCX.nii.gz")
            records = module.discover_xsection_inputs(
                _config(
                    root,
                    Path(temporary) / "out",
                    selected_arteries=("LCX",),
                )
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].case_id, "case_1")
            self.assertEqual(records[0].artery, "LCX")
            self.assertEqual(records[0].layout, "flat")

    def test_mixed_layout_is_rejected_in_auto_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mixed"
            _touch(root / "flatcase_RCA.nii.gz")
            _touch(
                root / "batch_0" / "nestedcase" / "Preproc"
                / "Xsection" / "mRCACTALI.nii.gz"
            )
            with self.assertRaisesRegex(ValueError, "Ambiguous mixed"):
                module.discover_xsection_inputs(
                    _config(root, Path(temporary) / "out")
                )
            flat_records = module.discover_xsection_inputs(
                _config(
                    root,
                    Path(temporary) / "out",
                    layout="flat",
                )
            )
            self.assertEqual(len(flat_records), 1)

    def test_duplicate_output_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            for batch in ("batch_0", "batch_1"):
                _touch(
                    root / batch / "same_case" / "Preproc"
                    / "Xsection" / "mRCACTALI.nii.gz"
                )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                module.discover_xsection_inputs(
                    _config(root, Path(temporary) / "out")
                )

    def test_prepared_parent_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "prepared"
            _touch(root / "caseA" / "caseB_RCA.nii.gz")
            with self.assertRaisesRegex(
                    ValueError, "does not match"):
                module.discover_xsection_inputs(
                    _config(
                        root,
                        Path(temporary) / "out",
                        layout="prepared",
                    )
                )


class ManifestTests(unittest.TestCase):
    def _read(self, path):
        with Path(path).open(
                "r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_compatible_merge_preserves_rows_and_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "upstream.csv"
            fallback = root / "fallback.csv"
            with manifest.open(
                    "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "case_id", "batch_name", "unrelated"
                    ],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "case_id": "case1",
                        "batch_name": "batch_0",
                        "unrelated": "keep-one",
                    },
                    {
                        "case_id": "case2",
                        "batch_name": "batch_0",
                        "unrelated": "keep-two",
                    },
                ])
            updates = [
                {
                    "case_id": "case1",
                    "batch_name": "batch_0",
                    "case_relative_path": "batch_0/case1",
                    "pt_state": "complete",
                },
                {
                    "case_id": "case3",
                    "batch_name": "batch_9",
                    "case_relative_path": "batch_9/case3",
                    "pt_state": "complete_with_missing",
                },
            ]
            counts = module.merge_pt_manifest_updates(
                manifest, fallback, updates
            )
            self.assertEqual(counts["upstream_count"], 1)
            self.assertEqual(counts["fallback_count"], 1)
            upstream_rows = self._read(manifest)
            self.assertEqual(len(upstream_rows), 2)
            self.assertEqual(
                upstream_rows[0]["unrelated"], "keep-one"
            )
            self.assertEqual(
                upstream_rows[1]["unrelated"], "keep-two"
            )
            self.assertEqual(
                upstream_rows[0]["pt_state"], "complete"
            )
            fallback_rows = self._read(fallback)
            self.assertEqual(len(fallback_rows), 1)
            self.assertEqual(fallback_rows[0]["case_id"], "case3")

    def test_unique_case_fallback_matches_legacy_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "legacy.csv"
            fallback = root / "fallback.csv"
            with manifest.open(
                    "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["case_id", "legacy_value"],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow({
                    "case_id": "legacy_case",
                    "legacy_value": "preserve",
                })
            module.merge_pt_manifest_updates(
                manifest,
                fallback,
                [{
                    "case_id": "legacy_case",
                    "batch_name": "batch_3",
                    "case_relative_path": "batch_3/legacy_case",
                    "pt_state": "complete",
                }],
            )
            rows = self._read(manifest)
            self.assertEqual(rows[0]["legacy_value"], "preserve")
            self.assertEqual(rows[0]["batch_name"], "batch_3")
            self.assertEqual(rows[0]["pt_state"], "complete")
            self.assertFalse(fallback.exists())

    def test_modern_mismatched_batch_does_not_cross_merge(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "upstream.csv"
            fallback = root / "fallback.csv"
            with manifest.open(
                    "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["case_id", "batch_name", "keep"],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow({
                    "case_id": "reused_case",
                    "batch_name": "batch_0",
                    "keep": "upstream",
                })
            counts = module.merge_pt_manifest_updates(
                manifest,
                fallback,
                [{
                    "case_id": "reused_case",
                    "batch_name": "batch_1",
                    "case_relative_path": "batch_1/reused_case",
                    "pt_state": "complete",
                }],
            )
            self.assertEqual(counts["upstream_count"], 0)
            self.assertEqual(counts["fallback_count"], 1)
            upstream = self._read(manifest)
            self.assertNotIn("pt_state", upstream[0])
            self.assertEqual(upstream[0]["keep"], "upstream")
            fallback_rows = self._read(fallback)
            self.assertEqual(
                fallback_rows[0]["batch_name"], "batch_1"
            )

    def test_previous_manifest_reads_use_locks_without_creating_absent_locks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = root / "upstream.csv"
            config = _config(
                root / "input",
                root / "output",
                manifest_path=str(upstream),
            )
            fallback = Path(config.fallback_manifest_path)
            fallback.parent.mkdir(parents=True)
            for path, state in (
                    (upstream, "upstream"),
                    (fallback, "fallback")):
                with path.open(
                        "w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "case_id",
                            "batch_name",
                            "pt_rca_source_state",
                        ],
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    writer.writerow({
                        "case_id": "case1",
                        "batch_name": "batch_0",
                        "pt_rca_source_state": state,
                    })
            record = module.PTInputRecord(
                source_path=str(root / "source.nii.gz"),
                case_id="case1",
                artery="RCA",
                batch_name="batch_0",
                layout="pipeline",
            )

            with mock.patch.object(
                    module,
                    "interprocess_file_lock",
                    wraps=module.interprocess_file_lock) as lock_mock:
                previous = module._previous_manifest_rows(
                    config, (record,)
                )

            self.assertEqual(
                previous[record.key]["pt_rca_source_state"],
                "upstream",
            )
            self.assertEqual(
                lock_mock.call_args_list,
                [
                    mock.call(str(upstream) + ".lock"),
                    mock.call(str(fallback) + ".lock"),
                ],
            )

            missing_config = _config(
                root / "input",
                root / "missing_output",
                manifest_path=str(root / "missing.csv"),
            )
            missing_fallback = Path(
                missing_config.fallback_manifest_path
            )
            with mock.patch.object(
                    module,
                    "interprocess_file_lock",
                    wraps=module.interprocess_file_lock) as lock_mock:
                self.assertEqual(
                    module._previous_manifest_rows(
                        missing_config, (record,)
                    ),
                    {},
                )
            lock_mock.assert_not_called()
            self.assertFalse(
                Path(str(root / "missing.csv") + ".lock").exists()
            )
            self.assertFalse(
                Path(str(missing_fallback) + ".lock").exists()
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TensorTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_exact_reference_mosaic_and_padding(self):
        array = np.arange(
            11 * 20 * 20, dtype=np.int16
        ).reshape(11, 20, 20)
        tensor, sampling = module._array_to_vit_tensor(array)
        self.assertEqual(tuple(tensor.shape), module.VIT_SHAPE)
        self.assertEqual(tensor.dtype, self.torch.float32)
        self.assertTrue(tensor.is_contiguous())
        self.assertTrue(self.torch.equal(tensor[0], tensor[1]))
        self.assertTrue(self.torch.equal(tensor[0], tensor[2]))
        expected_xyz = np.transpose(
            array, (2, 1, 0)
        ).astype(np.float32)
        tiles = np.transpose(
            expected_xyz[2:18, 2:18, ::5], (2, 0, 1)
        )
        self.assertTrue(
            np.array_equal(
                tensor[0, 0:16, 0:16].numpy(), tiles[0]
            )
        )
        self.assertTrue(
            np.array_equal(
                tensor[0, 0:16, 16:32].numpy(), tiles[1]
            )
        )
        self.assertEqual(sampling.sampled_count, 3)
        self.assertEqual(sampling.used_count, 3)
        self.assertEqual(sampling.padded_count, 193)
        self.assertEqual(sampling.truncated_count, 0)
        self.assertEqual(
            int(self.torch.count_nonzero(tensor[0, 16:, 48:])),
            0,
        )

    def test_long_volume_is_truncated_at_196(self):
        array = np.zeros((1000, 16, 16), dtype=np.float32)
        for index in range(1000):
            array[index, :, :] = index
        tensor, sampling = module._array_to_vit_tensor(array)
        self.assertEqual(sampling.sampled_count, 200)
        self.assertEqual(sampling.used_count, 196)
        self.assertEqual(sampling.padded_count, 0)
        self.assertEqual(sampling.truncated_count, 4)
        last_tile = tensor[0, 13 * 16:14 * 16, 13 * 16:14 * 16]
        self.assertTrue(
            self.torch.all(last_tile == 975).item()
        )

    def test_patch_volume_uses_nine_linear_neighbours(self):
        mosaic = self.torch.zeros((224, 224), dtype=self.torch.float32)
        for index in range(196):
            row, column = divmod(index, 14)
            mosaic[
                row * 16:(row + 1) * 16,
                column * 16:(column + 1) * 16,
            ] = float(index + 1)
        tensor = mosaic.unsqueeze(0).repeat(3, 1, 1).contiguous()
        patches = module.vit_tensor_to_patch_volume(tensor)
        self.assertEqual(tuple(patches.shape), module.PATCH_SHAPE)
        self.assertTrue(patches.is_contiguous())
        first_values = [
            float(
                patches[
                    (position // 3) * 16,
                    (position % 3) * 16,
                    0,
                ]
            )
            for position in range(9)
        ]
        self.assertEqual(
            first_values,
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        last_values = [
            float(
                patches[
                    (position // 3) * 16,
                    (position % 3) * 16,
                    195,
                ]
            )
            for position in range(9)
        ]
        self.assertEqual(
            last_values,
            [192.0, 193.0, 194.0, 195.0, 196.0,
             0.0, 0.0, 0.0, 0.0],
        )

    def test_nifti_api_reads_simpleitk_zyx(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.nii.gz"
            array = np.arange(
                21 * 18 * 20, dtype=np.int16
            ).reshape(21, 18, 20)
            _write_nifti(source, array)
            from_file = module.nifti_to_vit_tensor(source)
            direct, _sampling = module._array_to_vit_tensor(array)
            self.assertTrue(
                self.torch.equal(from_file, direct)
            )

    def test_atomic_save_failure_preserves_original_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "case_RCA.pt"
            original = self.torch.zeros(
                module.VIT_SHAPE, dtype=self.torch.float32
            ).contiguous()
            replacement = self.torch.ones(
                module.VIT_SHAPE, dtype=self.torch.float32
            ).contiguous()
            self.torch.save(original, str(destination))

            with mock.patch.object(
                    module,
                    "_replace_with_retry",
                    side_effect=PermissionError("destination locked")):
                with self.assertRaises(PermissionError):
                    module._atomic_save_tensor(
                        replacement, destination, "vit"
                    )

            published = self.torch.load(
                str(destination), map_location="cpu"
            )
            self.assertTrue(self.torch.equal(published, original))
            self.assertEqual(list(root.glob(".*.tmp")), [])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class EndToEndRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_both_formats_manifest_qc_progress_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            array = np.arange(
                41 * 20 * 20, dtype=np.int16
            ).reshape(41, 20, 20)
            _write_nifti(
                input_root / "case_from_pacs_RCA.nii.gz",
                array,
            )
            _write_nifti(
                input_root / "case_from_pacs_LAD.nii.gz",
                array + 3,
            )
            config = _config(
                input_root,
                output_root,
                write_patch=True,
                save_qc=True,
                qc_count=2,
                checkpoint_cases=1,
            )
            progress = []
            result = module.run_pt_preparation(
                config, progress_callback=progress.append
            )
            self.assertEqual(result.saved_count, 4)
            self.assertEqual(result.skipped_count, 0)
            self.assertEqual(result.failed_count, 0)
            self.assertEqual(result.missing_count, 2)
            self.assertEqual(progress[-1].percent, 100)
            self.assertEqual(
                [item.percent for item in progress],
                sorted(item.percent for item in progress),
            )
            vit_path = (
                output_root / module.VIT_OUTPUT_FOLDER
                / "case_from_pacs_RCA.pt"
            )
            patch_path = (
                output_root / module.PATCH_OUTPUT_FOLDER
                / "case_from_pacs_RCA.pt"
            )
            self.assertTrue(
                module.validate_pt_output(vit_path, "vit")
            )
            self.assertTrue(
                module.validate_pt_output(patch_path, "patch")
            )
            self.assertEqual(
                tuple(self.torch.load(str(vit_path)).shape),
                module.VIT_SHAPE,
            )
            self.assertEqual(
                tuple(self.torch.load(str(patch_path)).shape),
                module.PATCH_SHAPE,
            )
            self.assertTrue(
                (
                    output_root / module.VIT_OUTPUT_FOLDER
                    / "_qc"
                    / "case_from_pacs_RCA_mosaic.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    output_root / module.PATCH_OUTPUT_FOLDER
                    / "_qc"
                    / "case_from_pacs_RCA_patch_contact_sheet.png"
                ).is_file()
            )
            fallback = Path(result.fallback_manifest_path)
            with fallback.open(
                    "r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["pt_lcx_source_state"], "missing"
            )
            self.assertEqual(
                rows[0]["pt_vit_rca_complete"], "1"
            )
            self.assertEqual(
                rows[0]["pt_patch_rca_complete"], "1"
            )
            second = module.run_pt_preparation(config)
            self.assertEqual(second.saved_count, 0)
            self.assertEqual(second.skipped_count, 4)
            self.assertEqual(second.failed_count, 0)

    def test_zero_ctali_pipeline_case_is_reported_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "batch_0"
            (
                input_root / "empty_case" / "Preproc" / "Xsection"
            ).mkdir(parents=True)
            result = module.run_pt_preparation(
                _config(input_root, root / "output")
            )
            self.assertEqual(result.case_count, 1)
            self.assertEqual(result.source_count, 0)
            self.assertEqual(result.missing_count, 3)
            self.assertEqual(result.failed_count, 0)

    def test_progress_heartbeat_updates_elapsed_time_while_source_is_active(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            _write_nifti(
                input_root / "case_RCA.nii.gz",
                np.ones((21, 16, 16), dtype=np.int16),
            )
            progress = []
            original_process_record = module._process_record

            def slow_process_record(*args, **kwargs):
                time.sleep(0.04)
                return original_process_record(*args, **kwargs)

            with mock.patch.object(
                    module, "PROGRESS_HEARTBEAT_SECONDS", 0.01), mock.patch.object(
                        module,
                        "_process_record",
                        side_effect=slow_process_record):
                result = module.run_pt_preparation(
                    _config(
                        input_root,
                        root / "output",
                        selected_arteries=("RCA",),
                    ),
                    progress_callback=progress.append,
                )

            self.assertEqual(result.saved_count, 1)
            self.assertTrue(
                any(
                    item.completed_source_files == 0
                    and item.active_workers == 1
                    and item.elapsed_seconds > 0
                    for item in progress[1:]
                )
            )
            self.assertEqual(
                [item.percent for item in progress],
                sorted(item.percent for item in progress),
            )

    def test_invalid_existing_output_is_regenerated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            _write_nifti(
                input_root / "case_RCA.nii.gz",
                np.zeros((21, 16, 16), dtype=np.int16),
            )
            output_path = (
                output_root / module.VIT_OUTPUT_FOLDER
                / "case_RCA.pt"
            )
            _touch(output_path, b"not a tensor")
            os.utime(output_path, (time.time() + 2, time.time() + 2))
            result = module.run_pt_preparation(
                _config(
                    input_root,
                    output_root,
                    selected_arteries=("RCA",),
                )
            )
            self.assertEqual(result.saved_count, 1)
            self.assertTrue(
                module.validate_pt_output(output_path, "vit")
            )
            temporary_files = list(
                output_path.parent.glob(".*.tmp")
            )
            self.assertEqual(temporary_files, [])

    def test_valid_but_unproven_output_is_regenerated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            _write_nifti(
                input_root / "case_RCA.nii.gz",
                np.ones((21, 16, 16), dtype=np.int16),
            )
            output_path = (
                output_root / module.VIT_OUTPUT_FOLDER
                / "case_RCA.pt"
            )
            output_path.parent.mkdir(parents=True)
            stale = self.torch.zeros(
                module.VIT_SHAPE, dtype=self.torch.float32
            )
            self.torch.save(stale, str(output_path))
            os.utime(
                output_path,
                (time.time() + 2, time.time() + 2),
            )
            self.assertTrue(
                module.validate_pt_output(output_path, "vit")
            )
            result = module.run_pt_preparation(
                _config(
                    input_root,
                    output_root,
                    selected_arteries=("RCA",),
                )
            )
            self.assertEqual(result.saved_count, 1)
            actual = self.torch.load(
                str(output_path), map_location="cpu"
            )
            self.assertFalse(self.torch.equal(actual, stale))

    def test_failed_rerun_clears_stale_sampling_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            source = _write_nifti(
                input_root / "case_RCA.nii.gz",
                np.ones((21, 16, 16), dtype=np.int16),
            )
            config = _config(
                input_root,
                output_root,
                selected_arteries=("RCA",),
                checkpoint_cases=1,
            )
            first = module.run_pt_preparation(config)
            self.assertEqual(first.saved_count, 1)
            source.write_bytes(b"corrupt replacement")
            second = module.run_pt_preparation(config)
            self.assertEqual(second.failed_count, 1)
            with Path(second.fallback_manifest_path).open(
                    "r", encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["pt_rca_sampled_count"], "")
            self.assertEqual(row["pt_rca_used_count"], "")
            self.assertEqual(row["pt_rca_padded_count"], "")
            self.assertEqual(row["pt_rca_truncated_count"], "")

    def test_corrupt_source_isolated_as_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            _touch(input_root / "bad_RCA.nii.gz", b"bad")
            _write_nifti(
                input_root / "good_RCA.nii.gz",
                np.ones((21, 16, 16), dtype=np.int16),
            )
            result = module.run_pt_preparation(
                _config(
                    input_root,
                    output_root,
                    selected_arteries=("RCA",),
                )
            )
            self.assertEqual(result.failed_count, 1)
            self.assertEqual(result.saved_count, 1)
            self.assertFalse(
                (
                    output_root / module.VIT_OUTPUT_FOLDER
                    / "bad_RCA.pt"
                ).exists()
            )

    def test_changing_source_is_deferred_without_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _touch(root / "case_RCA.nii.gz")
            output = root / "out"
            config = _config(
                root,
                output,
                selected_arteries=("RCA",),
            )
            record = module.PTInputRecord(
                source_path=str(source),
                case_id="case",
                artery="RCA",
                batch_name="",
                layout="flat",
            )
            tensor = self.torch.zeros(
                module.VIT_SHAPE, dtype=self.torch.float32
            )
            sampling = module.TensorSamplingInfo(1, 1, 195, 0)
            with mock.patch.object(
                    module,
                    "_source_signature",
                    side_effect=[(5, 10), (5, 11)]), mock.patch.object(
                        module,
                        "_read_nifti_to_vit",
                        return_value=(tensor, sampling)):
                outcome = module._process_record(
                    record, config
                )
            self.assertEqual(outcome.source_state, "deferred")
            self.assertFalse(
                (
                    output / module.VIT_OUTPUT_FOLDER
                    / "case_RCA.pt"
                ).exists()
            )


class CliConsoleCompatibilityTests(unittest.TestCase):
    def test_cli_output_remains_unprefixed_and_not_duplicated(self):
        snapshot = module.PTProgressSnapshot(
            total_source_files=1,
            completed_source_files=1,
            total_cases=1,
            completed_cases=1,
            saved_outputs=1,
            skipped_outputs=0,
            failed_outputs=0,
            missing_outputs=0,
            deferred_outputs=0,
            active_workers=0,
            elapsed_seconds=1.0,
            eta_seconds=0.0,
            percent=100,
        )
        result = module.PTRunResult(
            input_layout="flat",
            source_count=1,
            case_count=1,
            saved_count=1,
            skipped_count=0,
            failed_count=0,
            missing_count=0,
            deferred_count=0,
            source_outcomes=(),
            manifest_path="",
            fallback_manifest_path="fallback.csv",
            log_path="run.log",
            duration_seconds=1.0,
        )

        def fake_run(
                config, progress_callback=None, log_callback=None):
            del config
            log_callback("CLI source result")
            progress_callback(snapshot)
            return result

        output = io.StringIO()
        with mock.patch.object(
                module,
                "run_pt_preparation",
                side_effect=fake_run), redirect_stdout(output):
            return_code = module.main([
                "--input-folder", "input",
                "--output-parent", "output",
                "--manifest", "",
            ])
        text = output.getvalue()
        self.assertEqual(return_code, 0)
        self.assertEqual(text.count("CLI source result"), 1)
        self.assertEqual(text.count("100% files 1/1"), 1)
        self.assertNotIn("Batch PT log |", text)
        self.assertNotIn("Batch PT config |", text)


@unittest.skipUnless(
    QT_TESTS_AVAILABLE and TORCH_AVAILABLE,
    "PySide2 and PyTorch are required for Qt controller tests",
)
class QtControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = (
            QApplication.instance() or QApplication([])
        )

    def _load_page(self):
        page = QUiLoader().load(
            str(GUI_ROOT / "ui" / "BatchPTDataPrepare.ui")
        )
        self.assertIsNotNone(page)
        return page

    def _process_events_until(self, predicate, timeout=10.0):
        deadline = time.time() + timeout
        while not predicate() and time.time() < deadline:
            self.application.processEvents()
            time.sleep(0.005)
        self.application.processEvents()
        self.assertTrue(predicate(), "Timed out waiting for Qt worker state")

    def test_ui_defaults_and_controller_mapping(self):
        page = self._load_page()
        controller = module.BatchPTDataPrepare(UI=page)
        config = controller._config_from_ui()
        self.assertEqual(config.input_folder, module.DEFAULT_INPUT_FOLDER)
        self.assertEqual(config.output_parent, module.DEFAULT_OUTPUT_PARENT)
        self.assertEqual(config.manifest_path, module.DEFAULT_MANIFEST_PATH)
        self.assertEqual(config.layout, "auto")
        self.assertEqual(config.selected_arteries, module.ARTERIES)
        self.assertTrue(config.write_vit)
        self.assertFalse(config.write_patch)
        self.assertEqual(config.max_workers, 4)
        self.assertFalse(config.save_qc)
        self.assertEqual(config.qc_count, 3)
        self.assertTrue(page.Log_Txt_BPTDP.isReadOnly())
        self.assertEqual(
            page.Log_Txt_BPTDP.document().maximumBlockCount(),
            500,
        )
        controller.deleteLater()
        page.deleteLater()
        self.application.processEvents()

    def test_background_run_progress_and_button_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            _write_nifti(
                input_root / "case_RCA.nii.gz",
                np.ones((21, 16, 16), dtype=np.int16),
            )
            page = self._load_page()
            page.InputFolderPath_Txt_BPTDP.setText(str(input_root))
            page.OutputFolderPath_Txt_BPTDP.setText(
                str(root / "output")
            )
            page.ManifestPath_Txt_BPTDP.clear()
            page.LAD_CB_BPTDP.setChecked(False)
            page.LCX_CB_BPTDP.setChecked(False)
            controller = module.BatchPTDataPrepare(UI=page)
            console = _FlushRecordingBuffer()
            with redirect_stdout(console):
                controller.startRun()
                self.assertFalse(page.Run_Btn_BPTDP.isEnabled())
                self._process_events_until(
                    lambda: controller._run_thread is None
                )
            result = controller._last_run_result
            self.assertIsNotNone(result)
            self.assertFalse(result.cancelled)
            self.assertEqual(result.saved_count, 1)
            self.assertEqual(page.Progress_PB_BPTDP.value(), 100)
            self.assertTrue(page.Run_Btn_BPTDP.isEnabled())
            self.assertEqual(
                page.Log_Txt_BPTDP.toPlainText().count(
                    result.summary()
                ),
                1,
            )
            console_text = console.getvalue()
            console_lines = console_text.splitlines()
            self.assertTrue(console_lines)
            self.assertRegex(
                console_lines[0],
                r"^\[\d{4}-\d{2}-\d{2}T.*Z\] "
                r"Batch PT config \| ",
            )
            self.assertIn(
                "input={0!r}".format(str(input_root)),
                console_lines[0],
            )
            self.assertIn(
                "output={0!r}".format(str(root / "output")),
                console_lines[0],
            )
            self.assertIn(
                "manifest={0!r}".format(
                    str(
                        root
                        / "output"
                        / module.VIT_OUTPUT_FOLDER
                        / module.FALLBACK_MANIFEST_NAME
                    )
                ),
                console_lines[0],
            )
            self.assertIn("layout=auto", console_lines[0])
            self.assertIn("arteries=RCA", console_lines[0])
            self.assertIn("formats=vit", console_lines[0])
            self.assertIn("workers=4", console_lines[0])
            self.assertIn("QC=off", console_lines[0])
            self.assertEqual(console_text.count(result.summary()), 1)
            self.assertGreaterEqual(
                console_text.count("Batch PT log |"), 3
            )
            self.assertGreater(console.flush_count, 0)
            controller.deleteLater()
            page.deleteLater()
            self.application.processEvents()

    def test_console_progress_is_detailed_and_throttled(self):
        page = self._load_page()
        controller = module.BatchPTDataPrepare(UI=page)

        def snapshot(completed, active, percent, elapsed):
            return module.PTProgressSnapshot(
                total_source_files=2,
                completed_source_files=completed,
                total_cases=2,
                completed_cases=completed,
                saved_outputs=completed,
                skipped_outputs=0,
                failed_outputs=0,
                missing_outputs=0,
                deferred_outputs=0,
                active_workers=active,
                elapsed_seconds=elapsed,
                eta_seconds=max(0.0, 20.0 - elapsed),
                percent=percent,
            )

        console = _FlushRecordingBuffer()
        monotonic_values = [0.0, 5.0, 10.0, 11.0, 12.0, 25.0]
        with mock.patch.object(
                module, "_utc_now",
                return_value="2026-10-31T12:34:56Z"), mock.patch.object(
                    module.time,
                    "monotonic",
                    side_effect=monotonic_values), redirect_stdout(console):
            controller._on_progress(snapshot(0, 2, 0, 0.0))
            controller._on_progress(snapshot(0, 2, 0, 5.0))
            controller._on_progress(snapshot(0, 2, 0, 10.0))
            controller._on_progress(snapshot(1, 1, 50, 11.0))
            controller._on_progress(snapshot(2, 0, 100, 12.0))
            controller._on_progress(snapshot(2, 0, 100, 25.0))

        lines = console.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(
            line.startswith(
                "[2026-10-31T12:34:56Z] Batch PT progress | "
            )
            for line in lines
        ))
        self.assertEqual(
            sum("files 0/2" in line for line in lines), 2
        )
        self.assertEqual(
            sum("files 1/2" in line for line in lines), 1
        )
        self.assertEqual(
            sum("files 2/2" in line for line in lines), 1
        )
        self.assertIn("saved 2, skipped 0", lines[-1])
        self.assertIn("active 0", lines[-1])
        self.assertIn("100%", lines[-1])
        self.assertEqual(console.flush_count, 4)
        self.assertEqual(page.Progress_PB_BPTDP.value(), 100)
        controller.deleteLater()
        page.deleteLater()
        self.application.processEvents()

    def test_ui_log_is_mirrored_to_timestamped_flushed_console(self):
        page = self._load_page()
        controller = module.BatchPTDataPrepare(UI=page)
        console = _FlushRecordingBuffer()
        with mock.patch.object(
                module,
                "_utc_now",
                return_value="2026-10-31T12:34:56Z"), redirect_stdout(
                    console):
            controller._append_log("case_RCA: vit=saved")
        self.assertEqual(
            console.getvalue(),
            "[2026-10-31T12:34:56Z] Batch PT log | "
            "case_RCA: vit=saved\n",
        )
        self.assertEqual(console.flush_count, 1)
        self.assertIn(
            "case_RCA: vit=saved",
            page.Log_Txt_BPTDP.toPlainText(),
        )
        controller.deleteLater()
        page.deleteLater()
        self.application.processEvents()

    def test_console_diagnostics_tolerate_missing_or_closed_stdout(self):
        page = self._load_page()
        controller = module.BatchPTDataPrepare(UI=page)
        with mock.patch.object(module.sys, "stdout", None):
            controller._append_log("stdout is unavailable")
        closed_stream = io.StringIO()
        closed_stream.close()
        with mock.patch.object(module.sys, "stdout", closed_stream):
            controller._append_log("stdout is closed")
        ui_text = page.Log_Txt_BPTDP.toPlainText()
        self.assertIn("stdout is unavailable", ui_text)
        self.assertIn("stdout is closed", ui_text)
        controller.deleteLater()
        page.deleteLater()
        self.application.processEvents()

    def test_failure_waits_for_thread_cleanup_before_enabling_run(self):
        page = self._load_page()
        controller = module.BatchPTDataPrepare(UI=page)
        page.Run_Btn_BPTDP.setEnabled(False)
        controller._run_thread = object()
        controller._on_failed("expected test failure")
        self.assertFalse(page.Run_Btn_BPTDP.isEnabled())
        controller._on_thread_finished()
        self.assertTrue(page.Run_Btn_BPTDP.isEnabled())
        controller.deleteLater()
        page.deleteLater()
        self.application.processEvents()

    def test_shutdown_cancels_queued_work_and_waits_safely(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            for index in range(12):
                _write_nifti(
                    input_root / "case_{0:02d}_RCA.nii.gz".format(index),
                    np.ones((21, 16, 16), dtype=np.int16) * index,
                )
            page = self._load_page()
            page.InputFolderPath_Txt_BPTDP.setText(str(input_root))
            page.OutputFolderPath_Txt_BPTDP.setText(
                str(root / "output")
            )
            page.ManifestPath_Txt_BPTDP.clear()
            page.LAD_CB_BPTDP.setChecked(False)
            page.LCX_CB_BPTDP.setChecked(False)
            controller = module.BatchPTDataPrepare(UI=page)
            original_process_record = module._process_record
            source_started = threading.Event()

            def slow_process_record(*args, **kwargs):
                source_started.set()
                time.sleep(0.05)
                return original_process_record(*args, **kwargs)

            with mock.patch.object(
                    module,
                    "_process_record",
                    side_effect=slow_process_record):
                controller.startRun()
                self._process_events_until(source_started.is_set)
                self.assertTrue(controller.shutdown())
            self.application.processEvents()
            self.assertIsNone(controller._run_thread)
            self.assertIsNotNone(controller._last_run_result)
            self.assertTrue(controller._last_run_result.cancelled)
            self.assertLess(
                len(controller._last_run_result.source_outcomes),
                12,
            )
            self.assertFalse(page.Run_Btn_BPTDP.isEnabled())
            controller.deleteLater()
            page.deleteLater()
            self.application.processEvents()

    def test_main_selector_matches_pre_sort_stack_ordinal(self):
        main_page = QUiLoader().load(str(GUI_ROOT / "ui" / "main.ui"))
        self.assertIsNotNone(main_page)
        selector = main_page.QComboBox_Modules
        platform_source = (
            GUI_ROOT / "HemodynamicsPlatform.py"
        ).read_text(encoding="utf-8")
        marker = (
            "self.ui.QStackedWidget_Module.addWidget("
            "self.BatchPTDataPrepareui.centralwidget)"
        )
        marker_offset = platform_source.index(marker)
        add_widget_text = "QStackedWidget_Module.addWidget"
        expected_index = platform_source[:marker_offset].count(
            add_widget_text
        )
        self.assertEqual(
            selector.findText("Batch PT Data Prepare"),
            expected_index,
        )
        self.assertEqual(
            selector.count(),
            platform_source.count(add_widget_text),
        )
        main_page.deleteLater()
        self.application.processEvents()


@unittest.skipUnless(
    TORCH_AVAILABLE
    and os.environ.get("HYMO_RUN_PT_GOLDEN") == "1"
    and Path(r"H:\FishAndChips\All").is_dir()
    and Path(r"H:\FishAndChips\All_pt").is_dir(),
    "Set HYMO_RUN_PT_GOLDEN=1 to run reference-drive golden tests",
)
class ReferenceGoldenTests(unittest.TestCase):
    def test_reference_pairs_include_short_and_long_vessels(self):
        import torch
        source_root = Path(r"H:\FishAndChips\All")
        output_root = Path(r"H:\FishAndChips\All_pt")
        expected_depths = (
            ("CASE_cd8059af0335_LAD", 7),
            ("CASE_d391840f595f_LAD", 1015),
            ("CASE_cb244a65f2f8_RCA", 2486),
        )
        for base_name, expected_depth in expected_depths:
            source = source_root / (base_name + ".nii.gz")
            reference = output_root / (base_name + ".pt")
            self.assertTrue(source.is_file())
            self.assertTrue(reference.is_file())
            self.assertEqual(
                sitk.ReadImage(str(source)).GetSize()[2],
                expected_depth,
            )
            actual = module.nifti_to_vit_tensor(source)
            expected = torch.load(
                str(reference), map_location="cpu"
            )
            self.assertTrue(
                torch.equal(actual, expected),
                msg="Golden mismatch for {0}".format(source),
            )


if __name__ == "__main__":
    unittest.main()
