# -*- coding: utf-8 -*-
"""Focused regression tests for Batch PT Data Prepare."""

import csv
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import SimpleITK as sitk

try:
    from . import BatchPTDataPrepareMain as module
except ImportError:
    import BatchPTDataPrepareMain as module


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


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
