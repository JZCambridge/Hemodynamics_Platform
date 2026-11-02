import csv
import gzip
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import xml.etree.ElementTree as ElementTree


MODULE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIRECTORY.parents[2]
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from BatchNIIDataPrepareMain import (  # noqa: E402
    BatchNIIDataPrepare,
    CANONICAL_CTA_RELATIVE_PATH,
    CTA_DESTINATION_RELATIVE_PATHS,
    DEFAULT_CASE_LIMIT,
    DEFAULT_CTA_DESTINATIONS,
    DEFAULT_PARALLEL_WORKERS,
    MAX_PARALLEL_WORKERS,
    MANIFEST_FILE_NAME,
    OUTPUT_FOLDER_NAME,
    PIPELINE_CTA_RELATIVE_PATHS,
    XSECTION_STATUS_FIELDS,
    build_case_plan,
    prepare_batch_data as _prepare_batch_data,
    sanitize_case_id,
    sha256_file,
)


def prepare_batch_data(*args, **kwargs):
    """Retain byte-copy semantics for the pre-existing unit-test fixtures."""

    kwargs.setdefault("normalize_for_segmentation", False)
    return _prepare_batch_data(*args, **kwargs)


class StubBatchPreproc:
    """Small stand-in that exposes the same required CreateFolders API."""

    def __init__(self):
        self.TableDir = None
        self.case_paths = []
        self.table_paths = []

    def CreateFolders(self, CasePath=None):
        case_path = Path(CasePath)
        self.case_paths.append(case_path)
        table_path = Path(self.TableDir)
        self.table_paths.append(table_path)
        table_path.mkdir(parents=True, exist_ok=True)
        for _, relative_path in PIPELINE_CTA_RELATIVE_PATHS:
            (case_path / relative_path).parent.mkdir(parents=True, exist_ok=True)


def write_source(folder, name, content):
    path = Path(folder) / name
    path.write_bytes(content)
    return path


def write_test_nifti(folder, name, orientation="LPS", scaled_hu=False):
    """Write a small deterministic CTA, optionally with a -1024 NIfTI intercept."""

    import numpy
    import SimpleITK as sitk

    stored = numpy.arange(24, dtype=numpy.int16).reshape(2, 3, 4) + 1000
    image = sitk.GetImageFromArray(stored)
    image.SetSpacing((0.7, 0.8, 1.2))
    if orientation == "LAS":
        image.SetDirection(
            (1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
        )
    elif orientation != "LPS":
        raise ValueError("Unsupported test orientation: {0}".format(orientation))
    path = Path(folder) / name
    sitk.WriteImage(image, str(path), True)
    if scaled_hu:
        payload = bytearray(gzip.decompress(path.read_bytes()))
        struct.pack_into("<f", payload, 112, 1.0)
        struct.pack_into("<f", payload, 116, -1024.0)
        path.write_bytes(gzip.compress(bytes(payload)))
    return path


class BatchNIIDataPrepareTests(unittest.TestCase):
    def test_ui_defaults_to_first_30_and_uses_generic_segmentation_wording(self):
        ui_path = PROJECT_ROOT / "GUI_V0" / "ui" / "BatchNIIDataPrepare.ui"
        ui_root = ElementTree.parse(str(ui_path)).getroot()
        case_limit_widget = ui_root.find(
            ".//widget[@name='CaseLimit_SB_BNDP']"
        )
        self.assertIsNotNone(case_limit_widget)
        case_limit_value = case_limit_widget.find(
            "./property[@name='value']/number"
        )
        self.assertIsNotNone(case_limit_value)
        self.assertEqual(case_limit_value.text, "30")
        limit_cases_widget = ui_root.find(
            ".//widget[@name='LimitCases_CB_BNDP']"
        )
        self.assertIsNotNone(limit_cases_widget)
        limit_cases_checked = limit_cases_widget.find(
            "./property[@name='checked']/bool"
        )
        self.assertIsNotNone(limit_cases_checked)
        self.assertEqual(limit_cases_checked.text, "true")

        parallel_workers_widget = ui_root.find(
            ".//widget[@name='ParallelWorkers_SB_BNDP']"
        )
        self.assertIsNotNone(parallel_workers_widget)
        parallel_minimum = parallel_workers_widget.find(
            "./property[@name='minimum']/number"
        )
        parallel_maximum = parallel_workers_widget.find(
            "./property[@name='maximum']/number"
        )
        parallel_default = parallel_workers_widget.find(
            "./property[@name='value']/number"
        )
        parallel_tooltip = parallel_workers_widget.find(
            "./property[@name='toolTip']/string"
        )
        self.assertEqual(parallel_minimum.text, "1")
        self.assertEqual(parallel_maximum.text, "8")
        self.assertEqual(parallel_default.text, "2")
        self.assertIn("mechanical s:", parallel_tooltip.text.casefold())
        self.assertIn("may be slower", parallel_tooltip.text.casefold())

        visible_text = " ".join(
            node.text or "" for node in ui_root.findall(".//string")
        ).casefold()
        self.assertIn("segmentation", visible_text)
        self.assertNotIn("tenoke", visible_text)

    def test_case_limit_zero_preserves_all_sorted_cases(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            file_names = (
                "case_z.nii.gz",
                "CASE_a.nii.gz",
                "Case_B.nii.gz",
                "case_c.nii.gz",
            )
            for file_name in file_names:
                write_source(input_path, file_name, file_name.encode("ascii"))

            default_plans = build_case_plan(input_path, output_path)
            explicit_all_plans = build_case_plan(
                input_path,
                output_path,
                case_limit=DEFAULT_CASE_LIMIT,
            )
            expected_names = sorted(
                file_names, key=lambda name: (name.casefold(), name)
            )

            self.assertEqual(
                [plan.source_path.name for plan in default_plans],
                expected_names,
            )
            self.assertEqual(default_plans, explicit_all_plans)
            self.assertEqual(
                [plan.rank for plan in explicit_all_plans],
                list(range(1, len(file_names) + 1)),
            )

    def test_case_limit_30_prepares_only_first_30_sorted_cases(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            file_names = [
                "CASE_{0:03d}.nii.gz".format(index)
                for index in reversed(range(35))
            ]
            for file_name in file_names:
                write_source(input_path, file_name, file_name.encode("ascii"))

            result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("canonical",),
                case_limit=30,
            )
            expected_names = sorted(
                file_names, key=lambda name: (name.casefold(), name)
            )[:30]
            with result.manifest_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(result.case_count, 30)
            self.assertEqual(
                [row["source_file"] for row in rows], expected_names
            )
            self.assertEqual(
                [int(row["rank"]) for row in rows], list(range(1, 31))
            )
            batch_path = output_path / OUTPUT_FOLDER_NAME / "batch_0"
            for file_name in expected_names:
                case_id = sanitize_case_id(file_name)
                self.assertTrue(
                    (batch_path / case_id / CANONICAL_CTA_RELATIVE_PATH).is_file()
                )
            for file_name in sorted(
                    file_names, key=lambda name: (name.casefold(), name))[30:]:
                self.assertFalse((batch_path / sanitize_case_id(file_name)).exists())

    def test_invalid_case_limit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            write_source(input_path, "CASE_001.nii.gz", b"source")

            with self.assertRaisesRegex(ValueError, "Case limit"):
                build_case_plan(input_path, output_path, case_limit=-1)
            for invalid_limit in (True, 1.5, "1"):
                with self.subTest(case_limit=invalid_limit):
                    with self.assertRaisesRegex(TypeError, "Case limit"):
                        build_case_plan(
                            input_path,
                            output_path,
                            case_limit=invalid_limit,
                        )

    def test_invalid_parallel_worker_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            input_path.mkdir()
            write_source(input_path, "CASE_001.nii.gz", b"source")

            for invalid_workers in (0, MAX_PARALLEL_WORKERS + 1):
                with self.subTest(parallel_workers=invalid_workers):
                    with self.assertRaisesRegex(ValueError, "Parallel workers"):
                        prepare_batch_data(
                            input_path,
                            temporary_path / "output",
                            create_pipeline_folders=False,
                            cta_destinations=("canonical",),
                            parallel_workers=invalid_workers,
                        )
            for invalid_workers in (True, 1.5, "2"):
                with self.subTest(parallel_workers=invalid_workers):
                    with self.assertRaisesRegex(TypeError, "Parallel workers"):
                        prepare_batch_data(
                            input_path,
                            temporary_path / "output",
                            create_pipeline_folders=False,
                            cta_destinations=("canonical",),
                            parallel_workers=invalid_workers,
                        )

    def test_controller_without_case_limit_widget_defaults_to_all(self):
        controller = object.__new__(BatchNIIDataPrepare)
        controller.ui = SimpleNamespace(
            Summary_Txt_BNDP=SimpleNamespace(setPlainText=mock.Mock())
        )
        controller.modelui = None
        controller.batch_preproc = None
        controller.UpdateMsgLog = mock.Mock()
        expected_result = SimpleNamespace(summary=lambda: "prepared")

        with mock.patch(
                "BatchNIIDataPrepareMain.prepare_batch_data",
                return_value=expected_result) as prepare_mock:
            actual_result = controller.PrepareData(
                input_folder="input",
                output_folder="output",
                create_folders=False,
                cases_per_batch=100,
                cta_destinations=("canonical",),
                normalize_for_segmentation=False,
            )

        self.assertIs(actual_result, expected_result)
        self.assertEqual(
            prepare_mock.call_args[1]["case_limit"], DEFAULT_CASE_LIMIT
        )
        self.assertEqual(
            prepare_mock.call_args[1]["parallel_workers"],
            DEFAULT_PARALLEL_WORKERS,
        )
        self.assertIs(
            prepare_mock.call_args[1]["progress_callback"].__self__,
            controller,
        )

    def test_controller_limit_checkbox_selects_first_n_or_all_cases(self):
        expected_result = SimpleNamespace(summary=lambda: "prepared")
        for checked, expected_limit in ((True, 30), (False, 0)):
            with self.subTest(checked=checked):
                controller = object.__new__(BatchNIIDataPrepare)
                controller.ui = SimpleNamespace(
                    LimitCases_CB_BNDP=SimpleNamespace(
                        isChecked=mock.Mock(return_value=checked)
                    ),
                    CaseLimit_SB_BNDP=SimpleNamespace(
                        value=mock.Mock(return_value=30)
                    ),
                    Summary_Txt_BNDP=SimpleNamespace(
                        setPlainText=mock.Mock()
                    ),
                )
                controller.modelui = None
                controller.batch_preproc = None
                controller.UpdateMsgLog = mock.Mock()

                with mock.patch(
                        "BatchNIIDataPrepareMain.prepare_batch_data",
                        return_value=expected_result) as prepare_mock:
                    result = controller.PrepareData(
                        input_folder="input",
                        output_folder="output",
                        create_folders=False,
                        cases_per_batch=100,
                        cta_destinations=("canonical",),
                        normalize_for_segmentation=False,
                    )

                self.assertIs(result, expected_result)
                self.assertEqual(
                    prepare_mock.call_args[1]["case_limit"], expected_limit
                )

    def test_controller_checkbox_toggles_case_limit_spinbox(self):
        clicked = SimpleNamespace(connect=mock.Mock())
        toggled = SimpleNamespace(connect=mock.Mock())
        case_limit_widget = SimpleNamespace(setEnabled=mock.Mock())
        limit_widget = SimpleNamespace(
            toggled=toggled,
            isChecked=mock.Mock(return_value=True),
        )
        ui = SimpleNamespace(
            ChooseInputFolder_Btn_BNDP=SimpleNamespace(clicked=clicked),
            ChooseOutputFolder_Btn_BNDP=SimpleNamespace(clicked=clicked),
            Prepare_Btn_BNDP=SimpleNamespace(clicked=clicked),
            LimitCases_CB_BNDP=limit_widget,
            CaseLimit_SB_BNDP=case_limit_widget,
        )

        BatchNIIDataPrepare(UI=ui)

        toggled.connect.assert_called_once_with(case_limit_widget.setEnabled)
        case_limit_widget.setEnabled.assert_called_once_with(True)

    def test_controller_reads_parallel_worker_widget(self):
        controller = object.__new__(BatchNIIDataPrepare)
        controller.ui = SimpleNamespace(
            ParallelWorkers_SB_BNDP=SimpleNamespace(
                value=mock.Mock(return_value=6)
            ),
            Summary_Txt_BNDP=SimpleNamespace(setPlainText=mock.Mock()),
        )
        controller.modelui = None
        controller.batch_preproc = None
        controller.UpdateMsgLog = mock.Mock()
        expected_result = SimpleNamespace(summary=lambda: "prepared")

        with mock.patch(
                "BatchNIIDataPrepareMain.prepare_batch_data",
                return_value=expected_result) as prepare_mock:
            controller.PrepareData(
                input_folder="input",
                output_folder="output",
                create_folders=False,
                cases_per_batch=100,
                cta_destinations=("canonical",),
                normalize_for_segmentation=False,
                case_limit=0,
            )

        self.assertEqual(
            prepare_mock.call_args[1]["parallel_workers"], 6
        )

    def test_controller_progress_updates_ui_and_log(self):
        progress_label = SimpleNamespace(setText=mock.Mock())
        progress_bar = SimpleNamespace(
            setRange=mock.Mock(),
            setValue=mock.Mock(),
            setFormat=mock.Mock(),
        )
        controller = object.__new__(BatchNIIDataPrepare)
        controller.ui = SimpleNamespace(
            Progress_Label_BNDP=progress_label,
            ProgressBar_BNDP=progress_bar,
        )
        controller.UpdateMsgLog = mock.Mock()

        controller._update_preparation_progress({
            "processed": 4,
            "total": 30,
            "passed": 3,
            "changed": 2,
            "unchanged": 1,
            "failed": 1,
            "excluded": 0,
            "current_case": "CASE_004",
            "status": "failed",
            "message": "Invalid 4-D input",
        })

        expected = (
            "Processed 4/30 | passed 3 | changed 2 | unchanged 1 | "
            "failed 1 | excluded 0 | CASE_004 | failed | Invalid 4-D input"
        )
        progress_label.setText.assert_called_once_with(expected)
        progress_bar.setRange.assert_called_once_with(0, 30)
        progress_bar.setValue.assert_called_once_with(4)
        progress_bar.setFormat.assert_called_once_with("%v/%m cases (%p%)")
        controller.UpdateMsgLog.assert_called_once_with(expected)

    def test_controller_progress_is_compatible_with_legacy_ui(self):
        controller = object.__new__(BatchNIIDataPrepare)
        controller.ui = SimpleNamespace()
        controller.UpdateMsgLog = mock.Mock()

        controller._update_preparation_progress({
            "processed": 0,
            "total": 0,
            "failed": 0,
            "excluded": 0,
        })

        controller.UpdateMsgLog.assert_called_once_with(
            "Processed 0/0 | passed 0 | changed 0 | unchanged 0 | "
            "failed 0 | excluded 0"
        )

    def test_format_audit_records_changed_case_checks_and_progress(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            write_test_nifti(
                input_path,
                "CASE_001.nii.gz",
                orientation="LAS",
                scaled_hu=True,
            )
            progress_events = []

            result = _prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("canonical",),
                normalize_for_segmentation=True,
                continue_on_case_error=True,
                progress_callback=progress_events.append,
            )

            self.assertEqual(result.total_case_count, 1)
            self.assertEqual(result.processed_case_count, 1)
            self.assertEqual(result.case_count, 1)
            self.assertEqual(result.passed_case_count, 1)
            self.assertEqual(result.changed_case_count, 1)
            self.assertEqual(result.unchanged_case_count, 0)
            self.assertEqual(result.failed_case_count, 0)
            self.assertEqual(result.excluded_case_count, 0)
            self.assertTrue(result.audit_path.is_file())

            with result.audit_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(row["case_id"], "CASE_001")
            self.assertEqual(row["status"], "passed_changed")
            for field in (
                    "check_readable_nifti",
                    "check_3d",
                    "check_scalar",
                    "check_finite",
                    "check_integral_hu",
                    "check_int16_range",
                    "check_direction_orthonormal",
                    "check_output_lps",
                    "check_output_int16",
                    "check_geometry_preserved",
                    "check_values_preserved",
                    "check_destination_sha256"):
                self.assertEqual(row[field], "PASS", field)

            final_event = progress_events[-1]
            self.assertEqual(final_event["processed"], 1)
            self.assertEqual(final_event["total"], 1)
            self.assertEqual(final_event["passed"], 1)
            self.assertEqual(final_event["changed"], 1)
            self.assertEqual(final_event["unchanged"], 0)
            self.assertEqual(final_event["failed"], 0)
            self.assertEqual(final_event["excluded"], 0)
            self.assertEqual(final_event["status"], "passed_changed")

    def test_format_audit_records_already_compliant_case_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            write_test_nifti(
                input_path, "CASE_001.nii.gz", orientation="LPS"
            )

            result = _prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("canonical",),
                normalize_for_segmentation=True,
                continue_on_case_error=True,
            )

            self.assertEqual(result.passed_case_count, 1)
            self.assertEqual(result.changed_case_count, 0)
            self.assertEqual(result.unchanged_case_count, 1)
            with result.audit_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(row["status"], "passed_unchanged")
            self.assertEqual(row["check_output_lps"], "PASS")
            self.assertEqual(row["check_output_int16"], "PASS")
            self.assertEqual(row["check_geometry_preserved"], "PASS")
            self.assertEqual(row["check_values_preserved"], "PASS")
            self.assertEqual(row["check_destination_sha256"], "PASS")

    def test_excluded_case_is_audited_and_next_valid_case_continues(self):
        import numpy
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            invalid_path = input_path / "CASE_001.nii.gz"
            invalid_image = sitk.GetImageFromArray(
                numpy.array([[[0.0, 0.5]]], dtype=numpy.float32)
            )
            sitk.WriteImage(invalid_image, str(invalid_path), True)
            write_test_nifti(
                input_path, "CASE_002.nii.gz", orientation="LPS"
            )
            progress_events = []

            result = _prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("canonical",),
                normalize_for_segmentation=True,
                continue_on_case_error=True,
                progress_callback=progress_events.append,
            )

            self.assertEqual(result.total_case_count, 2)
            self.assertEqual(result.processed_case_count, 2)
            self.assertEqual(result.case_count, 1)
            self.assertEqual(result.passed_case_count, 1)
            self.assertEqual(result.failed_case_count, 0)
            self.assertEqual(result.excluded_case_count, 1)
            with result.audit_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                audit_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(audit_rows), 2)
            audit_by_case = {row["case_id"]: row for row in audit_rows}
            excluded_row = audit_by_case["CASE_001"]
            self.assertEqual(excluded_row["status"], "excluded")
            self.assertEqual(excluded_row["check_readable_nifti"], "PASS")
            self.assertEqual(excluded_row["check_3d"], "PASS")
            self.assertEqual(excluded_row["check_scalar"], "PASS")
            self.assertEqual(excluded_row["check_integral_hu"], "FAIL")
            self.assertIn(
                "fractional", " ".join(excluded_row.values()).casefold()
            )
            self.assertEqual(
                audit_by_case["CASE_002"]["status"], "passed_unchanged"
            )
            self.assertEqual(audit_by_case["CASE_002"]["processed_so_far"], "2")
            self.assertEqual(audit_by_case["CASE_002"]["selected_total"], "2")
            self.assertEqual(audit_by_case["CASE_002"]["passed_so_far"], "1")
            self.assertEqual(audit_by_case["CASE_002"]["failed_so_far"], "0")
            self.assertEqual(audit_by_case["CASE_002"]["excluded_so_far"], "1")

            with result.manifest_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                manifest_rows = list(csv.DictReader(csv_file))
            self.assertEqual(
                [row["case_id"] for row in manifest_rows], ["CASE_002"]
            )
            invalid_destination = (
                result.manifest_path.parent
                / "batch_0"
                / "CASE_001"
                / CANONICAL_CTA_RELATIVE_PATH
            )
            valid_destination = (
                result.manifest_path.parent
                / "batch_0"
                / "CASE_002"
                / CANONICAL_CTA_RELATIVE_PATH
            )
            self.assertFalse(invalid_destination.exists())
            self.assertTrue(valid_destination.is_file())
            self.assertEqual(progress_events[-1]["processed"], 2)
            self.assertEqual(progress_events[-1]["passed"], 1)
            self.assertEqual(progress_events[-1]["excluded"], 1)

    def test_four_dimensional_input_is_explicitly_excluded(self):
        import numpy
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            source_path = input_path / "CASE_4D.nii.gz"
            image = sitk.GetImageFromArray(
                numpy.zeros((2, 2, 2, 2), dtype=numpy.int16),
                isVector=False,
            )
            sitk.WriteImage(image, str(source_path), True)

            result = _prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("canonical",),
                normalize_for_segmentation=True,
                continue_on_case_error=True,
            )

            self.assertEqual(result.case_count, 0)
            self.assertEqual(result.excluded_case_count, 1)
            with result.audit_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(row["status"], "excluded")
            self.assertEqual(row["source_dimension"], "4")
            self.assertEqual(row["check_readable_nifti"], "PASS")
            self.assertEqual(row["check_3d"], "FAIL")
            self.assertEqual(row["check_scalar"], "NOT_RUN")
            self.assertIn("3-d", row["format_check_message"].casefold())
            self.assertFalse(
                (
                    result.manifest_path.parent
                    / "batch_0"
                    / "CASE_4D"
                    / "0.dcm"
                ).exists()
            )

    def test_unexpected_case_failure_is_audited_and_processing_continues(self):
        import BatchNIIDataPrepareMain as prepare_module

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            write_test_nifti(
                input_path, "CASE_001.nii.gz", orientation="LPS"
            )
            write_test_nifti(
                input_path, "CASE_002.nii.gz", orientation="LPS"
            )
            original_copy_case_cta = prepare_module._copy_case_cta

            def copy_or_fail(plan, *args, **kwargs):
                if plan.case_id == "CASE_001":
                    raise OSError("simulated copy failure")
                return original_copy_case_cta(plan, *args, **kwargs)

            with mock.patch.object(
                    prepare_module,
                    "_copy_case_cta",
                    side_effect=copy_or_fail):
                result = _prepare_batch_data(
                    input_path,
                    output_path,
                    create_pipeline_folders=False,
                    cta_destinations=("canonical",),
                    normalize_for_segmentation=True,
                    continue_on_case_error=True,
                )

            self.assertEqual(result.total_case_count, 2)
            self.assertEqual(result.processed_case_count, 2)
            self.assertEqual(result.case_count, 1)
            self.assertEqual(result.failed_case_count, 1)
            self.assertEqual(result.excluded_case_count, 0)
            with result.audit_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                audit_rows = {
                    row["case_id"]: row for row in csv.DictReader(csv_file)
                }
            self.assertEqual(audit_rows["CASE_001"]["status"], "failed")
            self.assertIn(
                "simulated copy failure",
                " ".join(audit_rows["CASE_001"].values()).casefold(),
            )
            self.assertEqual(
                audit_rows["CASE_002"]["status"], "passed_unchanged"
            )

    def test_parallel_workers_overlap_and_manifest_stays_ranked(self):
        import BatchNIIDataPrepareMain as prepare_module

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            for rank in range(1, 5):
                write_source(
                    input_path,
                    "CASE_{0:03d}.nii.gz".format(rank),
                    ("source-{0}".format(rank).encode("ascii")) * 200,
                )

            original_copy_case_cta = prepare_module._copy_case_cta
            original_append_audit = prepare_module._append_format_audit_row
            guard = threading.Lock()
            second_worker_started = threading.Event()
            worker_state = {"active": 0, "maximum": 0}
            completion_order = []

            def delayed_copy(plan, *args, **kwargs):
                with guard:
                    worker_state["active"] += 1
                    worker_state["maximum"] = max(
                        worker_state["maximum"], worker_state["active"]
                    )
                try:
                    if plan.rank == 1:
                        self.assertTrue(second_worker_started.wait(2.0))
                        time.sleep(0.05)
                    else:
                        second_worker_started.set()
                    result = original_copy_case_cta(plan, *args, **kwargs)
                    completion_order.append(plan.rank)
                    return result
                finally:
                    with guard:
                        worker_state["active"] -= 1

            callback_thread_ids = []
            audit_thread_ids = []
            progress_events = []

            def record_progress(event):
                callback_thread_ids.append(threading.get_ident())
                progress_events.append(event)

            def record_audit(*args, **kwargs):
                audit_thread_ids.append(threading.get_ident())
                return original_append_audit(*args, **kwargs)

            coordinator_thread_id = threading.get_ident()
            with mock.patch.object(
                    prepare_module,
                    "_copy_case_cta",
                    side_effect=delayed_copy), mock.patch.object(
                        prepare_module,
                        "_append_format_audit_row",
                        side_effect=record_audit):
                    result = prepare_batch_data(
                        input_path,
                        output_path,
                        create_pipeline_folders=False,
                        cta_destinations=("canonical",),
                        continue_on_case_error=True,
                        parallel_workers=3,
                        progress_callback=record_progress,
                    )

            self.assertGreaterEqual(worker_state["maximum"], 2)
            self.assertNotEqual(completion_order[0], 1)
            self.assertEqual(result.parallel_workers, 3)
            self.assertGreater(result.elapsed_seconds, 0.0)
            self.assertIn("throughput", result.summary().casefold())
            self.assertTrue(all(
                thread_id == coordinator_thread_id
                for thread_id in callback_thread_ids
            ))
            self.assertTrue(all(
                thread_id == coordinator_thread_id
                for thread_id in audit_thread_ids
            ))
            self.assertEqual(progress_events[-1]["processed"], 4)
            self.assertEqual(progress_events[-1]["passed"], 4)

            with result.manifest_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                manifest_rows = list(csv.DictReader(csv_file))
            self.assertEqual(
                [int(row["rank"]) for row in manifest_rows],
                [1, 2, 3, 4],
            )
            self.assertTrue(all(
                row["preparation_parallel_workers"] == "3"
                for row in manifest_rows
            ))
            with result.audit_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                audit_rows = list(csv.DictReader(csv_file))
            self.assertNotEqual(int(audit_rows[0]["rank"]), 1)
            self.assertEqual(
                [int(row["processed_so_far"]) for row in audit_rows],
                [1, 2, 3, 4],
            )
            self.assertEqual(audit_rows[-1]["passed_so_far"], "4")
            self.assertTrue(all(
                row["preparation_parallel_workers"] == "3"
                for row in audit_rows
            ))

    def test_parallel_layout_creation_serializes_shared_batch_preproc_state(self):
        class ProbingBatchPreproc(StubBatchPreproc):
            def __init__(self):
                super().__init__()
                self.guard = threading.Lock()
                self.active = 0
                self.maximum_active = 0
                self.table_mismatches = []

            def CreateFolders(self, CasePath=None):
                case_path = Path(CasePath)
                with self.guard:
                    self.active += 1
                    self.maximum_active = max(
                        self.maximum_active, self.active
                    )
                try:
                    expected_table = case_path.parent / "Table"
                    table_before = Path(self.TableDir)
                    time.sleep(0.02)
                    table_after = Path(self.TableDir)
                    if (
                            table_before != expected_table
                            or table_after != expected_table):
                        self.table_mismatches.append(
                            (case_path, table_before, table_after)
                        )
                    super().CreateFolders(CasePath=CasePath)
                finally:
                    with self.guard:
                        self.active -= 1

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            for rank in range(1, 6):
                write_source(
                    input_path,
                    "CASE_{0:03d}.nii.gz".format(rank),
                    b"source" * 100,
                )
            batch_preproc = ProbingBatchPreproc()

            result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=True,
                batch_preproc=batch_preproc,
                cases_per_batch=2,
                cta_destinations=("ori_data",),
                continue_on_case_error=True,
                parallel_workers=4,
            )

            self.assertEqual(result.case_count, 5)
            self.assertEqual(batch_preproc.maximum_active, 1)
            self.assertEqual(batch_preproc.table_mismatches, [])
            self.assertEqual(len(batch_preproc.case_paths), 5)
            self.assertEqual(
                Path(batch_preproc.TableDir),
                result.manifest_path.parent / "batch_2" / "Table",
            )

    def test_parallel_failure_continues_with_accurate_cumulative_counts(self):
        import BatchNIIDataPrepareMain as prepare_module

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            for rank in range(1, 4):
                write_source(
                    input_path,
                    "CASE_{0:03d}.nii.gz".format(rank),
                    b"source" * 100,
                )
            original_copy_case_cta = prepare_module._copy_case_cta

            def copy_or_fail(plan, *args, **kwargs):
                if plan.rank == 2:
                    raise OSError("parallel simulated failure")
                return original_copy_case_cta(plan, *args, **kwargs)

            with mock.patch.object(
                    prepare_module,
                    "_copy_case_cta",
                    side_effect=copy_or_fail):
                result = prepare_batch_data(
                    input_path,
                    output_path,
                    create_pipeline_folders=False,
                    cta_destinations=("canonical",),
                    continue_on_case_error=True,
                    parallel_workers=3,
                )

            self.assertEqual(result.processed_case_count, 3)
            self.assertEqual(result.passed_case_count, 2)
            self.assertEqual(result.failed_case_count, 1)
            with result.manifest_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                manifest_rows = list(csv.DictReader(csv_file))
            self.assertEqual(
                [int(row["rank"]) for row in manifest_rows], [1, 3]
            )
            with result.audit_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                audit_rows = list(csv.DictReader(csv_file))
            self.assertEqual(
                [int(row["processed_so_far"]) for row in audit_rows],
                [1, 2, 3],
            )
            self.assertEqual(audit_rows[-1]["passed_so_far"], "2")
            self.assertEqual(audit_rows[-1]["failed_so_far"], "1")

    def test_requested_parallelism_preserves_fail_fast_behavior(self):
        import BatchNIIDataPrepareMain as prepare_module

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            input_path.mkdir()
            for rank in range(1, 4):
                write_source(
                    input_path,
                    "CASE_{0:03d}.nii.gz".format(rank),
                    b"source",
                )
            attempted_cases = []

            def fail_first(plan, *args, **kwargs):
                attempted_cases.append(plan.case_id)
                raise OSError("stop here")

            with mock.patch.object(
                    prepare_module,
                    "_copy_case_cta",
                    side_effect=fail_first):
                with self.assertRaisesRegex(OSError, "stop here"):
                    prepare_batch_data(
                        input_path,
                        temporary_path / "output",
                        create_pipeline_folders=False,
                        cta_destinations=("canonical",),
                        continue_on_case_error=False,
                        parallel_workers=4,
                    )

            self.assertEqual(attempted_cases, ["CASE_001"])

    def test_simpleitk_thread_limit_is_restored_after_error(self):
        import BatchNIIDataPrepareMain as prepare_module

        class FakeProcessObject:
            threads = 16
            history = []

            @classmethod
            def GetGlobalDefaultNumberOfThreads(cls):
                return cls.threads

            @classmethod
            def SetGlobalDefaultNumberOfThreads(cls, value):
                cls.threads = value
                cls.history.append(value)

        fake_sitk = SimpleNamespace(ProcessObject=FakeProcessObject)
        with mock.patch.object(prepare_module.os, "cpu_count", return_value=8):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                with prepare_module._simpleitk_thread_limit(fake_sitk, 4):
                    self.assertEqual(FakeProcessObject.threads, 2)
                    raise RuntimeError("simulated")

        self.assertEqual(FakeProcessObject.threads, 16)
        self.assertEqual(FakeProcessObject.history, [2, 16])

        FakeProcessObject.history = []
        with prepare_module._simpleitk_thread_limit(fake_sitk, 1):
            self.assertEqual(FakeProcessObject.threads, 16)
        self.assertEqual(FakeProcessObject.history, [])

    def test_parallel_normalization_restores_real_simpleitk_threads(self):
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            for rank in range(1, 4):
                write_test_nifti(
                    input_path,
                    "CASE_{0:03d}.nii.gz".format(rank),
                    orientation="LAS",
                    scaled_hu=True,
                )

            original_threads = (
                sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
            )
            result = _prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("canonical",),
                normalize_for_segmentation=True,
                continue_on_case_error=True,
                parallel_workers=2,
            )

            self.assertEqual(result.passed_case_count, 3)
            self.assertEqual(result.parallel_workers, 2)
            self.assertEqual(
                sitk.ProcessObject.GetGlobalDefaultNumberOfThreads(),
                original_threads,
            )

    def test_scaled_las_cta_is_losslessly_normalized_once_for_all_destinations(self):
        import numpy
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            source = write_test_nifti(
                input_path,
                "CASE_001.nii.gz",
                orientation="LAS",
                scaled_hu=True,
            )
            source_image = sitk.ReadImage(str(source))
            self.assertEqual(source_image.GetPixelID(), sitk.sitkFloat32)
            self.assertEqual(
                sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
                    source_image.GetDirection()
                ),
                "LAS",
            )

            result = _prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("canonical", "ori_data"),
                normalize_for_segmentation=True,
            )

            case_path = result.manifest_path.parent / "batch_0" / "CASE_001"
            canonical = case_path / dict(CTA_DESTINATION_RELATIVE_PATHS)["canonical"]
            ori_data = case_path / dict(CTA_DESTINATION_RELATIVE_PATHS)["ori_data"]
            canonical_image = sitk.ReadImage(str(canonical))
            expected_image = sitk.DICOMOrient(source_image, "LPS")
            expected_data = sitk.GetArrayFromImage(expected_image).astype(numpy.int16)
            actual_data = sitk.GetArrayFromImage(canonical_image)

            self.assertEqual(result.normalized_case_count, 1)
            self.assertEqual(canonical_image.GetPixelID(), sitk.sitkInt16)
            self.assertEqual(
                sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
                    canonical_image.GetDirection()
                ),
                "LPS",
            )
            numpy.testing.assert_array_equal(actual_data, expected_data)
            self.assertEqual(sha256_file(canonical), sha256_file(ori_data))
            self.assertNotEqual(sha256_file(source), sha256_file(canonical))
            self.assertFalse(list(case_path.glob(".CTA.normalized.*")))

            with result.manifest_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(row["normalize_for_segmentation"], "True")
            self.assertEqual(row["source_orientation"], "LAS")
            self.assertEqual(row["prepared_orientation"], "LPS")
            self.assertEqual(row["prepared_pixel_type"], "16-bit signed integer")
            self.assertEqual(row["prepared_cta_sha256"], sha256_file(canonical))
            self.assertEqual(row["canonical_verified"], "True")
            self.assertEqual(row["pipeline_copies_verified"], "True")
            self.assertEqual(row["all_verified"], "True")

            rerun = _prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("canonical", "ori_data"),
                normalize_for_segmentation=True,
            )
            self.assertEqual(rerun.copied_file_count, 0)
            self.assertEqual(rerun.unchanged_file_count, 2)

    def test_lps_int16_normalization_is_a_byte_copy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            source = write_test_nifti(
                input_path, "CASE_001.nii.gz", orientation="LPS"
            )

            result = _prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("ori_data",),
                normalize_for_segmentation=True,
            )

            destination = (
                result.manifest_path.parent
                / "batch_0"
                / "CASE_001"
                / dict(CTA_DESTINATION_RELATIVE_PATHS)["ori_data"]
            )
            self.assertEqual(result.normalized_case_count, 0)
            self.assertEqual(sha256_file(source), sha256_file(destination))
            with result.manifest_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(row["source_orientation"], "LPS")
            self.assertEqual(row["prepared_orientation"], "LPS")
            self.assertEqual(row["source_sha256"], row["prepared_cta_sha256"])

    def test_changed_prepared_representation_invalidates_processing_status(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            source = write_test_nifti(
                input_path,
                "CASE_001.nii.gz",
                orientation="LAS",
                scaled_hu=True,
            )
            source_hash = sha256_file(source)

            first_result = _prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("ori_data",),
                normalize_for_segmentation=False,
            )
            with first_result.manifest_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                fieldnames = reader.fieldnames
                row = next(reader)
            row["run_segmentation"] = "1"
            row["complete_segmentation"] = "1"
            with first_result.manifest_path.open(
                    "w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=fieldnames, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerow(row)

            _prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("ori_data",),
                normalize_for_segmentation=True,
            )
            with first_result.manifest_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                rerun_row = next(csv.DictReader(csv_file))

            self.assertEqual(rerun_row["source_sha256"], source_hash)
            self.assertNotEqual(
                rerun_row["prepared_cta_sha256"], row["prepared_cta_sha256"]
            )
            self.assertEqual(rerun_row["run_segmentation"], "0")
            self.assertEqual(rerun_row["complete_segmentation"], "0")

    def test_fractional_cta_is_rejected_without_leaving_a_normalized_temp(self):
        import numpy
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            source = input_path / "CASE_001.nii.gz"
            image = sitk.GetImageFromArray(
                numpy.array([[[0.0, 0.5]]], dtype=numpy.float32)
            )
            sitk.WriteImage(image, str(source), True)

            with self.assertRaisesRegex(ValueError, "fractional values"):
                _prepare_batch_data(
                    input_path,
                    output_path,
                    create_pipeline_folders=False,
                    cta_destinations=("ori_data",),
                    normalize_for_segmentation=True,
                )

            case_path = (
                output_path / OUTPUT_FOLDER_NAME / "batch_0" / "CASE_001"
            )
            self.assertFalse(
                (case_path / dict(CTA_DESTINATION_RELATIVE_PATHS)["ori_data"]).exists()
            )
            self.assertFalse(list(case_path.glob(".CTA.normalized.*")))

    def test_explicit_ori_data_only_selection_creates_one_verified_cta(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            source = write_source(
                input_path,
                "CASE_001.nii.gz",
                b"single-authoritative-cta" * 100,
            )

            result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=True,
                batch_preproc=StubBatchPreproc(),
                cta_destinations=DEFAULT_CTA_DESTINATIONS,
            )

            case_path = (
                output_path / OUTPUT_FOLDER_NAME / "batch_0" / "CASE_001"
            )
            destination_paths = {
                key: case_path / relative_path
                for key, relative_path in CTA_DESTINATION_RELATIVE_PATHS
            }
            self.assertEqual(result.cta_destinations, ("ori_data",))
            self.assertTrue(result.pipeline_ready)
            self.assertEqual(result.copied_file_count, 1)
            self.assertTrue(destination_paths["ori_data"].is_file())
            self.assertEqual(
                sha256_file(destination_paths["ori_data"]),
                sha256_file(source),
            )
            self.assertTrue(all(
                not path.exists()
                for key, path in destination_paths.items()
                if key != "ori_data"
            ))

            with result.manifest_path.open(
                    "r", encoding="utf-8", newline="") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(row["canonical_cta_path"], "")
            self.assertEqual(row["canonical_cta_sha256"], "")
            self.assertEqual(row["canonical_verified"], "False")
            self.assertEqual(
                Path(row["ori_data_cta_path"]).resolve(),
                destination_paths["ori_data"].resolve(),
            )
            self.assertEqual(row["ori_data_cta_sha256"], sha256_file(source))
            self.assertEqual(row["pipeline_copies_requested"], "True")
            self.assertEqual(row["pipeline_copies_verified"], "True")
            self.assertEqual(row["all_verified"], "True")

    def test_every_explicit_cta_destination_can_be_selected_individually(self):
        for destination_key, relative_path in CTA_DESTINATION_RELATIVE_PATHS:
            with self.subTest(destination=destination_key), tempfile.TemporaryDirectory() as temporary_directory:
                temporary_path = Path(temporary_directory)
                input_path = temporary_path / "input"
                output_path = temporary_path / "output"
                input_path.mkdir()
                source = write_source(input_path, "CASE_001.nii.gz", b"source")

                result = prepare_batch_data(
                    input_path,
                    output_path,
                    create_pipeline_folders=False,
                    cta_destinations=destination_key,
                )

                case_path = (
                    output_path / OUTPUT_FOLDER_NAME / "batch_0" / "CASE_001"
                )
                self.assertEqual(result.cta_destinations, (destination_key,))
                self.assertEqual(
                    sha256_file(case_path / relative_path),
                    sha256_file(source),
                )
                self.assertEqual(
                    [
                        key
                        for key, candidate in CTA_DESTINATION_RELATIVE_PATHS
                        if (case_path / candidate).is_file()
                    ],
                    [destination_key],
                )

    def test_invalid_cta_destinations_are_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            input_path.mkdir()
            write_source(input_path, "CASE_001.nii.gz", b"source")

            for selection, message in (((), "at least one"), (("unknown",), "Unknown")):
                output_path = temporary_path / ("output_" + (message.lower().replace(" ", "_")))
                with self.subTest(selection=selection), self.assertRaisesRegex(
                        ValueError, message):
                    prepare_batch_data(
                        input_path,
                        output_path,
                        create_pipeline_folders=False,
                        cta_destinations=selection,
                    )
                self.assertFalse(output_path.exists())

    def test_unselected_existing_cta_is_preserved_and_reported_on_rerun(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            write_source(input_path, "CASE_001.nii.gz", b"source")
            first_result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("ori_data",),
            )
            case_path = first_result.manifest_path.parent / "batch_0" / "CASE_001"
            legacy_path = case_path / dict(CTA_DESTINATION_RELATIVE_PATHS)["branches"]
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.write_bytes(b"preserve-this-legacy-copy")

            second_result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cta_destinations=("ori_data",),
            )

            self.assertEqual(second_result.copied_file_count, 0)
            self.assertEqual(second_result.unchanged_file_count, 1)
            self.assertEqual(second_result.preserved_unselected_file_count, 1)
            self.assertEqual(legacy_path.read_bytes(), b"preserve-this-legacy-copy")
            self.assertIn("preserved 1", second_result.summary())

    def test_blank_paths_and_empty_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            input_path.mkdir()
            write_source(input_path, "CASE_001.nii.gz", b"data")

            with self.assertRaisesRegex(ValueError, "Output folder is required"):
                build_case_plan(input_path, "  ")
            with self.assertRaisesRegex(ValueError, "Input folder is required"):
                build_case_plan("", temporary_path / "output")

            write_source(input_path, "CASE_002.nii.gz", b"")
            with self.assertRaisesRegex(ValueError, "Empty .nii.gz"):
                build_case_plan(input_path, temporary_path / "output")

    def test_sanitize_case_id_removes_special_characters(self):
        self.assertEqual(
            sanitize_case_id("CASE 00-a!_b.nii.gz"),
            "CASE00a_b",
        )
        with self.assertRaises(ValueError):
            sanitize_case_id("---.nii.gz")
        with self.assertRaises(ValueError):
            sanitize_case_id("CON.nii.gz")
        with self.assertRaises(ValueError):
            sanitize_case_id("case.nii")

    def test_collision_is_rejected_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            write_source(input_path, "CASE-a.nii.gz", b"one")
            write_source(input_path, "CASEa.nii.gz", b"two")

            with self.assertRaisesRegex(ValueError, "collision"):
                prepare_batch_data(
                    input_path,
                    output_path,
                    create_pipeline_folders=False,
                )

            self.assertFalse(output_path.exists())

    def test_plan_is_ranked_and_batches_are_limited_to_100(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            for index in reversed(range(205)):
                write_source(
                    input_path,
                    "CASE_{0:03d}.nii.gz".format(index),
                    str(index).encode("ascii"),
                )

            plans = build_case_plan(input_path, output_path)

            self.assertEqual(len(plans), 205)
            self.assertEqual(plans[0].source_path.name, "CASE_000.nii.gz")
            self.assertEqual(plans[0].rank, 1)
            self.assertEqual(plans[0].batch_name, "batch_0")
            self.assertEqual(plans[99].batch_position, 100)
            self.assertEqual(plans[100].batch_name, "batch_1")
            self.assertEqual(plans[200].batch_name, "batch_2")
            self.assertEqual(plans[204].batch_position, 5)
            self.assertFalse(output_path.exists())

    def test_custom_batch_size_five_creates_deterministic_5_5_2_layout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            for index in reversed(range(12)):
                write_source(
                    input_path,
                    "CASE_{0:03d}.nii.gz".format(index),
                    str(index).encode("ascii"),
                )

            plans = build_case_plan(input_path, output_path, cases_per_batch=5)
            self.assertEqual(
                [sum(plan.batch_name == name for plan in plans) for name in (
                    "batch_0", "batch_1", "batch_2"
                )],
                [5, 5, 2],
            )
            self.assertEqual([plan.batch_position for plan in plans], [
                1, 2, 3, 4, 5,
                1, 2, 3, 4, 5,
                1, 2,
            ])

            result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
                cases_per_batch=5,
            )
            self.assertEqual(result.batch_count, 3)
            self.assertEqual(result.cases_per_batch, 5)
            with result.manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual([row["cases_per_batch"] for row in rows], ["5"] * 12)
            self.assertEqual([row["batch_name"] for row in rows], [
                "batch_0", "batch_0", "batch_0", "batch_0", "batch_0",
                "batch_1", "batch_1", "batch_1", "batch_1", "batch_1",
                "batch_2", "batch_2",
            ])

    def test_invalid_cases_per_batch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            input_path.mkdir()
            write_source(input_path, "CASE_001.nii.gz", b"source")

            for invalid_value in (0, -1, 101):
                with self.assertRaisesRegex(ValueError, "between 1 and 100"):
                    build_case_plan(
                        input_path,
                        temporary_path / "output",
                        cases_per_batch=invalid_value,
                    )
            for invalid_value in (True, 2.5, "5"):
                with self.assertRaisesRegex(TypeError, "must be an integer"):
                    build_case_plan(
                        input_path,
                        temporary_path / "output",
                        cases_per_batch=invalid_value,
                    )

    def test_changed_batch_size_is_rejected_before_duplicate_cases_are_created(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            for index in range(6):
                write_source(
                    input_path,
                    "CASE_{0:03d}.nii.gz".format(index),
                    str(index).encode("ascii"),
                )

            first_result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
            )
            original_manifest = first_result.manifest_path.read_bytes()

            with self.assertRaisesRegex(ValueError, "Existing batch layout conflicts"):
                prepare_batch_data(
                    input_path,
                    output_path,
                    create_pipeline_folders=False,
                    cases_per_batch=5,
                )

            data_root = output_path / OUTPUT_FOLDER_NAME
            self.assertFalse((data_root / "batch_1").exists())
            self.assertEqual(first_result.manifest_path.read_bytes(), original_manifest)

    def test_canonical_preparation_is_verified_and_manifest_is_stable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            source_a = write_source(input_path, "CASE_B.nii.gz", b"B" * 513)
            source_b = write_source(input_path, "case_a.nii.gz", b"A" * 1025)

            first_result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
            )
            manifest_path = (
                output_path / OUTPUT_FOLDER_NAME / MANIFEST_FILE_NAME
            )
            first_manifest = manifest_path.read_bytes()

            # An unrelated result in an existing case must survive a rerun.
            extra_file = (
                output_path
                / OUTPUT_FOLDER_NAME
                / "batch_0"
                / "case_a"
                / "keep.txt"
            )
            extra_file.write_text("keep", encoding="utf-8")

            second_result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
            )

            self.assertEqual(first_result.case_count, 2)
            self.assertEqual(first_result.cases_per_batch, 100)
            self.assertEqual(first_result.copied_file_count, 2)
            self.assertEqual(second_result.copied_file_count, 0)
            self.assertEqual(second_result.unchanged_file_count, 2)
            self.assertEqual(first_manifest, manifest_path.read_bytes())
            self.assertEqual(extra_file.read_text(encoding="utf-8"), "keep")

            with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual([row["source_file"] for row in rows], [
                "case_a.nii.gz",
                "CASE_B.nii.gz",
            ])
            self.assertEqual([row["rank"] for row in rows], ["1", "2"])
            self.assertEqual(rows[0]["pipeline_copies_requested"], "False")
            self.assertEqual(rows[0]["cases_per_batch"], "100")
            self.assertEqual(rows[0]["all_verified"], "True")

            for row, source in zip(rows, (source_b, source_a)):
                destination = Path(row["canonical_cta_path"])
                self.assertEqual(sha256_file(source), sha256_file(destination))
                marker = destination.parents[1] / "0.dcm"
                self.assertTrue(marker.is_file())
                self.assertEqual(marker.stat().st_size, 0)

    def test_pipeline_folders_and_all_cta_copies_are_refreshed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            source = write_source(input_path, "CASE_001.nii.gz", b"real-source" * 100)
            data_root = output_path / OUTPUT_FOLDER_NAME
            case_path = data_root / "batch_0" / "CASE_001"

            # Simulate a stale prior run in one downstream location.
            stale_path = case_path / PIPELINE_CTA_RELATIVE_PATHS[1][1]
            stale_path.parent.mkdir(parents=True)
            stale_path.write_bytes(b"stale")

            batch_preproc = StubBatchPreproc()
            result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=True,
                batch_preproc=batch_preproc,
            )

            self.assertTrue(result.pipeline_ready)
            batch_table = data_root / "batch_0" / "Table"
            self.assertEqual(batch_preproc.TableDir, str(batch_table))
            self.assertEqual(batch_preproc.table_paths, [batch_table])
            self.assertTrue(batch_table.is_dir())
            self.assertFalse((data_root / "Table").exists())
            self.assertEqual(batch_preproc.case_paths, [case_path])
            expected_hash = sha256_file(source)
            all_destinations = [case_path / CANONICAL_CTA_RELATIVE_PATH]
            all_destinations.extend(
                case_path / relative_path
                for _, relative_path in PIPELINE_CTA_RELATIVE_PATHS
            )
            self.assertEqual(len(all_destinations), 5)
            for destination in all_destinations:
                self.assertTrue(destination.is_file())
                self.assertEqual(sha256_file(destination), expected_hash)

            manifest_path = data_root / MANIFEST_FILE_NAME
            with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(row["pipeline_copies_verified"], "True")
            self.assertEqual(row["all_verified"], "True")
            self.assertEqual(row["branches_cta_sha256"], expected_hash)

    def test_each_batch_gets_its_own_table_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            for index in range(101):
                write_source(
                    input_path,
                    "CASE_{0:03d}.nii.gz".format(index),
                    "source-{0}".format(index).encode("ascii"),
                )

            batch_preproc = StubBatchPreproc()
            result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=True,
                batch_preproc=batch_preproc,
            )

            data_root = output_path / OUTPUT_FOLDER_NAME
            expected_tables = {
                data_root / "batch_0" / "Table",
                data_root / "batch_1" / "Table",
            }
            self.assertEqual(result.batch_count, 2)
            self.assertEqual(set(batch_preproc.table_paths), expected_tables)
            self.assertTrue(all(path.is_dir() for path in expected_tables))
            self.assertFalse((data_root / "Table").exists())

    def test_batch_table_file_conflict_is_rejected_before_case_creation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            write_source(input_path, "CASE_001.nii.gz", b"source")

            batch_root = output_path / OUTPUT_FOLDER_NAME / "batch_0"
            batch_root.mkdir(parents=True)
            (batch_root / "Table").write_text("not a directory", encoding="utf-8")

            with self.assertRaisesRegex(NotADirectoryError, "batch Table path"):
                prepare_batch_data(
                    input_path,
                    output_path,
                    create_pipeline_folders=True,
                    batch_preproc=StubBatchPreproc(),
                )

            self.assertFalse((batch_root / "CASE_001").exists())

    def test_actual_batch_coronary_folder_creator_and_scanner(self):
        gui_root = MODULE_DIRECTORY.parents[1]
        functions_root = gui_root.parent / "Functions_JZ"
        for import_path in (gui_root, functions_root):
            import_string = str(import_path)
            if import_string not in sys.path:
                sys.path.insert(0, import_string)

        from Modules_YL.BatchCoronary import BatchCoronaryMain

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            source = write_source(
                input_path,
                "CASE_001.nii.gz",
                b"integration-source" * 100,
            )

            batch_preproc = BatchCoronaryMain.BatchCoronary()
            result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=True,
                batch_preproc=batch_preproc,
            )

            data_root = output_path / OUTPUT_FOLDER_NAME
            case_path = data_root / "batch_0" / "CASE_001"
            expected_directories = (
                "CTA",
                "Preproc/OriData/Branches",
                "Preproc/OriVolume",
                "Preproc/CSV",
                "Preproc/Xsection",
                "Preproc/XsectionFix",
                "Preproc/XsectionFixExt",
                "CFD/STL",
                "CFD/MurraysLaw",
                "CFD/result",
                "FFR/result",
                "Table_Preproc",
                "Table_CFD",
                "PostProgress",
                "FFRPostProgress",
                "IFRPostProgress",
            )
            self.assertTrue(result.pipeline_ready)
            batch_root = data_root / "batch_0"
            self.assertTrue((batch_root / "Table").is_dir())
            self.assertFalse((data_root / "Table").exists())
            for relative_directory in expected_directories:
                self.assertTrue(
                    (case_path / relative_directory).is_dir(),
                    relative_directory,
                )

            source_hash = sha256_file(source)
            destinations = [case_path / CANONICAL_CTA_RELATIVE_PATH]
            destinations.extend(
                case_path / relative_path
                for _, relative_path in PIPELINE_CTA_RELATIVE_PATHS
            )
            self.assertTrue(all(sha256_file(path) == source_hash for path in destinations))

            batch_preproc.WorkingDir = str(batch_root)
            batch_preproc.DCMDirList = []
            batch_preproc.CTADirList = []
            batch_preproc.ScanDCMDir()
            self.assertEqual(
                {str(Path(path).resolve()) for path in batch_preproc.DCMDirList},
                {str(case_path.resolve())},
            )
            self.assertEqual(len(batch_preproc.CTADirList), 1)
            self.assertTrue(Path(batch_preproc.CTADirList[0]).is_file())

            class CenterlineGenerationRecorder:
                def __init__(self):
                    self.table_path = None
                    self.batch_called = False

                def InitCenterlineEx(self, exePath=None, TablePath=None):
                    self.table_path = Path(TablePath)

                def BatchViaExcel(self):
                    self.batch_called = True

            recorder = CenterlineGenerationRecorder()
            batch_preproc.model = SimpleNamespace(CenterlineGeneration=recorder)
            batch_preproc.CGEXEPath = "not-launched-in-test.exe"
            batch_preproc.StoreListInformation()
            batch_preproc.BatchCenterlineGen()
            self.assertTrue(recorder.batch_called)
            self.assertEqual(
                recorder.table_path.resolve(),
                (batch_root / "Table" / "BatchCenterlineGeneration.csv").resolve(),
            )
            with recorder.table_path.open("r", encoding="utf-8", newline="") as csv_file:
                centerline_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(centerline_rows), 1)
            centerline_folder = Path(centerline_rows[0]["FolderPath"])
            self.assertEqual(
                centerline_folder.resolve(),
                (case_path / "Preproc/OriData").resolve(),
            )
            self.assertTrue((centerline_folder / "CTA.nii.gz").is_file())

    def test_batch_coronary_main_always_creates_working_table_directory(self):
        gui_root = MODULE_DIRECTORY.parents[1]
        functions_root = gui_root.parent / "Functions_JZ"
        for import_path in (gui_root, functions_root):
            import_string = str(import_path)
            if import_string not in sys.path:
                sys.path.insert(0, import_string)

        from Modules_YL.BatchCoronary import BatchCoronaryMain

        class UncheckedWidget:
            def __init__(self, text=""):
                self._text = text

            def isChecked(self):
                return False

            def toPlainText(self):
                return self._text

        class AllProcessingDisabledUI:
            def __init__(self, working_directory):
                self.LoadFolderPath_Txt_BPP = UncheckedWidget(working_directory)
                self.CGExePath_Txt_BPP = UncheckedWidget()
                self.CPRExePath_Txt_BPP = UncheckedWidget()
                self.STLGExePath_Txt_BPP = UncheckedWidget()

            def __getattr__(self, _name):
                return UncheckedWidget()

        with tempfile.TemporaryDirectory() as temporary_directory:
            batch_root = Path(temporary_directory) / "batch_0"
            batch_root.mkdir()
            batch_preproc = BatchCoronaryMain.BatchCoronary()
            batch_preproc.ui = AllProcessingDisabledUI(str(batch_root))

            with mock.patch.object(BatchCoronaryMain.os, "system", return_value=0):
                batch_preproc.mainFunction()

            self.assertTrue((batch_root / "Table").is_dir())

    def test_processing_status_fields_default_and_survive_preparation_rerun(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            source_path = write_source(input_path, "CASE_001.nii.gz", b"source")

            default_status = {
                "run_segmentation": "0",
                "seg_exe_failed": "0",
                "seg_exe_er_msg": "",
                "coronary_segment_failed": "0",
                "complete_segmentation": "0",
                "run_centerline_csv_check": "0",
                "centerline_csv_failed": "0",
                "centerline_csv_er_msg": "",
                "complete_centerline_csv": "0",
                "run_filter_mask": "0",
                "filter_mask_failed": "0",
                "filter_mask_er_msg": "",
                "complete_filter_mask": "0",
                "run_xsection": "0",
                "LCx_xsection_CTA_exe_failed": "0",
                "LCx_xsection_CTA_exe_er_msg": "",
                "LCx_xsection_CTA_complete": "0",
                "mLAD_xsection_CTA_exe_failed": "0",
                "mLAD_xsection_CTA_exe_er_msg": "",
                "mLAD_xsection_CTA_complete": "0",
                "mRCA_xsection_CTA_exe_failed": "0",
                "mRCA_xsection_CTA_exe_er_msg": "",
                "mRCA_xsection_CTA_complete": "0",
                "complete_xsection_CTA": "0",
                "LCx_xsection_seg_exe_failed": "0",
                "LCx_xsection_seg_exe_er_msg": "",
                "LCx_xsection_seg_complete": "0",
                "mLAD_xsection_seg_exe_failed": "0",
                "mLAD_xsection_seg_exe_er_msg": "",
                "mLAD_xsection_seg_complete": "0",
                "mRCA_xsection_seg_exe_failed": "0",
                "mRCA_xsection_seg_exe_er_msg": "",
                "mRCA_xsection_seg_complete": "0",
                "complete_xsection_seg": "0",
                "LCx_lumen_complete": "0",
                "LCx_lumen_er_msg": "",
                "mLAD_lumen_complete": "0",
                "mLAD_lumen_er_msg": "",
                "mRCA_lumen_complete": "0",
                "mRCA_lumen_er_msg": "",
                "complete_lumen_correction": "0",
            }
            recorded_status = {
                "run_segmentation": "1",
                "seg_exe_failed": "1",
                "seg_exe_er_msg": "Part1 exited with code 9",
                "coronary_segment_failed": "1",
                "complete_segmentation": "0",
                "run_centerline_csv_check": "1",
                "centerline_csv_failed": "1",
                "centerline_csv_er_msg": "Preproc/CSV is empty",
                "complete_centerline_csv": "0",
                "run_filter_mask": "1",
                "filter_mask_failed": "0",
                "filter_mask_er_msg": "",
                "complete_filter_mask": "1",
                "run_xsection": "1",
                "LCx_xsection_CTA_exe_failed": "0",
                "LCx_xsection_CTA_exe_er_msg": "",
                "LCx_xsection_CTA_complete": "1",
                "mLAD_xsection_CTA_exe_failed": "1",
                "mLAD_xsection_CTA_exe_er_msg": "CTA resampling exited with code 7",
                "mLAD_xsection_CTA_complete": "0",
                "mRCA_xsection_CTA_exe_failed": "0",
                "mRCA_xsection_CTA_exe_er_msg": "",
                "mRCA_xsection_CTA_complete": "1",
                "complete_xsection_CTA": "0",
                "LCx_xsection_seg_exe_failed": "0",
                "LCx_xsection_seg_exe_er_msg": "",
                "LCx_xsection_seg_complete": "1",
                "mLAD_xsection_seg_exe_failed": "0",
                "mLAD_xsection_seg_exe_er_msg": "",
                "mLAD_xsection_seg_complete": "1",
                "mRCA_xsection_seg_exe_failed": "1",
                "mRCA_xsection_seg_exe_er_msg": "segmentation output missing",
                "mRCA_xsection_seg_complete": "0",
                "complete_xsection_seg": "0",
                "LCx_lumen_complete": "1",
                "LCx_lumen_er_msg": "",
                "mLAD_lumen_complete": "0",
                "mLAD_lumen_er_msg": "corrected lumen is empty",
                "mRCA_lumen_complete": "1",
                "mRCA_lumen_er_msg": "",
                "complete_lumen_correction": "0",
            }

            result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
            )
            with result.manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                fieldnames = reader.fieldnames
                rows = list(reader)

            self.assertEqual(
                tuple(field for field in fieldnames if field in XSECTION_STATUS_FIELDS),
                XSECTION_STATUS_FIELDS,
            )

            self.assertEqual(
                {field: rows[0][field] for field in default_status},
                default_status,
            )

            rows[0].update(recorded_status)
            with result.manifest_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)

            prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
            )
            with result.manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                rerun_row = next(csv.DictReader(csv_file))

            self.assertEqual(
                {field: rerun_row[field] for field in recorded_status},
                recorded_status,
            )

            source_path.write_bytes(b"changed source")
            prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
            )
            with result.manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                changed_source_row = next(csv.DictReader(csv_file))

            self.assertEqual(
                {field: changed_source_row[field] for field in default_status},
                default_status,
            )

    def test_unknown_manifest_columns_survive_rerun_by_batch_and_case(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            write_source(input_path, "CASE_A.nii.gz", b"source-a")
            write_source(input_path, "CASE_B.nii.gz", b"source-b")

            result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
            )
            with result.manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                fieldnames = list(reader.fieldnames)
                rows = list(reader)

            extra_fieldnames = (
                "pt_reference_complete",
                "pt_reference_output_path",
                "review_note",
            )
            expected_by_case = {}
            for row in rows:
                case_id = row["case_id"]
                expected_by_case[case_id] = {
                    "pt_reference_complete": "1",
                    "pt_reference_output_path": "outputs/{0}.pt".format(case_id),
                    "review_note": "reviewed-{0}".format(case_id),
                }
                row.update(expected_by_case[case_id])

            with result.manifest_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=fieldnames + list(extra_fieldnames),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(reversed(rows))

            prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
            )
            with result.manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                rerun_fieldnames = reader.fieldnames
                rerun_rows = {
                    row["case_id"]: row
                    for row in reader
                }

            self.assertEqual(tuple(rerun_fieldnames[-3:]), extra_fieldnames)
            self.assertEqual(set(rerun_rows), set(expected_by_case))
            for case_id, expected_values in expected_by_case.items():
                self.assertEqual(
                    {
                        field: rerun_rows[case_id][field]
                        for field in extra_fieldnames
                    },
                    expected_values,
                )

    def test_unknown_manifest_values_survive_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            changed_source = write_source(input_path, "CASE_A.nii.gz", b"source-a")
            write_source(input_path, "CASE_B.nii.gz", b"source-b")

            result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
            )
            with result.manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                fieldnames = list(reader.fieldnames)
                rows = list(reader)

            extra_fieldnames = ("pt_run_id", "pt_RCA_state")
            for row in rows:
                row["pt_run_id"] = "run-for-{0}".format(row["case_id"])
                row["pt_RCA_state"] = "complete"
            with result.manifest_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=fieldnames + list(extra_fieldnames),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)

            changed_source.write_bytes(b"changed-source-a")
            prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
            )
            with result.manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                rerun_fieldnames = reader.fieldnames
                rerun_rows = {
                    row["case_id"]: row
                    for row in reader
                }

            self.assertEqual(tuple(rerun_fieldnames[-2:]), extra_fieldnames)
            self.assertEqual(
                {field: rerun_rows["CASE_A"][field] for field in extra_fieldnames},
                {
                    "pt_run_id": "run-for-CASE_A",
                    "pt_RCA_state": "complete",
                },
            )
            self.assertEqual(
                {field: rerun_rows["CASE_B"][field] for field in extra_fieldnames},
                {
                    "pt_run_id": "run-for-CASE_B",
                    "pt_RCA_state": "complete",
                },
            )

    def test_legacy_manifest_gets_defaults_for_new_processing_status_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            write_source(input_path, "CASE_001.nii.gz", b"source")

            result = prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
            )
            with result.manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                fieldnames = reader.fieldnames
                row = next(reader)

            downstream_fields = (
                "run_centerline_csv_check",
                "centerline_csv_failed",
                "centerline_csv_er_msg",
                "complete_centerline_csv",
                "run_filter_mask",
                "filter_mask_failed",
                "filter_mask_er_msg",
                "complete_filter_mask",
            )
            new_processing_fields = downstream_fields + XSECTION_STATUS_FIELDS
            legacy_fieldnames = [
                field for field in fieldnames if field not in new_processing_fields
            ]
            legacy_row = {field: row[field] for field in legacy_fieldnames}
            legacy_row["run_segmentation"] = "1"
            legacy_row["complete_segmentation"] = "1"
            with result.manifest_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=legacy_fieldnames,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(legacy_row)

            prepare_batch_data(
                input_path,
                output_path,
                create_pipeline_folders=False,
            )
            with result.manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
                upgraded_row = next(csv.DictReader(csv_file))

            self.assertEqual(upgraded_row["run_segmentation"], "1")
            self.assertEqual(upgraded_row["complete_segmentation"], "1")
            self.assertEqual(
                {field: upgraded_row[field] for field in new_processing_fields},
                {
                    "run_centerline_csv_check": "0",
                    "centerline_csv_failed": "0",
                    "centerline_csv_er_msg": "",
                    "complete_centerline_csv": "0",
                    "run_filter_mask": "0",
                    "filter_mask_failed": "0",
                    "filter_mask_er_msg": "",
                    "complete_filter_mask": "0",
                    **{
                        field: "" if field.endswith("_er_msg") else "0"
                        for field in XSECTION_STATUS_FIELDS
                    },
                },
            )

    def test_nonempty_dicom_marker_is_never_truncated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_path = temporary_path / "input"
            output_path = temporary_path / "output"
            input_path.mkdir()
            write_source(input_path, "CASE_001.nii.gz", b"source")
            marker = (
                output_path
                / OUTPUT_FOLDER_NAME
                / "batch_0"
                / "CASE_001"
                / "0.dcm"
            )
            marker.parent.mkdir(parents=True)
            marker.write_bytes(b"not-empty")

            with self.assertRaisesRegex(ValueError, "non-empty"):
                prepare_batch_data(
                    input_path,
                    output_path,
                    create_pipeline_folders=False,
                )

            self.assertEqual(marker.read_bytes(), b"not-empty")
            self.assertFalse((marker.parent / CANONICAL_CTA_RELATIVE_PATH).exists())


if __name__ == "__main__":
    unittest.main()

