import csv
import errno
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import xml.etree.ElementTree as ElementTree

import numpy
import SimpleITK as sitk


MODULE_DIRECTORY = Path(__file__).resolve().parent
GUI_ROOT = MODULE_DIRECTORY.parents[1]
FUNCTIONS_ROOT = GUI_ROOT.parent / "Functions_JZ"
for import_path in (GUI_ROOT, FUNCTIONS_ROOT):
    import_string = str(import_path)
    if import_string not in sys.path:
        sys.path.insert(0, import_string)


from Modules_SH.BatchSliceRadiomics import BatchSliceRadiomicsMain  # noqa: E402


EXPECTED_STATUS_FIELDS = (
    "run_xsection",
    "complete_segmentation",
    "LCx_xsection_CTA_exe_failed",
    "LCx_xsection_CTA_exe_er_msg",
    "LCx_xsection_CTA_complete",
    "mLAD_xsection_CTA_exe_failed",
    "mLAD_xsection_CTA_exe_er_msg",
    "mLAD_xsection_CTA_complete",
    "mRCA_xsection_CTA_exe_failed",
    "mRCA_xsection_CTA_exe_er_msg",
    "mRCA_xsection_CTA_complete",
    "complete_xsection_CTA",
    "LCx_xsection_seg_exe_failed",
    "LCx_xsection_seg_exe_er_msg",
    "LCx_xsection_seg_complete",
    "mLAD_xsection_seg_exe_failed",
    "mLAD_xsection_seg_exe_er_msg",
    "mLAD_xsection_seg_complete",
    "mRCA_xsection_seg_exe_failed",
    "mRCA_xsection_seg_exe_er_msg",
    "mRCA_xsection_seg_complete",
    "complete_xsection_seg",
    "LCx_lumen_complete",
    "LCx_lumen_er_msg",
    "mLAD_lumen_complete",
    "mLAD_lumen_er_msg",
    "mRCA_lumen_complete",
    "mRCA_lumen_er_msg",
    "complete_lumen_correction",
)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def read_csv_header(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file).fieldnames or [])


class CheckBoxStub:
    def __init__(self, checked):
        self.checked = bool(checked)

    def isChecked(self):
        return self.checked


def write_nifti(path, nonzero=True, value=1):
    """Write a small genuine NIfTI image suitable for semantic validation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = numpy.zeros((3, 4, 5), dtype=numpy.int16)
    if nonzero:
        array[1, 2, 3] = value
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.7, 0.8, 1.2))
    image.SetOrigin((1.0, 2.0, 3.0))
    sitk.WriteImage(image, str(path), True)
    return path


def xsection_record(case_id, batch_name, **updates):
    record = {
        "case_id": case_id,
        "batch_name": batch_name,
        "case_relative_path": "{0}/{1}".format(batch_name, case_id),
    }
    record.update(BatchSliceRadiomicsMain.XSECTION_STATUS_DEFAULTS)
    for field_name in BatchSliceRadiomicsMain.XSECTION_FINGERPRINT_FIELDS:
        record[field_name] = ""
    record.update(updates)
    return record


def merge_worker(manifest_path, record_path, result_queue):
    """Top-level worker so Windows' spawn multiprocessing can pickle it."""

    try:
        result = BatchSliceRadiomicsMain.merge_xsection_records(
            manifest_path,
            record_path,
        )
        result_queue.put((True, result["record_count"], result["row_count"]))
    except Exception as error:
        result_queue.put((False, repr(error), -1))


def artery_metadata(artery, label):
    return {
        "branch_name": artery,
        "label": [label],
        "use": True,
        "original_branch": artery + "_OriB_",
        "original_csv": artery + "Ext.csv",
        "3D_csv": artery + "3D.csv",
        "cross_section_image_LI": artery + "CTALI.nii.gz",
        "cross_section_mask_NN": artery + "ValNN.nii.gz",
        "cross_section_mask_fix": artery + "ValNNFixVal.nii.gz",
    }


class RecordingXsectionRunner:
    """Small injected CPR replacement which writes selected NIfTI outcomes."""

    def __init__(self, modes=None):
        self.modes = dict(modes or {})
        self.calls = []

    def __call__(self, arguments, timeout_seconds, output_limit):
        output_path = Path(arguments[3])
        mode = self.modes.get(output_path.name, "valid")
        self.calls.append(
            {
                "arguments": list(arguments),
                "output_name": output_path.name,
                "timeout_seconds": timeout_seconds,
                "output_limit": output_limit,
                "mode": mode,
            }
        )
        if mode in ("valid", "exe_failure_with_output"):
            write_nifti(output_path, nonzero=True)
        elif mode == "zero":
            write_nifti(output_path, nonzero=False)
        elif mode != "missing":
            raise AssertionError("Unknown fake Xsection mode: {0}".format(mode))

        if mode == "exe_failure_with_output":
            return {
                "exe_failed": True,
                "error_message": "injected vendor exit code 7",
            }
        return {"exe_failed": False, "error_message": ""}


class RecordingLumenRunner:
    """Injected one-row Lumen Correction replacement."""

    def __init__(self, modes=None):
        self.modes = dict(modes or {})
        self.calls = []

    def __call__(self, table_path):
        rows = read_csv(table_path)
        if len(rows) != 1:
            raise AssertionError("Lumen runner expected exactly one table row")
        row = rows[0]
        output_path = (
            Path(row["Output Directory"])
            / (row["Output Name"] + "Val.nii.gz")
        )
        mode = self.modes.get(output_path.name, "valid")
        self.calls.append(
            {
                "output_name": output_path.name,
                "mask_path": row["Mask Path"],
                "image_path": row["Image Path"],
                "mode": mode,
            }
        )
        if mode == "valid":
            write_nifti(output_path, nonzero=True)
        elif mode == "zero":
            write_nifti(output_path, nonzero=False)
        elif mode == "missing":
            return None
        elif mode == "failure":
            raise RuntimeError("injected lumen failure")
        else:
            raise AssertionError("Unknown fake lumen mode: {0}".format(mode))
        return None


class RecordingFilterMaskRunner:
    """Injected Filter Mask replacement creating both legacy output variants."""

    def __init__(self):
        self.calls = []

    def __call__(self, table_path):
        rows = read_csv(table_path)
        self.calls.append(
            {
                "row_count": len(rows),
                "file_names": [row["FileNameRef"] for row in rows],
            }
        )
        for row in rows:
            output_folder = Path(row["OutputFolder"])
            file_name_reference = row["FileNameRef"]
            write_nifti(
                output_folder / (file_name_reference + "Val.nii.gz"),
                nonzero=True,
            )
            write_nifti(
                output_folder / (file_name_reference + "NVl.nii.gz"),
                nonzero=True,
            )


def make_controller(
        temporary_directory,
        selected_arteries=("LCx", "mLAD", "mRCA"),
        xsection_runner=None,
        lumen_runner=None,
        filter_mask_runner=None,
        case_ids=("CASE_001",)):
    temporary_path = Path(temporary_directory)
    working_path = temporary_path / "0_Initial_Segmentation" / "batch_0"
    table_path = working_path / "Table"
    table_path.mkdir(parents=True)
    executable_path = temporary_path / "fake CPR executable.exe"
    executable_path.write_bytes(b"fake executable identity")
    lumen_source_path = temporary_path / "LCProcess.csv"
    lumen_source_path.write_text(
        "3D Connectivity Filter\nfirst\n2\n",
        encoding="utf-8",
    )

    runner = xsection_runner or RecordingXsectionRunner()
    correction_runner = lumen_runner or RecordingLumenRunner()
    controller = BatchSliceRadiomicsMain.BatchSliceRadiomics(
        xsection_runner=runner,
        lumen_runner=correction_runner,
        filter_mask_runner=filter_mask_runner,
    )
    controller.WorkingDir = str(working_path)
    controller.BatchName = working_path.name
    controller.TableDir = str(table_path)
    controller.ManifestPath = str(
        temporary_path / "records" / "BatchNIIDataPrepare_manifest.csv"
    )
    controller.FailedRootPath = str(temporary_path)
    controller.CPREXEPath = str(executable_path)
    controller.LumenProcessSourcePath = str(lumen_source_path)
    controller.SelectedArteries = list(selected_arteries)
    controller.XsectionTimeoutSeconds = 5
    controller.UpdateMsgLog = lambda _message="": None

    labels = {"LCx": 13, "mLAD": 7, "mRCA": 2}
    case_paths = []
    for case_id in case_ids:
        case_path = working_path / case_id
        controller.CreateFolders(str(case_path))
        case_paths.append(case_path)
        ori_data_path = case_path / "Preproc" / "OriData"
        branch_path = ori_data_path / "Branches"
        csv_path = case_path / "Preproc" / "CSV"
        write_nifti(ori_data_path / "CTA.nii.gz", nonzero=True, value=100)
        write_nifti(
            ori_data_path / "sc_labelled_coronary_segment.nii.gz",
            nonzero=True,
            value=13,
        )
        metadata = []
        for artery in selected_arteries:
            metadata.append(artery_metadata(artery, labels[artery]))
            write_nifti(
                branch_path / (artery + "_OriB_Val.nii.gz"),
                nonzero=True,
                value=labels[artery],
            )
            (csv_path / (artery + "Ext.csv")).write_text(
                "0,0,0\n1,1,1\n",
                encoding="utf-8",
            )
        (csv_path / "coronarySegmentsUse_BSRJZ.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

    controller.DCMDirList = [str(path) for path in case_paths]
    controller.StoreListInformation()
    controller._prepare_lumen_process()
    return controller, case_paths, runner, correction_runner


def set_case_artery_use(case_path, enabled_arteries):
    json_path = (
        Path(case_path) / "Preproc" / "CSV" /
        "coronarySegmentsUse_BSRJZ.json"
    )
    branches = json.loads(json_path.read_text(encoding="utf-8"))
    enabled = set(enabled_arteries)
    for branch in branches:
        branch["use"] = branch.get("branch_name") in enabled
    json_path.write_text(json.dumps(branches), encoding="utf-8")


class ResumePolicyTests(unittest.TestCase):
    def test_resume_state_matrix(self):
        classify = BatchSliceRadiomicsMain.BatchSliceRadiomics._resume_state
        interrupted = BatchSliceRadiomicsMain.XSECTION_INTERRUPTED_MESSAGE

        self.assertEqual(
            classify(0, 0, "", "", "current", interrupted),
            "pending",
        )
        self.assertEqual(
            classify(1, 0, "", "same", "same", interrupted),
            "complete",
        )
        self.assertEqual(
            classify(0, 1, "vendor failed", "same", "same", interrupted),
            "failed",
        )
        self.assertEqual(
            classify(0, 1, interrupted, "same", "same", interrupted),
            "interrupted",
        )
        self.assertEqual(
            classify(
                0,
                0,
                BatchSliceRadiomicsMain.FILTER_MASK_REGENERATION_MESSAGE,
                "same",
                "same",
                interrupted,
                (BatchSliceRadiomicsMain.FILTER_MASK_REGENERATION_MESSAGE,),
            ),
            "pending",
        )
        self.assertEqual(
            classify(
                0,
                1,
                interrupted + "; "
                + BatchSliceRadiomicsMain.FILTER_MASK_REGENERATION_MESSAGE,
                "same",
                "same",
                interrupted,
                (BatchSliceRadiomicsMain.FILTER_MASK_REGENERATION_MESSAGE,),
            ),
            "interrupted",
        )
        self.assertEqual(
            classify(0, 1, "vendor failed", "old", "new", interrupted),
            "changed",
        )
        self.assertEqual(
            classify(0, 0, "legacy failure", "", "new", interrupted),
            "failed",
        )

    def test_rerun_failed_checkbox_is_explicitly_unchecked_and_read(self):
        ui_path = GUI_ROOT / "ui" / "BatchSliceRadiomics.ui"
        root = ElementTree.parse(str(ui_path)).getroot()
        widget = root.find(
            ".//widget[@name='RerunFailedCases_CB_BSRJZ']"
        )
        self.assertIsNotNone(widget)
        checked = widget.find("./property[@name='checked']/bool")
        self.assertIsNotNone(checked)
        self.assertEqual((checked.text or "").strip().lower(), "false")

        controller = BatchSliceRadiomicsMain.BatchSliceRadiomics()
        self.assertFalse(controller.RerunFailedFlag)
        controller.ui = SimpleNamespace(
            RerunFailedCases_CB_BSRJZ=CheckBoxStub(True)
        )
        controller.CheckCheckBoxStatus()
        self.assertTrue(controller.RerunFailedFlag)


class NiftiValidationTests(unittest.TestCase):
    def test_missing_empty_directory_and_corrupt_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)

            valid, message = BatchSliceRadiomicsMain.validate_nifti(
                temporary_path / "missing.nii.gz"
            )
            self.assertFalse(valid)
            self.assertIn("Missing or empty", message)

            empty_path = temporary_path / "empty.nii.gz"
            empty_path.write_bytes(b"")
            valid, message = BatchSliceRadiomicsMain.validate_nifti(empty_path)
            self.assertFalse(valid)
            self.assertIn("Missing or empty", message)

            directory_path = temporary_path / "directory.nii.gz"
            directory_path.mkdir()
            valid, message = BatchSliceRadiomicsMain.validate_nifti(directory_path)
            self.assertFalse(valid)
            self.assertIn("Missing or empty", message)

            corrupt_path = temporary_path / "corrupt.nii.gz"
            corrupt_path.write_bytes(b"not-a-nifti")
            valid, message = BatchSliceRadiomicsMain.validate_nifti(corrupt_path)
            self.assertFalse(valid)
            self.assertIn("Unreadable NIfTI", message)

    def test_all_zero_nifti_is_valid_image_but_invalid_segmentation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = write_nifti(
                Path(temporary_directory) / "all_zero.nii.gz",
                nonzero=False,
            )

            valid_image, image_message = BatchSliceRadiomicsMain.validate_nifti(
                image_path,
                require_nonzero=False,
            )
            valid_seg, seg_message = BatchSliceRadiomicsMain.validate_nifti(
                image_path,
                require_nonzero=True,
            )

            self.assertTrue(valid_image, image_message)
            self.assertFalse(valid_seg)
            self.assertIn("no nonzero segmentation voxels", seg_message)

    def test_one_nonzero_voxel_is_a_valid_segmentation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = write_nifti(
                Path(temporary_directory) / "segmentation.nii.gz",
                nonzero=True,
                value=13,
            )

            valid, message = BatchSliceRadiomicsMain.validate_nifti(
                image_path,
                require_nonzero=True,
            )

            self.assertTrue(valid, message)
            image = sitk.ReadImage(str(image_path))
            self.assertEqual(tuple(image.GetSize()), (5, 4, 3))


class SafeCommandRunnerTests(unittest.TestCase):
    def test_success_and_nonzero_exit_are_reported_without_raising(self):
        success = BatchSliceRadiomicsMain.run_xsection_command(
            [sys.executable, "-c", "print('command ok')"],
            timeout_seconds=5,
            output_limit=100,
        )
        self.assertEqual(success["returncode"], 0)
        self.assertFalse(success["timed_out"])
        self.assertFalse(success["exe_failed"])
        self.assertEqual(success["error_message"], "")

        failure = BatchSliceRadiomicsMain.run_xsection_command(
            [
                sys.executable,
                "-c",
                "print('vendor failure'); raise SystemExit(7)",
            ],
            timeout_seconds=5,
            output_limit=100,
        )
        self.assertEqual(failure["returncode"], 7)
        self.assertFalse(failure["timed_out"])
        self.assertTrue(failure["exe_failed"])
        self.assertIn("return code 7", failure["error_message"])
        self.assertIn("vendor failure", failure["error_message"])

    def test_missing_executable_is_returned_as_a_failure(self):
        result = BatchSliceRadiomicsMain.run_xsection_command(
            [r"Z:\definitely_missing\cpr_not_present.exe", "argument"],
            timeout_seconds=5,
            output_limit=100,
        )

        self.assertIsNone(result["returncode"])
        self.assertFalse(result["timed_out"])
        self.assertTrue(result["exe_failed"])
        self.assertIn("Could not start process", result["error_message"])

    def test_timeout_is_returned_promptly(self):
        started_at = time.monotonic()
        result = BatchSliceRadiomicsMain.run_xsection_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=0.2,
            output_limit=100,
        )
        elapsed = time.monotonic() - started_at

        self.assertTrue(result["timed_out"])
        self.assertTrue(result["exe_failed"])
        self.assertIn("Timed out", result["error_message"])
        self.assertLess(elapsed, 10)

    def test_nonzero_output_keeps_the_requested_tail(self):
        result = BatchSliceRadiomicsMain.run_xsection_command(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "print('HEAD-' + 'x' * 500 + '-TAIL'); "
                    "sys.exit(9)"
                ),
            ],
            timeout_seconds=5,
            output_limit=40,
        )

        self.assertTrue(result["exe_failed"])
        self.assertIn("TAIL", result["error_message"])
        self.assertNotIn("HEAD", result["error_message"])

    @unittest.skipUnless(sys.platform == "win32", "Windows process-tree behavior")
    def test_timeout_terminates_descendant_process(self):
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil is unavailable")

        with tempfile.TemporaryDirectory() as temporary_directory:
            pid_path = Path(temporary_directory) / "child.pid"
            child_code = "import time; time.sleep(30)"
            parent_code = (
                "import pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-c', {0!r}]); "
                "pathlib.Path({1!r}).write_text(str(child.pid)); "
                "time.sleep(30)"
            ).format(child_code, str(pid_path))

            result = BatchSliceRadiomicsMain.run_xsection_command(
                [sys.executable, "-c", parent_code],
                timeout_seconds=0.6,
                output_limit=100,
            )

            self.assertTrue(result["timed_out"])
            self.assertTrue(pid_path.is_file())
            child_pid = int(pid_path.read_text(encoding="utf-8"))
            for _ in range(30):
                if not psutil.pid_exists(child_pid):
                    break
                time.sleep(0.1)
            self.assertFalse(psutil.pid_exists(child_pid))


class AtomicPersistenceRegressionTests(unittest.TestCase):
    def test_new_lock_and_atomic_csv_tolerate_fsync_ebadf(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            lock_path = temporary_path / "new.lock"
            csv_path = temporary_path / "record.csv"
            fsync_error = OSError(errno.EBADF, "network handle cannot fsync")

            with mock.patch.object(
                    BatchSliceRadiomicsMain.os,
                    "fsync",
                    side_effect=fsync_error) as fsync_mock:
                with BatchSliceRadiomicsMain.interprocess_file_lock(lock_path):
                    self.assertTrue(lock_path.is_file())
                BatchSliceRadiomicsMain._write_csv_atomic(
                    csv_path,
                    ("case_id", "complete"),
                    [{"case_id": "CASE_001", "complete": 1}],
                )

            # A new lock may avoid fsync entirely; the atomic CSV publication
            # must still exercise and tolerate the failing durability call.
            self.assertGreaterEqual(fsync_mock.call_count, 1)
            self.assertEqual(
                read_csv(csv_path),
                [{"case_id": "CASE_001", "complete": "1"}],
            )
            self.assertEqual(list(temporary_path.glob(".*.tmp")), [])

    def test_atomic_csv_retries_transient_replace_permission_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            csv_path = temporary_path / "record.csv"
            csv_path.write_text("case_id,complete\nOLD,0\n", encoding="utf-8")
            original_replace = BatchSliceRadiomicsMain.os.replace
            replace_calls = []

            def flaky_replace(source, destination):
                replace_calls.append((source, destination))
                if len(replace_calls) < 3:
                    raise PermissionError(
                        errno.EACCES,
                        "transient network sharing violation",
                    )
                return original_replace(source, destination)

            with mock.patch.object(
                    BatchSliceRadiomicsMain.os,
                    "replace",
                    side_effect=flaky_replace), mock.patch.object(
                        BatchSliceRadiomicsMain.time,
                        "sleep"):
                BatchSliceRadiomicsMain._write_csv_atomic(
                    csv_path,
                    ("case_id", "complete"),
                    [{"case_id": "CASE_NEW", "complete": 1}],
                )

            self.assertEqual(len(replace_calls), 3)
            self.assertEqual(
                read_csv(csv_path),
                [{"case_id": "CASE_NEW", "complete": "1"}],
            )
            self.assertEqual(list(temporary_path.glob(".*.tmp")), [])

    def test_permanent_replace_failure_is_bounded_and_preserves_old_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            csv_path = temporary_path / "record.csv"
            csv_path.write_text("case_id,complete\nKEEP,1\n", encoding="utf-8")
            original_bytes = csv_path.read_bytes()
            permanent_error = PermissionError(
                errno.EACCES,
                "persistent network sharing violation",
            )

            with mock.patch.object(
                    BatchSliceRadiomicsMain.os,
                    "replace",
                    side_effect=permanent_error) as replace_mock, mock.patch.object(
                        BatchSliceRadiomicsMain.time,
                        "sleep"):
                with self.assertRaises(PermissionError):
                    BatchSliceRadiomicsMain._write_csv_atomic(
                        csv_path,
                        ("case_id", "complete"),
                        [{"case_id": "MUST_NOT_PUBLISH", "complete": 0}],
                    )

            self.assertGreaterEqual(replace_mock.call_count, 2)
            self.assertLessEqual(replace_mock.call_count, 10)
            self.assertEqual(csv_path.read_bytes(), original_bytes)
            self.assertEqual(list(temporary_path.glob(".*.tmp")), [])


class XsectionRecordTests(unittest.TestCase):
    def test_confirmed_status_schema_has_no_deprecated_output_failed_fields(self):
        self.assertEqual(
            BatchSliceRadiomicsMain.XSECTION_STATUS_FIELDS,
            EXPECTED_STATUS_FIELDS,
        )
        self.assertFalse(
            any(
                field_name.endswith("_CTA_failed")
                or field_name.endswith("_seg_failed")
                for field_name in BatchSliceRadiomicsMain.XSECTION_STATUS_FIELDS
            )
        )
        for field_name in EXPECTED_STATUS_FIELDS:
            expected_default = "" if field_name.endswith("_er_msg") else 0
            self.assertEqual(
                BatchSliceRadiomicsMain.XSECTION_STATUS_DEFAULTS[field_name],
                expected_default,
            )

    def test_batch_ledger_is_atomic_sorted_and_replaces_prior_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            record_path = (
                Path(temporary_directory)
                / "batch_0"
                / "Table"
                / "xsection_record"
                / "batch_xsection.csv"
            )
            records = [
                xsection_record(
                    "CASE_B",
                    "batch_0",
                    run_xsection=1,
                    LCx_xsection_CTA_exe_failed=1,
                    LCx_xsection_CTA_exe_er_msg="exit code 3",
                ),
                xsection_record(
                    "CASE_A",
                    "batch_0",
                    run_xsection=1,
                    LCx_xsection_CTA_complete=1,
                ),
            ]

            BatchSliceRadiomicsMain.write_xsection_records(record_path, records)

            first_rows = read_csv(record_path)
            self.assertEqual(
                [row["case_id"] for row in first_rows],
                ["CASE_A", "CASE_B"],
            )
            self.assertEqual(
                read_csv_header(record_path),
                list(BatchSliceRadiomicsMain.XSECTION_RECORD_FIELDS),
            )

            replacement = xsection_record(
                "CASE_B",
                "batch_0",
                run_xsection=1,
                LCx_xsection_CTA_complete=1,
                complete_xsection_CTA=1,
            )
            BatchSliceRadiomicsMain.write_xsection_records(
                record_path,
                [replacement],
            )

            rows = read_csv(record_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["case_id"], "CASE_B")
            self.assertEqual(rows[0]["LCx_xsection_CTA_complete"], "1")
            self.assertEqual(rows[0]["LCx_xsection_CTA_exe_er_msg"], "")
            self.assertTrue(Path(str(record_path) + ".lock").is_file())
            self.assertEqual(list(record_path.parent.glob(".*.tmp")), [])

    def test_manifest_merge_preserves_upstream_columns_and_hides_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            manifest_path = temporary_path / "BatchNIIDataPrepare_manifest.csv"
            record_path = (
                temporary_path
                / "batch_0"
                / "Table"
                / "xsection_record"
                / "batch_xsection.csv"
            )
            with manifest_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=(
                        "case_id",
                        "batch_name",
                        "case_relative_path",
                        "source_sha256",
                        "run_segmentation",
                    ),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": "CASE_001",
                        "batch_name": "batch_0",
                        "case_relative_path": "batch_0/CASE_001",
                        "source_sha256": "preserve-this-hash",
                        "run_segmentation": "1",
                    }
                )

            record = xsection_record(
                "CASE_001",
                "batch_0",
                run_xsection=1,
                complete_segmentation=1,
                LCx_xsection_CTA_complete=1,
                LCx_xsection_CTA_fingerprint="local-only-fingerprint",
            )
            BatchSliceRadiomicsMain.write_xsection_records(record_path, [record])

            result = BatchSliceRadiomicsMain.merge_xsection_records(
                manifest_path,
                record_path,
            )

            self.assertEqual(result["record_count"], 1)
            self.assertEqual(result["row_count"], 1)
            rows = read_csv(manifest_path)
            self.assertEqual(rows[0]["source_sha256"], "preserve-this-hash")
            self.assertEqual(rows[0]["run_segmentation"], "1")
            self.assertEqual(rows[0]["run_xsection"], "1")
            self.assertEqual(rows[0]["complete_segmentation"], "1")
            self.assertEqual(rows[0]["LCx_xsection_CTA_complete"], "1")
            self.assertNotIn("LCx_xsection_CTA_fingerprint", rows[0])
            self.assertTrue(Path(str(manifest_path) + ".lock").is_file())
            self.assertEqual(list(manifest_path.parent.glob(".*.tmp")), [])

    def test_missing_manifest_is_created_and_unique_legacy_case_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            manifest_path = temporary_path / "records" / "manifest.csv"
            first_record_path = temporary_path / "batch_0.csv"
            second_record_path = temporary_path / "batch_1.csv"

            BatchSliceRadiomicsMain.write_xsection_records(
                first_record_path,
                [
                    xsection_record(
                        "CASE_NEW",
                        "batch_0",
                        run_xsection=1,
                        complete_xsection_CTA=1,
                    )
                ],
            )
            BatchSliceRadiomicsMain.merge_xsection_records(
                manifest_path,
                first_record_path,
            )
            self.assertEqual(len(read_csv(manifest_path)), 1)

            # Simulate an older manifest whose one matching case has no batch name.
            legacy_rows = read_csv(manifest_path)
            legacy_rows[0]["batch_name"] = ""
            BatchSliceRadiomicsMain._write_csv_atomic(
                manifest_path,
                read_csv_header(manifest_path),
                legacy_rows,
            )
            BatchSliceRadiomicsMain.write_xsection_records(
                second_record_path,
                [
                    xsection_record(
                        "CASE_NEW",
                        "batch_1",
                        run_xsection=1,
                        complete_xsection_seg=1,
                    )
                ],
            )
            BatchSliceRadiomicsMain.merge_xsection_records(
                manifest_path,
                second_record_path,
            )

            rows = read_csv(manifest_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["case_id"], "CASE_NEW")
            self.assertEqual(rows[0]["complete_xsection_seg"], "1")

    def test_manifest_without_identity_columns_is_rejected_without_replacement(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            manifest_path = temporary_path / "manifest.csv"
            record_path = temporary_path / "record.csv"
            manifest_path.write_text("case_id,value\nCASE_1,keep\n", encoding="utf-8")
            original_bytes = manifest_path.read_bytes()
            BatchSliceRadiomicsMain.write_xsection_records(
                record_path,
                [xsection_record("CASE_1", "batch_0", run_xsection=1)],
            )

            with self.assertRaisesRegex(ValueError, "batch_name"):
                BatchSliceRadiomicsMain.merge_xsection_records(
                    manifest_path,
                    record_path,
                )

            self.assertEqual(manifest_path.read_bytes(), original_bytes)

    def test_five_processes_merge_batch_ledgers_without_lost_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            manifest_path = temporary_path / "BatchNIIDataPrepare_manifest.csv"
            record_paths = []
            for batch_number in range(5):
                batch_name = "batch_{0}".format(batch_number)
                record_path = (
                    temporary_path
                    / batch_name
                    / "Table"
                    / "xsection_record"
                    / "batch_xsection.csv"
                )
                records = [
                    xsection_record(
                        "CASE_{0}_{1:03d}".format(batch_number, case_number),
                        batch_name,
                        run_xsection=1,
                        complete_segmentation=1,
                        complete_xsection_CTA=1,
                        complete_xsection_seg=1,
                        complete_lumen_correction=1,
                    )
                    for case_number in range(20)
                ]
                BatchSliceRadiomicsMain.write_xsection_records(record_path, records)
                record_paths.append(record_path)

            context = multiprocessing.get_context("spawn")
            result_queue = context.Queue()
            processes = [
                context.Process(
                    target=merge_worker,
                    args=(str(manifest_path), str(record_path), result_queue),
                )
                for record_path in record_paths
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(30)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                    self.fail("manifest merge deadlocked")
                self.assertEqual(process.exitcode, 0)

            results = [result_queue.get(timeout=5) for _ in processes]
            self.assertTrue(all(result[0] for result in results), results)
            rows = read_csv(manifest_path)
            self.assertEqual(len(rows), 100)
            self.assertEqual(
                len({(row["batch_name"], row["case_id"]) for row in rows}),
                100,
            )
            self.assertTrue(all(row["run_xsection"] == "1" for row in rows))


class HybridParallelBatchTests(unittest.TestCase):
    def test_zero_byte_dicom_marker_is_not_treated_as_dicom_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            case_path = Path(temporary_directory) / "CASE_DICOM"
            case_path.mkdir()
            marker_path = case_path / "discovery.dcm"
            marker_path.write_bytes(b"")
            self.assertFalse(
                BatchSliceRadiomicsMain._contains_dicom(case_path)
            )
            marker_path.write_bytes(b"non-empty test DICOM placeholder")
            self.assertTrue(
                BatchSliceRadiomicsMain._contains_dicom(case_path)
            )

    """Pure synthetic coverage for hybrid discovery and coordination."""

    @staticmethod
    def _make_prepared_case(parent_path, case_id):
        case_path = Path(parent_path) / case_id
        input_path = case_path / "Preproc" / "OriData" / "CTA.nii.gz"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(b"synthetic prepared CTA")
        return case_path

    @staticmethod
    def _write_manifest(manifest_path, rows):
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        field_names = [
            "case_id",
            "batch_name",
            "case_relative_path",
            "upstream_note",
        ]
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(rows)
        return manifest_path

    def test_nested_discovery_preserves_exact_identity_and_sorts_largest_first(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root_path = Path(temporary_directory) / "0_Initial_Segmentation"
            large_path = root_path / "batch_10"
            small_path = root_path / "batch_2"
            empty_path = root_path / "batch_empty"
            for case_id in ("CASE_C", "CASE_A", "CASE_B"):
                self._make_prepared_case(large_path, case_id)
            self._make_prepared_case(small_path, "CASE_A")
            (empty_path / "not_a_case").mkdir(parents=True)
            (root_path / "notes").mkdir(parents=True)

            manifest_path = self._write_manifest(
                root_path / "BatchNIIDataPrepare_manifest.csv",
                [
                    {
                        "case_id": "CASE_A",
                        "batch_name": "batch_10",
                        "case_relative_path": "batch_10/CASE_A",
                        "upstream_note": "in scope",
                    },
                    {
                        "case_id": "MOVED_CASE",
                        "batch_name": "batch_10",
                        "case_relative_path": "batch_10/MOVED_CASE",
                        "upstream_note": "manifest only",
                    },
                    {
                        "case_id": "UNRELATED",
                        "batch_name": "legacy_batch",
                        "case_relative_path": "legacy_batch/UNRELATED",
                        "upstream_note": "out of scope",
                    },
                ],
            )

            discovery = BatchSliceRadiomicsMain.discover_batch_slice_work(
                root_path,
                manifest_path=manifest_path,
                scan_dicom=False,
            )

            self.assertEqual(discovery.layout, "nested_batches")
            self.assertEqual(discovery.discovered_count, 4)
            self.assertEqual(discovery.manifest_only_count, 1)
            self.assertEqual(discovery.out_of_scope_count, 1)
            self.assertEqual(discovery.excluded_count, 2)
            self.assertEqual(
                [unit.batch_name for unit in discovery.work_units],
                ["batch_10", "batch_2"],
            )
            self.assertEqual(
                [unit.case_count for unit in discovery.work_units],
                [3, 1],
            )
            self.assertEqual(
                discovery.work_units[0].case_ids,
                ("CASE_A", "CASE_B", "CASE_C"),
            )
            self.assertEqual(
                discovery.work_units[1].case_ids,
                ("CASE_A",),
            )
            self.assertNotEqual(
                discovery.work_units[0].unit_id,
                discovery.work_units[1].unit_id,
            )

    def test_flat_and_single_batch_roots_schedule_independent_case_units(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root_path = Path(temporary_directory) / "batch_7"
            self._make_prepared_case(root_path, "CASE_002")
            self._make_prepared_case(root_path, "CASE_001")
            (root_path / "not_a_case").mkdir(parents=True)
            (root_path / "Table").mkdir(parents=True)

            discovery = BatchSliceRadiomicsMain.discover_batch_slice_work(
                root_path,
                scan_dicom=False,
            )

            self.assertEqual(discovery.layout, "direct_cases")
            self.assertEqual(discovery.discovered_count, 2)
            self.assertEqual(discovery.excluded_count, 1)
            self.assertEqual(
                [unit.case_ids for unit in discovery.work_units],
                [("CASE_001",), ("CASE_002",)],
            )
            self.assertTrue(all(
                unit.batch_name == "batch_7"
                for unit in discovery.work_units
            ))
            self.assertEqual(len({
                unit.record_path for unit in discovery.work_units
            }), 2)
            self.assertEqual(len({
                unit.table_dir for unit in discovery.work_units
            }), 2)
            self.assertTrue(all(
                Path(unit.table_dir).parent
                == root_path / "Table" / "case_workers"
                for unit in discovery.work_units
            ))
            self.assertTrue(all(
                Path(unit.record_path).parent
                == root_path / "Table" / "xsection_record" / "pending"
                for unit in discovery.work_units
            ))
            self.assertEqual(len({
                unit.canonical_record_path for unit in discovery.work_units
            }), 1)
            self.assertTrue(all(
                Path(unit.canonical_record_path)
                == root_path / "Table" / "xsection_record" /
                "batch_xsection.csv"
                for unit in discovery.work_units
            ))

    def test_parallel_worker_limit_respects_request_units_and_logical_cpus(self):
        clamp = BatchSliceRadiomicsMain.clamp_parallel_workers

        self.assertEqual(clamp(32, 100, logical_cpus=24), 24)
        self.assertEqual(clamp(24, 7, logical_cpus=64), 7)
        self.assertEqual(clamp(0, 7, logical_cpus=64), 1)
        self.assertEqual(clamp("invalid", 100, logical_cpus=64), 4)
        self.assertEqual(clamp(32, 0, logical_cpus=64), 0)

    def test_direct_case_stages_only_its_rows_from_shared_tables(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root_path = Path(temporary_directory) / "flat_root"
            case_paths = {
                case_id: self._make_prepared_case(root_path, case_id)
                for case_id in ("CASE_001", "CASE_002")
            }
            shared_table_path = root_path / "Table"
            table_writer = BatchSliceRadiomicsMain.BatchSliceRadiomics._write_table

            filter_rows = []
            xsection_rows = []
            fix_rows = []
            for case_id, case_path in case_paths.items():
                preproc_path = case_path / "Preproc"
                ori_data_path = preproc_path / "OriData"
                csv_path = preproc_path / "CSV"
                xsection_path = preproc_path / "Xsection"
                fix_path = preproc_path / "XsectionFix"
                filter_rows.append({
                    "MaskPath": str(
                        ori_data_path / "sc_labelled_coronary_segment.nii.gz"
                    ),
                    "MaskVal": "13",
                    "NewMaskVal": "1",
                    "OutputFolder": str(ori_data_path / "Branches"),
                    "FileNameRef": "LCx_OriB_",
                })
                xsection_rows.append({
                    "inputDir": str(ori_data_path),
                    "inputCSVDir": str(csv_path),
                    "outputDir": str(xsection_path),
                    "outputCSVDir": str(csv_path),
                    "resampleDepth": "0.1",
                    "CPRAngle": "0",
                    "cpus": "91",
                    "casePath": str(
                        shared_table_path / (case_id + "_Xsection.csv")
                    ),
                    "EXEPath": "stale.exe",
                })
                fix_rows.append({
                    "Image Path": str(xsection_path / "LCxCTALI.nii.gz"),
                    "Mask Path": str(xsection_path / "LCxValNN.nii.gz"),
                    "Process Path": str(shared_table_path / "LCProcess.csv"),
                    "Fill Original Value": "No",
                    "Output Directory": str(fix_path),
                    "Output Name": "LCxValNNFix",
                })

            table_writer(
                shared_table_path / "FilterMask_BSRJZ.csv",
                BatchSliceRadiomicsMain.BatchSliceRadiomics.FILTER_MASK_HEADERS,
                filter_rows,
            )
            table_writer(
                shared_table_path / "Xsection_BSRJZ.csv",
                BatchSliceRadiomicsMain.BatchSliceRadiomics.XSECTION_HEADERS,
                xsection_rows,
            )
            table_writer(
                shared_table_path / "XsectionFix_BSRJZ.csv",
                BatchSliceRadiomicsMain.BatchSliceRadiomics.XSECTION_FIX_HEADERS,
                fix_rows,
            )
            config = BatchSliceRadiomicsMain.BatchSliceRunConfig(
                root_path=str(root_path),
                manifest_path=str(
                    root_path / "BatchNIIDataPrepare_manifest.csv"
                ),
                failed_root_path=str(Path(temporary_directory) / "failed"),
                cpr_exe_path="current.exe",
                lumen_process_path="LCProcess.csv",
                generate_tables=False,
                selected_arteries=("LCx",),
            )
            discovery = BatchSliceRadiomicsMain.discover_batch_slice_work(config)
            case_unit = next(
                unit for unit in discovery.work_units
                if unit.case_ids == ("CASE_001",)
            )
            controller = BatchSliceRadiomicsMain.BatchSliceRadiomics()

            controller._apply_run_config(config, case_unit)

            staged_tables = (
                (
                    Path(controller.FilterMaskTablePath),
                    controller.FILTER_MASK_HEADERS,
                ),
                (
                    Path(controller.XsectionTablePath),
                    controller.XSECTION_HEADERS,
                ),
                (
                    Path(controller.XsectionFixTablePath),
                    controller.XSECTION_FIX_HEADERS,
                ),
            )
            self.assertTrue(all(
                table_path.parent == Path(case_unit.table_dir)
                for table_path, _ in staged_tables
            ))
            for table_path, expected_headers in staged_tables:
                self.assertEqual(read_csv_header(table_path), list(expected_headers))
                rows = read_csv(table_path)
                self.assertEqual(len(rows), 1)
                staged_text = " ".join(rows[0].values())
                self.assertIn("CASE_001", staged_text)
                self.assertNotIn("CASE_002", staged_text)

    def test_runtime_executable_and_inner_worker_override_stale_table_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, _, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
            )
            controller.MaskXsectionFlag = True
            controller.GenerateTable(
                cpus="97",
                maskXsection=True,
                maskXsectionOnly=False,
            )
            table_path = Path(controller.XsectionTablePath)
            field_names = read_csv_header(table_path)
            rows = read_csv(table_path)
            for row in rows:
                row["EXEPath"] = str(
                    Path(temporary_directory) / "stale CPR.exe"
                )
                row["cpus"] = "97"
            with table_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=field_names)
                writer.writeheader()
                writer.writerows(rows)

            runtime_executable = (
                Path(temporary_directory) / "current CPR.exe"
            )
            runtime_executable.write_bytes(b"current executable")
            controller.CPREXEPath = str(runtime_executable)
            controller.InnerXsectionWorkers = 1

            jobs = controller._load_xsection_jobs()

            self.assertEqual(len(jobs), 2)
            self.assertTrue(all(
                job["arguments"][0] == str(runtime_executable)
                for job in jobs.values()
            ))
            self.assertTrue(all(
                job["workers"] == 1 for job in jobs.values()
            ))
            self.assertFalse(any(
                "stale CPR.exe" in argument
                for job in jobs.values()
                for argument in job["arguments"]
            ))

    def test_synchronous_headless_worker_starts_with_all_stages_disabled(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root_path = Path(temporary_directory) / "flat_root"
            self._make_prepared_case(root_path, "CASE_001")
            manifest_path = root_path / "BatchNIIDataPrepare_manifest.csv"
            config = BatchSliceRadiomicsMain.BatchSliceRunConfig(
                root_path=str(root_path),
                manifest_path=str(manifest_path),
                failed_root_path=str(Path(temporary_directory) / "failed"),
                cpr_exe_path=str(Path(temporary_directory) / "unused.exe"),
                lumen_process_path=str(
                    Path(temporary_directory) / "unused.csv"
                ),
                max_workers=1,
                create_folders=False,
                branch_dictionary=False,
                generate_tables=False,
                filter_mask=False,
                xsection=False,
                lumen_correction=False,
                selected_arteries=("LCx",),
            )
            discovery = BatchSliceRadiomicsMain.discover_batch_slice_work(
                config
            )
            work_unit = discovery.work_units[0]
            prior_itk_threads = (
                sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
            )
            prior_environment = {
                name: os.environ.get(name)
                for name in BatchSliceRadiomicsMain._WORKER_THREAD_ENVIRONMENT
            }
            try:
                result = BatchSliceRadiomicsMain.run_batch_slice_work_unit(
                    config,
                    work_unit,
                )
            finally:
                sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(
                    prior_itk_threads
                )
                for name, prior_value in prior_environment.items():
                    if prior_value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = prior_value

            self.assertEqual(result.error_message, "")
            self.assertEqual(result.output_tail, "")
            self.assertEqual(result.case_count, 1)
            self.assertTrue(Path(result.record_path).is_file())
            records = read_csv(result.record_path)
            self.assertEqual(
                [(row["batch_name"], row["case_id"]) for row in records],
                [("flat_root", "CASE_001")],
            )

    def test_pending_compaction_preserves_manifest_only_rows_and_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root_path = Path(temporary_directory)
            canonical_path = (
                root_path / "Table" / "xsection_record" /
                "batch_xsection.csv"
            )
            pending_path = canonical_path.parent / "pending" / "case_b.csv"
            manifest_path = self._write_manifest(
                root_path / "BatchNIIDataPrepare_manifest.csv",
                [
                    {
                        "case_id": "CASE_A",
                        "batch_name": "flat_root",
                        "case_relative_path": "flat_root/CASE_A",
                        "upstream_note": "keep A metadata",
                    },
                    {
                        "case_id": "UNRELATED",
                        "batch_name": "legacy_batch",
                        "case_relative_path": "legacy_batch/UNRELATED",
                        "upstream_note": "must survive",
                    },
                ],
            )
            BatchSliceRadiomicsMain.write_xsection_records(
                canonical_path,
                [xsection_record(
                    "CASE_A",
                    "flat_root",
                    complete_xsection_CTA=1,
                )],
            )
            BatchSliceRadiomicsMain.write_xsection_records(
                pending_path,
                [xsection_record(
                    "CASE_B",
                    "flat_root",
                    complete_xsection_seg=1,
                )],
            )

            result = BatchSliceRadiomicsMain.compact_pending_ledgers(
                canonical_path,
                manifest_path=manifest_path,
            )

            self.assertEqual(result["pending_file_count"], 1)
            self.assertEqual(result["errors"], ())
            self.assertFalse(pending_path.exists())
            canonical_rows = read_csv(canonical_path)
            self.assertEqual(
                [(row["batch_name"], row["case_id"]) for row in canonical_rows],
                [("flat_root", "CASE_A"), ("flat_root", "CASE_B")],
            )
            manifest_rows = read_csv(manifest_path)
            by_identity = {
                (row["batch_name"], row["case_id"]): row
                for row in manifest_rows
            }
            self.assertEqual(len(manifest_rows), 3)
            self.assertEqual(
                by_identity[("legacy_batch", "UNRELATED")]["upstream_note"],
                "must survive",
            )
            self.assertEqual(
                by_identity[("flat_root", "CASE_A")]["upstream_note"],
                "keep A metadata",
            )
            self.assertEqual(
                by_identity[("flat_root", "CASE_B")]["complete_xsection_seg"],
                "1",
            )

    def test_progress_is_monotonic_deduplicated_and_counts_terminal_cases(self):
        units = (
            BatchSliceRadiomicsMain.BatchSliceWorkUnit(
                unit_id="batch:batch_0",
                layout="nested_batches",
                working_dir="batch_0",
                batch_name="batch_0",
                case_paths=("batch_0/CASE_A", "batch_0/CASE_B"),
                table_dir="batch_0/Table",
                record_path="batch_0/Table/record.csv",
                canonical_record_path="batch_0/Table/record.csv",
            ),
        )
        discovery = BatchSliceRadiomicsMain.BatchSliceDiscovery(
            layout="nested_batches",
            work_units=units,
            discovered_count=2,
            excluded_count=0,
            manifest_only_count=0,
            out_of_scope_count=0,
            scope_batch_names=("batch_0",),
        )
        stages = ("Filter Mask", "Xsection", "Lumen Correction")
        tracker = BatchSliceRadiomicsMain.BatchSliceProgressTracker(
            discovery,
            stages,
            started_at=time.monotonic() - 1.0,
        )
        percentages = [tracker.snapshot().percent]
        tracker.consume(BatchSliceRadiomicsMain.BatchSliceProgress(
            kind="unit_started",
            unit_id="batch:batch_0",
        ))
        for case_id, state in (("CASE_A", "completed"), ("CASE_B", "skipped")):
            for stage in stages:
                snapshot = tracker.consume(
                    BatchSliceRadiomicsMain.BatchSliceProgress(
                        kind="case_stage",
                        batch_name="batch_0",
                        case_id=case_id,
                        stage=stage,
                        state=state,
                    )
                )
                percentages.append(snapshot.percent)
        duplicate = tracker.consume(BatchSliceRadiomicsMain.BatchSliceProgress(
            kind="case_stage",
            batch_name="batch_0",
            case_id="CASE_A",
            stage="Xsection",
            state="failed",
        ))
        tracker.consume(BatchSliceRadiomicsMain.BatchSliceProgress(
            kind="unit_finished",
            unit_id="batch:batch_0",
        ))
        final_snapshot = BatchSliceRadiomicsMain.summarize_progress(tracker)

        self.assertEqual(percentages, sorted(percentages))
        self.assertEqual(duplicate.done_steps, 6)
        self.assertEqual(final_snapshot.percent, 100)
        self.assertEqual(final_snapshot.completed_cases, 1)
        self.assertEqual(final_snapshot.skipped_cases, 1)
        self.assertEqual(final_snapshot.failed_cases, 0)
        self.assertEqual(final_snapshot.active_workers, 0)
        self.assertGreaterEqual(final_snapshot.elapsed_seconds, 1.0)
        self.assertEqual(final_snapshot.eta_seconds, 0.0)

    def test_enabled_stage_order_is_serial_and_stable(self):
        config = BatchSliceRadiomicsMain.BatchSliceRunConfig(
            root_path="root",
            manifest_path="manifest.csv",
            failed_root_path="failed",
            cpr_exe_path="cpr.exe",
            lumen_process_path="LCProcess.csv",
        )

        self.assertEqual(
            config.enabled_stages,
            ("Filter Mask", "Xsection", "Lumen Correction"),
        )

    def test_legacy_lumen_path_is_restored_and_fallback_exists(self):
        ui_path = GUI_ROOT / "ui" / "BatchSliceRadiomics.ui"
        root = ElementTree.parse(str(ui_path)).getroot()
        widget = root.find(
            ".//widget[@name='LumenCorrectionPath_Txt_BSRJZ']"
        )
        self.assertIsNotNone(widget)
        html_element = widget.find("./property[@name='html']/string")
        self.assertIsNotNone(html_element)
        html_text = html_element.text or ""
        paragraph_content = html_text.rsplit("<p ", 1)[1]
        paragraph_content = paragraph_content.split(">", 1)[1]
        paragraph_content = paragraph_content.split("</p>", 1)[0]
        self.assertEqual(
            paragraph_content.strip(),
            "D:/Software/HymoV01_270424/Base/Tables/LC/LCProcess.csv",
        )

        default_path = Path(
            BatchSliceRadiomicsMain.BatchSliceRadiomics
            ._default_lumen_process_path()
        )
        self.assertTrue(default_path.is_file())
        self.assertEqual(default_path.name, "LCProcess.csv")
        self.assertEqual(default_path.parent.name, "LC")

        expected_checkbox_defaults = {
            "ScanFile_CB_BSRJZ": "false",
            "CreateFolders_CB_BSRJZ": "true",
            "BranchDict_CB_BSRJZ": "false",
            "GenerateTable_CB_BSRJZ": "true",
            "FM_CB_BSRJZ": "false",
            "Xsection_CB_BSRJZ": "true",
            "LumenCorrection_CB_BSRJZ": "true",
            "maskXsection_CB_BSRJZ": "true",
            "maskXsectionOnly_CB_BSRJZ": "false",
        }
        for widget_name, expected_value in expected_checkbox_defaults.items():
            checkbox = root.find(
                ".//widget[@name='{0}']".format(widget_name)
            )
            self.assertIsNotNone(checkbox, widget_name)
            checked = checkbox.find("./property[@name='checked']/bool")
            self.assertIsNotNone(checked, widget_name)
            self.assertEqual(
                (checked.text or "").strip().lower(),
                expected_value,
                widget_name,
            )


class BatchSliceRadiomicsControllerTests(unittest.TestCase):
    def test_cta_scan_prefers_ori_data_and_supports_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            working_path = Path(temporary_directory) / "batch_0"
            table_path = working_path / "Table"
            table_path.mkdir(parents=True)
            preferred_case = working_path / "CASE_001"
            legacy_case = working_path / "CASE_002"
            ignored_case = working_path / "CASE_003"
            preferred_path = write_nifti(
                preferred_case / "Preproc" / "OriData" / "CTA.nii.gz"
            )
            legacy_path = write_nifti(
                legacy_case / "CTA" / "CTA.nii.gz"
            )
            ignored_case.mkdir()

            controller = BatchSliceRadiomicsMain.BatchSliceRadiomics()
            controller.WorkingDir = str(working_path)
            controller.TableDir = str(table_path)
            controller.ScanCTADir()
            controller.StoreListInformation()

            self.assertEqual(
                controller.DCMDirList,
                [str(preferred_case), str(legacy_case)],
            )
            self.assertEqual(
                controller.CTADirList,
                [str(preferred_path), str(legacy_path)],
            )

    def test_suffixed_tables_have_exact_headers_rows_and_output_names_by_mode(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths, _, _ = make_controller(temporary_directory)
            case_path = case_paths[0]
            unsuffixed_table = Path(controller.TableDir) / "Xsection.csv"
            unsuffixed_case = case_path / "Preproc" / "CSV" / "case.csv"
            unsuffixed_table.write_text("preserve-table", encoding="utf-8")
            unsuffixed_case.write_text("preserve-case", encoding="utf-8")

            mode_expectations = (
                ("CTA-only", False, False, 3, 0, 0),
                ("CTA-and-NN", True, False, 3, 3, 3),
                ("mask-only", True, True, 0, 3, 3),
            )
            for (
                    mode_name,
                    mask_xsection,
                    mask_only,
                    expected_cta_rows,
                    expected_seg_rows,
                    expected_fix_rows) in mode_expectations:
                with self.subTest(mode=mode_name):
                    successful_cases = controller.GenerateTable(
                        resampleDepth="0.2",
                        cpus="3",
                        CPRAngle="5",
                        maskXsection=mask_xsection,
                        maskXsectionOnly=mask_only,
                    )
                    self.assertEqual(successful_cases, [str(case_path)])

                    filter_path = Path(controller.FilterMaskTablePath)
                    xsection_path = Path(controller.XsectionTablePath)
                    fix_path = Path(controller.XsectionFixTablePath)
                    case_table_path = (
                        case_path / "Preproc" / "CSV" / "case_BSRJZ.csv"
                    )
                    self.assertEqual(
                        read_csv_header(filter_path),
                        list(controller.FILTER_MASK_HEADERS),
                    )
                    self.assertEqual(
                        read_csv_header(xsection_path),
                        list(controller.XSECTION_HEADERS),
                    )
                    self.assertEqual(
                        read_csv_header(case_table_path),
                        list(controller.CASE_HEADERS),
                    )
                    self.assertEqual(
                        read_csv_header(fix_path),
                        list(controller.XSECTION_FIX_HEADERS),
                    )
                    self.assertEqual(filter_path.name, "FilterMask_BSRJZ.csv")
                    self.assertEqual(xsection_path.name, "Xsection_BSRJZ.csv")
                    self.assertEqual(fix_path.name, "XsectionFix_BSRJZ.csv")
                    self.assertEqual(case_table_path.name, "case_BSRJZ.csv")

                    case_rows = read_csv(case_table_path)
                    cta_rows = [
                        row for row in case_rows
                        if row["outputImagePath"].endswith("CTALI.nii.gz")
                    ]
                    seg_rows = [
                        row for row in case_rows
                        if row["outputImagePath"].endswith("ValNN.nii.gz")
                    ]
                    self.assertEqual(len(cta_rows), expected_cta_rows)
                    self.assertEqual(len(seg_rows), expected_seg_rows)
                    self.assertEqual(
                        {row["outputImagePath"] for row in cta_rows},
                        {
                            artery + "CTALI.nii.gz"
                            for artery in controller.SelectedArteries
                        } if expected_cta_rows else set(),
                    )
                    self.assertEqual(
                        {row["outputImagePath"] for row in seg_rows},
                        {
                            artery + "ValNN.nii.gz"
                            for artery in controller.SelectedArteries
                        } if expected_seg_rows else set(),
                    )
                    self.assertTrue(
                        all(
                            row["Interpolation"] == "0"
                            and row["OutCSV"] == "0"
                            for row in cta_rows
                        )
                    )
                    self.assertTrue(
                        all(
                            row["Interpolation"] == "1"
                            and row["OutCSV"] == "0"
                            for row in seg_rows
                        )
                    )
                    self.assertTrue(
                        all(row["inputImagePath"] == "CTA.nii.gz" for row in cta_rows)
                    )
                    self.assertTrue(
                        all(
                            Path(row["inputImagePath"]).parts[0] == "Branches"
                            for row in seg_rows
                        )
                    )

                    xsection_rows = read_csv(xsection_path)
                    self.assertEqual(len(xsection_rows), 1)
                    self.assertEqual(
                        Path(xsection_rows[0]["inputDir"]),
                        case_path / "Preproc" / "OriData",
                    )
                    self.assertEqual(xsection_rows[0]["resampleDepth"], "0.2")
                    self.assertEqual(xsection_rows[0]["CPRAngle"], "5")
                    self.assertEqual(xsection_rows[0]["cpus"], "3")
                    self.assertTrue(
                        xsection_rows[0]["casePath"].endswith("case_BSRJZ.csv")
                    )

                    fix_rows = read_csv(fix_path)
                    self.assertEqual(len(fix_rows), expected_fix_rows)
                    if expected_fix_rows:
                        self.assertEqual(
                            {Path(row["Mask Path"]).name for row in fix_rows},
                            {
                                artery + "ValNN.nii.gz"
                                for artery in controller.SelectedArteries
                            },
                        )
                        self.assertEqual(
                            {Path(row["Image Path"]).name for row in fix_rows},
                            {
                                artery + "CTALI.nii.gz"
                                for artery in controller.SelectedArteries
                            },
                        )
                        self.assertEqual(
                            {row["Output Name"] for row in fix_rows},
                            {
                                artery + "ValNNFix"
                                for artery in controller.SelectedArteries
                            },
                        )
                        self.assertTrue(
                            all(
                                Path(row["Process Path"]).name
                                == "LCProcess_BSRJZ.csv"
                                for row in fix_rows
                            )
                        )
                        self.assertTrue(
                            all(row["Fill Original Value"] == "Yes" for row in fix_rows)
                        )

                    self.assertTrue(
                        (case_path / "Preproc" / "OriData" / "CTA.nii.gz").is_file()
                    )
                    self.assertFalse((case_path / "CTA" / "CTA.nii.gz").exists())
                    self.assertFalse(
                        (case_path / "Preproc" / "OriData" / "Branches" /
                         "CTA.nii.gz").exists()
                    )

            self.assertEqual(
                Path(controller.LCProcessPath).read_bytes(),
                Path(controller.LumenProcessSourcePath).read_bytes(),
            )
            self.assertEqual(unsuffixed_table.read_text(encoding="utf-8"), "preserve-table")
            self.assertEqual(unsuffixed_case.read_text(encoding="utf-8"), "preserve-case")

    def test_resume_skips_complete_and_reruns_only_missing_or_changed_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = RecordingXsectionRunner()
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                xsection_runner=runner,
            )
            case_path = case_paths[0]
            controller.MaskXsectionFlag = True
            controller.MaskXsectionOnlyFlag = False
            controller.GenerateTable(
                maskXsection=True,
                maskXsectionOnly=False,
            )

            first_outcomes = controller.BatchXsection()
            self.assertEqual(len(runner.calls), 6)
            self.assertTrue(all(value["complete"] for value in first_outcomes.values()))
            first_record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(first_record["complete_xsection_CTA"], "1")
            self.assertEqual(first_record["complete_xsection_seg"], "1")

            runner.calls = []
            second_outcomes = controller.BatchXsection()
            self.assertEqual(runner.calls, [])
            self.assertTrue(all(value["skipped"] for value in second_outcomes.values()))

            missing_output = (
                case_path / "Preproc" / "Xsection" / "LCxValNN.nii.gz"
            )
            missing_output.unlink()
            runner.calls = []
            controller.BatchXsection()
            self.assertEqual(
                [call["output_name"] for call in runner.calls],
                ["LCxValNN.nii.gz"],
            )

            changed_input = (
                case_path
                / "Preproc"
                / "OriData"
                / "Branches"
                / "mLAD_OriB_Val.nii.gz"
            )
            write_nifti(changed_input, nonzero=True, value=99)
            runner.calls = []
            controller.BatchXsection()
            self.assertEqual(
                [call["output_name"] for call in runner.calls],
                ["mLADValNN.nii.gz"],
            )
            final_record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(final_record["complete_xsection_CTA"], "1")
            self.assertEqual(final_record["complete_xsection_seg"], "1")

    def test_executable_failure_and_exit_zero_missing_output_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = RecordingXsectionRunner(
                {
                    "LCxCTALI.nii.gz": "exe_failure_with_output",
                    "mLADCTALI.nii.gz": "missing",
                }
            )
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx", "mLAD"),
                xsection_runner=runner,
            )
            controller.MaskXsectionFlag = False
            controller.MaskXsectionOnlyFlag = False
            controller.GenerateTable(
                maskXsection=False,
                maskXsectionOnly=False,
            )

            outcomes = controller.BatchXsection()
            record = read_csv(controller.XsectionRecordPath)[0]

            self.assertFalse(outcomes[("CASE_001", "LCx", "CTA")]["complete"])
            self.assertEqual(record["LCx_xsection_CTA_exe_failed"], "1")
            self.assertEqual(record["LCx_xsection_CTA_complete"], "0")
            self.assertIn("exit code 7", record["LCx_xsection_CTA_exe_er_msg"])

            self.assertFalse(outcomes[("CASE_001", "mLAD", "CTA")]["complete"])
            self.assertEqual(record["mLAD_xsection_CTA_exe_failed"], "0")
            self.assertEqual(record["mLAD_xsection_CTA_complete"], "0")
            self.assertIn(
                "Missing or empty NIfTI",
                record["mLAD_xsection_CTA_exe_er_msg"],
            )
            self.assertEqual(record["complete_xsection_CTA"], "0")

            failed_artifact = (
                Path(temporary_directory)
                / "0_Failed_Xsection"
                / "batch_0"
                / "CASE_001"
                / "LCxCTALI.nii.gz"
            )
            self.assertTrue(failed_artifact.is_file())
            self.assertTrue(case_paths[0].is_dir())
            self.assertFalse(
                (
                    case_paths[0]
                    / "Preproc"
                    / "Xsection"
                    / "LCxCTALI.nii.gz"
                ).exists()
            )

    def test_all_zero_segmentation_is_moved_without_moving_the_case(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = RecordingXsectionRunner({"LCxValNN.nii.gz": "zero"})
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
                xsection_runner=runner,
            )
            case_path = case_paths[0]
            controller.MaskXsectionFlag = True
            controller.MaskXsectionOnlyFlag = False
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)

            controller.BatchXsection()

            record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(record["LCx_xsection_CTA_complete"], "1")
            self.assertEqual(record["LCx_xsection_seg_exe_failed"], "0")
            self.assertEqual(record["LCx_xsection_seg_complete"], "0")
            self.assertEqual(record["complete_xsection_CTA"], "1")
            self.assertEqual(record["complete_xsection_seg"], "0")
            self.assertIn(
                "no nonzero segmentation voxels",
                record["LCx_xsection_seg_exe_er_msg"],
            )
            failed_artifact = (
                Path(temporary_directory)
                / "0_Failed_Xsection"
                / "batch_0"
                / "CASE_001"
                / "LCxValNN.nii.gz"
            )
            self.assertTrue(failed_artifact.is_file())
            self.assertFalse(
                (case_path / "Preproc" / "Xsection" / "LCxValNN.nii.gz").exists()
            )
            self.assertTrue(
                (
                    case_path
                    / "Preproc"
                    / "OriData"
                    / "Branches"
                    / "LCx_OriB_Val.nii.gz"
                ).is_file()
            )
            self.assertTrue(case_path.is_dir())

    def test_lumen_resume_is_independent_and_aggregates_all_three_arteries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            xsection_runner = RecordingXsectionRunner()
            lumen_runner = RecordingLumenRunner()
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                xsection_runner=xsection_runner,
                lumen_runner=lumen_runner,
            )
            case_path = case_paths[0]
            controller.MaskXsectionFlag = True
            controller.MaskXsectionOnlyFlag = False
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            controller.BatchXsection()
            self.assertEqual(len(xsection_runner.calls), 6)

            controller.BatchLumenCorrection()
            self.assertEqual(
                {call["output_name"] for call in lumen_runner.calls},
                {
                    "LCxValNNFixVal.nii.gz",
                    "mLADValNNFixVal.nii.gz",
                    "mRCAValNNFixVal.nii.gz",
                },
            )
            record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(record["complete_xsection_CTA"], "1")
            self.assertEqual(record["complete_xsection_seg"], "1")
            self.assertEqual(record["complete_lumen_correction"], "1")

            lumen_runner.calls = []
            xsection_runner.calls = []
            skipped_outcomes = controller.BatchLumenCorrection()
            self.assertEqual(lumen_runner.calls, [])
            self.assertTrue(all(value["skipped"] for value in skipped_outcomes.values()))

            missing_corrected = (
                case_path
                / "Preproc"
                / "XsectionFix"
                / "mLADValNNFixVal.nii.gz"
            )
            missing_corrected.unlink()
            lumen_runner.calls = []
            retry_outcomes = controller.BatchLumenCorrection()
            self.assertEqual(
                [call["output_name"] for call in lumen_runner.calls],
                ["mLADValNNFixVal.nii.gz"],
            )
            self.assertEqual(xsection_runner.calls, [])
            self.assertFalse(retry_outcomes[("CASE_001", "mLAD")]["skipped"])
            self.assertTrue(
                case_path.joinpath(
                    "Preproc", "Xsection", "mLADValNN.nii.gz"
                ).is_file()
            )
            final_record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(final_record["complete_lumen_correction"], "1")

    def test_all_zero_corrected_mask_moves_alone_and_preserves_raw_segmentation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            xsection_runner = RecordingXsectionRunner()
            lumen_runner = RecordingLumenRunner(
                {"LCxValNNFixVal.nii.gz": "zero"}
            )
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
                xsection_runner=xsection_runner,
                lumen_runner=lumen_runner,
            )
            case_path = case_paths[0]
            controller.MaskXsectionFlag = True
            controller.MaskXsectionOnlyFlag = False
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            controller.BatchXsection()

            controller.BatchLumenCorrection()

            raw_segmentation = (
                case_path / "Preproc" / "Xsection" / "LCxValNN.nii.gz"
            )
            corrected_segmentation = (
                case_path
                / "Preproc"
                / "XsectionFix"
                / "LCxValNNFixVal.nii.gz"
            )
            failed_corrected = (
                Path(temporary_directory)
                / "0_Failed_Xsection"
                / "batch_0"
                / "CASE_001"
                / "LCxValNNFixVal.nii.gz"
            )
            self.assertTrue(raw_segmentation.is_file())
            self.assertFalse(corrected_segmentation.exists())
            self.assertTrue(failed_corrected.is_file())
            self.assertTrue(case_path.is_dir())

            record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(record["LCx_xsection_seg_complete"], "1")
            self.assertEqual(record["LCx_lumen_complete"], "0")
            self.assertEqual(record["complete_lumen_correction"], "0")
            self.assertIn(
                "no nonzero segmentation voxels",
                record["LCx_lumen_er_msg"],
            )

    def test_full_filter_and_xsection_pipeline_second_pass_makes_no_runner_calls(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            filter_runner = RecordingFilterMaskRunner()
            xsection_runner = RecordingXsectionRunner()
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                xsection_runner=xsection_runner,
                filter_mask_runner=filter_runner,
            )
            case_path = case_paths[0]
            branch_path = case_path / "Preproc" / "OriData" / "Branches"
            for artery in controller.SelectedArteries:
                (branch_path / (artery + "_OriB_Val.nii.gz")).unlink()

            controller.MaskXsectionFlag = True
            controller.MaskXsectionOnlyFlag = False
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            first_filter_outcomes = controller.BatchFilterMask()
            first_xsection_outcomes = controller.BatchXsection()

            self.assertEqual(len(filter_runner.calls), 3)
            self.assertTrue(
                all(call["row_count"] == 1 for call in filter_runner.calls)
            )
            self.assertEqual(len(xsection_runner.calls), 6)
            self.assertTrue(
                all(value["complete"] for value in first_filter_outcomes.values())
            )
            self.assertTrue(
                all(value["complete"] for value in first_xsection_outcomes.values())
            )
            for artery in controller.SelectedArteries:
                self.assertTrue(
                    (branch_path / (artery + "_OriB_Val.nii.gz")).is_file()
                )
                self.assertTrue(
                    (branch_path / (artery + "_OriB_NVl.nii.gz")).is_file()
                )

            filter_runner.calls = []
            xsection_runner.calls = []
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            second_filter_outcomes = controller.BatchFilterMask()
            second_xsection_outcomes = controller.BatchXsection()

            self.assertEqual(filter_runner.calls, [])
            self.assertEqual(xsection_runner.calls, [])
            self.assertTrue(
                all(value["skipped"] for value in second_filter_outcomes.values())
            )
            self.assertTrue(
                all(value["skipped"] for value in second_xsection_outcomes.values())
            )
            record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(record["filter_mask_complete"], "1")
            self.assertTrue(record["filter_mask_fingerprint"])
            self.assertEqual(record["complete_xsection_CTA"], "1")
            self.assertEqual(record["complete_xsection_seg"], "1")

    def test_filter_mask_reports_exact_all_zero_nifti_reason(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner_calls = []

            def all_zero_filter_runner(table_path):
                rows = read_csv(table_path)
                runner_calls.append(len(rows))
                for row in rows:
                    output_folder = Path(row["OutputFolder"])
                    file_name_reference = row["FileNameRef"]
                    write_nifti(
                        output_folder / (file_name_reference + "Val.nii.gz"),
                        nonzero=False,
                    )
                    write_nifti(
                        output_folder / (file_name_reference + "NVl.nii.gz"),
                        nonzero=False,
                    )

            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
                filter_mask_runner=all_zero_filter_runner,
            )
            case_path = case_paths[0]
            value_output = (
                case_path
                / "Preproc"
                / "OriData"
                / "Branches"
                / "LCx_OriB_Val.nii.gz"
            )
            value_output.unlink()
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)

            outcomes = controller.BatchFilterMask()

            error_message = outcomes[str(case_path)]["error_message"]
            self.assertEqual(runner_calls, [1])
            self.assertFalse(outcomes[str(case_path)]["complete"])
            self.assertIn(
                "NIfTI contains no nonzero segmentation voxels",
                error_message,
            )
            self.assertIn("LCx_OriB_Val.nii.gz", error_message)
            record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(record["filter_mask_complete"], "0")
            self.assertEqual(record["filter_mask_er_msg"], error_message)
            self.assertEqual(record["LCx_filter_mask_complete"], "0")
            self.assertIn(
                "NIfTI contains no nonzero segmentation voxels",
                record["LCx_filter_mask_er_msg"],
            )

    def test_filter_mask_retries_transient_post_writer_nifti_unreadability(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            writer_state = {"returned": False}
            real_runner = RecordingFilterMaskRunner()

            def filter_runner(table_path):
                real_runner(table_path)
                writer_state["returned"] = True

            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
                filter_mask_runner=filter_runner,
            )
            case_path = case_paths[0]
            target_path = (
                case_path
                / "Preproc"
                / "OriData"
                / "Branches"
                / "LCx_OriB_Val.nii.gz"
            )
            target_path.unlink()
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            real_validate_nifti = BatchSliceRadiomicsMain.validate_nifti
            transient_attempts = []

            def transient_validate(file_path, require_nonzero=False):
                path = Path(file_path)
                if writer_state["returned"] and path == target_path:
                    transient_attempts.append(str(path))
                    if len(transient_attempts) <= 2:
                        return False, "Unreadable NIfTI during transient network visibility"
                return real_validate_nifti(
                    file_path,
                    require_nonzero=require_nonzero,
                )

            with mock.patch.object(
                    BatchSliceRadiomicsMain,
                    "validate_nifti",
                    side_effect=transient_validate), mock.patch.object(
                        BatchSliceRadiomicsMain.time,
                        "sleep"):
                outcomes = controller.BatchFilterMask()

            self.assertEqual(len(real_runner.calls), 1)
            self.assertGreaterEqual(len(transient_attempts), 3)
            self.assertTrue(outcomes[str(case_path)]["complete"])
            self.assertEqual(outcomes[str(case_path)]["error_message"], "")
            record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(record["filter_mask_complete"], "1")
            self.assertEqual(record["filter_mask_er_msg"], "")
            self.assertEqual(record["LCx_filter_mask_complete"], "1")
            self.assertEqual(record["LCx_filter_mask_er_msg"], "")

    def test_one_empty_filter_artery_does_not_block_valid_arteries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            filter_calls = []

            def selective_filter_runner(table_path):
                rows = read_csv(table_path)
                filter_calls.append(len(rows))
                for row in rows:
                    output_folder = Path(row["OutputFolder"])
                    file_name_reference = row["FileNameRef"]
                    nonzero = not file_name_reference.startswith("mLAD")
                    write_nifti(
                        output_folder / (file_name_reference + "Val.nii.gz"),
                        nonzero=nonzero,
                    )
                    write_nifti(
                        output_folder / (file_name_reference + "NVl.nii.gz"),
                        nonzero=nonzero,
                    )

            xsection_runner = RecordingXsectionRunner()
            lumen_runner = RecordingLumenRunner()
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                xsection_runner=xsection_runner,
                lumen_runner=lumen_runner,
                filter_mask_runner=selective_filter_runner,
            )
            case_path = case_paths[0]
            branch_path = case_path / "Preproc" / "OriData" / "Branches"
            for artery in controller.SelectedArteries:
                (branch_path / (artery + "_OriB_Val.nii.gz")).unlink()
            controller.MaskXsectionFlag = True
            controller.MaskXsectionOnlyFlag = False
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)

            filter_outcomes = controller.BatchFilterMask()
            xsection_outcomes = controller.BatchXsection()
            lumen_outcomes = controller.BatchLumenCorrection()

            self.assertEqual(filter_calls, [1, 1, 1])
            self.assertFalse(filter_outcomes[str(case_path)]["complete"])
            self.assertFalse(
                filter_outcomes[str(case_path)]["arteries"]["mLAD"]["complete"]
            )
            self.assertTrue(
                filter_outcomes[str(case_path)]["arteries"]["LCx"]["complete"]
            )
            self.assertTrue(
                filter_outcomes[str(case_path)]["arteries"]["mRCA"]["complete"]
            )
            self.assertEqual(
                {call["output_name"] for call in xsection_runner.calls},
                {
                    "LCxCTALI.nii.gz",
                    "LCxValNN.nii.gz",
                    "mLADCTALI.nii.gz",
                    "mRCACTALI.nii.gz",
                    "mRCAValNN.nii.gz",
                },
            )
            self.assertFalse(
                xsection_outcomes[("CASE_001", "mLAD", "seg")]["complete"]
            )
            self.assertEqual(
                {call["output_name"] for call in lumen_runner.calls},
                {"LCxValNNFixVal.nii.gz", "mRCAValNNFixVal.nii.gz"},
            )
            self.assertFalse(lumen_outcomes[("CASE_001", "mLAD")]["complete"])
            self.assertTrue(lumen_outcomes[("CASE_001", "LCx")]["complete"])
            self.assertTrue(lumen_outcomes[("CASE_001", "mRCA")]["complete"])
            self.assertTrue(case_path.is_dir())

            record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(record["filter_mask_complete"], "0")
            self.assertEqual(record["mLAD_filter_mask_complete"], "0")
            self.assertEqual(record["LCx_filter_mask_complete"], "1")
            self.assertEqual(record["mRCA_filter_mask_complete"], "1")
            self.assertEqual(record["mLAD_xsection_seg_complete"], "0")
            self.assertEqual(record["LCx_xsection_seg_complete"], "1")
            self.assertEqual(record["mRCA_xsection_seg_complete"], "1")
            self.assertEqual(record["mLAD_lumen_complete"], "0")
            self.assertEqual(record["LCx_lumen_complete"], "1")
            self.assertEqual(record["mRCA_lumen_complete"], "1")

    def test_filter_mask_resume_reruns_only_the_missing_artery(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            filter_runner = RecordingFilterMaskRunner()
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                filter_mask_runner=filter_runner,
            )
            case_path = case_paths[0]
            branch_path = case_path / "Preproc" / "OriData" / "Branches"
            for artery in controller.SelectedArteries:
                (branch_path / (artery + "_OriB_Val.nii.gz")).unlink()
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)

            first_outcomes = controller.BatchFilterMask()

            self.assertTrue(first_outcomes[str(case_path)]["complete"])
            self.assertEqual(len(filter_runner.calls), 3)
            preserved_paths = [
                branch_path / "LCx_OriB_Val.nii.gz",
                branch_path / "LCx_OriB_NVl.nii.gz",
                branch_path / "mRCA_OriB_Val.nii.gz",
                branch_path / "mRCA_OriB_NVl.nii.gz",
            ]
            preserved_bytes = {
                path: path.read_bytes() for path in preserved_paths
            }
            (branch_path / "mLAD_OriB_Val.nii.gz").unlink()
            filter_runner.calls = []
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)

            resumed_outcomes = controller.BatchFilterMask()

            self.assertEqual(len(filter_runner.calls), 1)
            self.assertEqual(
                filter_runner.calls[0]["file_names"], ["mLAD_OriB_"]
            )
            for path, expected_bytes in preserved_bytes.items():
                self.assertEqual(path.read_bytes(), expected_bytes)
            case_outcome = resumed_outcomes[str(case_path)]
            self.assertTrue(case_outcome["complete"])
            self.assertTrue(case_outcome["arteries"]["LCx"]["skipped"])
            self.assertFalse(case_outcome["arteries"]["mLAD"]["skipped"])
            self.assertTrue(case_outcome["arteries"]["mRCA"]["skipped"])

    def test_filter_mask_noop_runner_cannot_adopt_stale_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            initial_runner = RecordingFilterMaskRunner()
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
                filter_mask_runner=initial_runner,
            )
            case_path = case_paths[0]
            branch_path = case_path / "Preproc" / "OriData" / "Branches"
            (branch_path / "LCx_OriB_Val.nii.gz").unlink()
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            self.assertTrue(
                controller.BatchFilterMask()[str(case_path)]["complete"]
            )

            source_segmentation = (
                case_path
                / "Preproc"
                / "OriData"
                / "sc_labelled_coronary_segment.nii.gz"
            )
            write_nifti(source_segmentation, nonzero=True, value=7)
            noop_calls = []

            def noop_runner(table_path):
                noop_calls.append(read_csv(table_path)[0]["FileNameRef"])

            controller._filter_mask_runner = noop_runner
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)

            outcomes = controller.BatchFilterMask()

            self.assertEqual(noop_calls, ["LCx_OriB_"])
            case_outcome = outcomes[str(case_path)]
            self.assertFalse(case_outcome["complete"])
            self.assertIn("was not updated by this run", case_outcome["error_message"])
            record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(record["filter_mask_complete"], "0")
            self.assertEqual(record["LCx_filter_mask_complete"], "0")

    def test_filter_mask_checkpoint_survives_interruption_between_arteries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            calls = []

            def interrupting_runner(table_path):
                row = read_csv(table_path)[0]
                file_name_reference = row["FileNameRef"]
                calls.append(file_name_reference)
                if file_name_reference == "mLAD_OriB_":
                    raise KeyboardInterrupt("injected hard interruption")
                output_folder = Path(row["OutputFolder"])
                write_nifti(
                    output_folder / (file_name_reference + "Val.nii.gz"),
                    nonzero=True,
                )
                write_nifti(
                    output_folder / (file_name_reference + "NVl.nii.gz"),
                    nonzero=True,
                )

            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                filter_mask_runner=interrupting_runner,
            )
            case_path = case_paths[0]
            branch_path = case_path / "Preproc" / "OriData" / "Branches"
            for artery in controller.SelectedArteries:
                (branch_path / (artery + "_OriB_Val.nii.gz")).unlink()
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)

            with self.assertRaises(KeyboardInterrupt):
                controller.BatchFilterMask()

            self.assertEqual(calls, ["LCx_OriB_", "mLAD_OriB_"])
            interrupted_record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(
                interrupted_record["LCx_filter_mask_complete"], "1"
            )
            self.assertEqual(
                interrupted_record["mLAD_filter_mask_complete"], "0"
            )
            self.assertEqual(interrupted_record["filter_mask_complete"], "0")

            resume_runner = RecordingFilterMaskRunner()
            controller._filter_mask_runner = resume_runner
            resumed_outcomes = controller.BatchFilterMask()

            self.assertEqual(
                [call["file_names"] for call in resume_runner.calls],
                [["mLAD_OriB_"], ["mRCA_OriB_"]],
            )
            case_outcome = resumed_outcomes[str(case_path)]
            self.assertTrue(case_outcome["complete"])
            self.assertTrue(case_outcome["arteries"]["LCx"]["skipped"])
            completed_record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(completed_record["filter_mask_complete"], "1")

    def test_generate_table_requires_an_artery_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, _, _, _ = make_controller(
                temporary_directory,
                selected_arteries=(),
            )

            with self.assertRaisesRegex(ValueError, "at least one artery"):
                controller.GenerateTable(
                    maskXsection=True,
                    maskXsectionOnly=True,
                )

    def test_lumen_snapshot_error_blocks_stale_process_and_records_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            xsection_runner = RecordingXsectionRunner()
            lumen_runner = RecordingLumenRunner()
            controller, _, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
                xsection_runner=xsection_runner,
                lumen_runner=lumen_runner,
            )
            controller.MaskXsectionFlag = True
            controller.MaskXsectionOnlyFlag = False
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            controller.BatchXsection()
            stale_snapshot = Path(controller.LCProcessPath)
            stale_bytes = stale_snapshot.read_bytes()

            outcomes = controller.BatchLumenCorrection(
                process_snapshot_error=(
                    "Could not snapshot current lumen process table; stale copy blocked"
                )
            )

            self.assertEqual(lumen_runner.calls, [])
            self.assertEqual(stale_snapshot.read_bytes(), stale_bytes)
            self.assertFalse(outcomes[("CASE_001", "LCx")]["complete"])
            self.assertFalse(outcomes[("CASE_001", "LCx")]["skipped"])
            record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(record["LCx_lumen_complete"], "0")
            self.assertEqual(record["complete_lumen_correction"], "0")
            self.assertIn("stale copy blocked", record["LCx_lumen_er_msg"])

    def test_current_branch_validation_failure_never_reuses_stale_suffixed_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
            )
            case_path = case_paths[0]
            csv_path = case_path / "Preproc" / "CSV"
            stale_path = csv_path / "coronarySegmentsUse_BSRJZ.json"
            stale_bytes = stale_path.read_bytes()

            with mock.patch.object(
                    BatchSliceRadiomicsMain.BranchDictionary,
                    "ValidateUse",
                    side_effect=RuntimeError("injected current validation failure")):
                outcomes = controller.CheckWriteDict()

            self.assertEqual(outcomes, {str(csv_path): False})
            self.assertTrue(controller._branch_validation_attempted)
            self.assertEqual(controller._validated_use_paths, {})
            self.assertEqual(stale_path.read_bytes(), stale_bytes)

            successful_cases = controller.GenerateTable(
                maskXsection=True,
                maskXsectionOnly=False,
            )

            self.assertEqual(successful_cases, [])
            self.assertEqual(read_csv(controller.FilterMaskTablePath), [])
            self.assertEqual(read_csv(controller.XsectionTablePath), [])
            self.assertEqual(read_csv(controller.XsectionFixTablePath), [])
            self.assertFalse((csv_path / "case_BSRJZ.csv").exists())

    def test_invalidated_source_blocks_filter_xsection_and_lumen_resume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            filter_runner = RecordingFilterMaskRunner()
            xsection_runner = RecordingXsectionRunner()
            lumen_runner = RecordingLumenRunner()
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
                xsection_runner=xsection_runner,
                lumen_runner=lumen_runner,
                filter_mask_runner=filter_runner,
            )
            case_path = case_paths[0]
            controller.MaskXsectionFlag = True
            controller.MaskXsectionOnlyFlag = False
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)

            controller.BatchFilterMask()
            controller.BatchXsection()
            controller.BatchLumenCorrection()
            first_record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(len(filter_runner.calls), 1)
            self.assertEqual(len(xsection_runner.calls), 2)
            self.assertEqual(len(lumen_runner.calls), 1)
            self.assertEqual(first_record["complete_segmentation"], "1")
            self.assertEqual(first_record["filter_mask_complete"], "1")
            self.assertEqual(first_record["LCx_xsection_CTA_complete"], "1")
            self.assertEqual(first_record["LCx_xsection_seg_complete"], "1")
            self.assertEqual(first_record["LCx_lumen_complete"], "1")

            source_segmentation = (
                case_path
                / "Preproc"
                / "OriData"
                / "sc_labelled_coronary_segment.nii.gz"
            )
            write_nifti(source_segmentation, nonzero=False)
            filter_runner.calls = []
            xsection_runner.calls = []
            lumen_runner.calls = []

            filter_outcomes = controller.BatchFilterMask()
            xsection_outcomes = controller.BatchXsection()
            lumen_outcomes = controller.BatchLumenCorrection()

            self.assertEqual(filter_runner.calls, [])
            self.assertEqual(xsection_runner.calls, [])
            self.assertEqual(lumen_runner.calls, [])
            self.assertFalse(filter_outcomes[str(case_path)]["complete"])
            self.assertFalse(
                xsection_outcomes[("CASE_001", "LCx", "seg")]["complete"]
            )
            self.assertTrue(
                xsection_outcomes[("CASE_001", "LCx", "CTA")]["skipped"]
            )
            self.assertFalse(lumen_outcomes[("CASE_001", "LCx")]["complete"])

            record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(record["complete_segmentation"], "0")
            self.assertEqual(record["filter_mask_complete"], "0")
            self.assertEqual(record["LCx_xsection_CTA_complete"], "1")
            self.assertEqual(record["complete_xsection_CTA"], "1")
            self.assertEqual(record["LCx_xsection_seg_complete"], "0")
            self.assertEqual(record["complete_xsection_seg"], "0")
            self.assertEqual(record["LCx_lumen_complete"], "0")
            self.assertEqual(record["complete_lumen_correction"], "0")
            self.assertIn(
                "no nonzero segmentation voxels",
                record["filter_mask_er_msg"],
            )
            self.assertIn(
                "no nonzero segmentation voxels",
                record["LCx_xsection_seg_exe_er_msg"],
            )
            self.assertIn(
                "no nonzero segmentation voxels",
                record["LCx_lumen_er_msg"],
            )

    def test_legacy_filter_outputs_are_not_adopted_when_source_is_all_zero(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            filter_runner = RecordingFilterMaskRunner()
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
                filter_mask_runner=filter_runner,
            )
            case_path = case_paths[0]
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            source_segmentation = (
                case_path
                / "Preproc"
                / "OriData"
                / "sc_labelled_coronary_segment.nii.gz"
            )
            write_nifti(source_segmentation, nonzero=False)

            branch_path = case_path / "Preproc" / "OriData" / "Branches"
            value_output = branch_path / "LCx_OriB_Val.nii.gz"
            non_value_output = branch_path / "LCx_OriB_NVl.nii.gz"
            write_nifti(value_output, nonzero=True)
            write_nifti(non_value_output, nonzero=True)
            self.assertFalse(Path(controller.XsectionRecordPath).exists())

            outcomes = controller.BatchFilterMask()

            self.assertEqual(filter_runner.calls, [])
            self.assertFalse(outcomes[str(case_path)]["complete"])
            self.assertFalse(outcomes[str(case_path)]["skipped"])
            self.assertTrue(value_output.is_file())
            self.assertTrue(non_value_output.is_file())
            record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(record["filter_mask_complete"], "0")
            self.assertIn(
                "no nonzero segmentation voxels",
                record["filter_mask_er_msg"],
            )

    def test_completed_xsection_skips_after_configured_executable_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            xsection_runner = RecordingXsectionRunner()
            controller, _, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
                xsection_runner=xsection_runner,
            )
            controller.MaskXsectionFlag = True
            controller.MaskXsectionOnlyFlag = False
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            first_outcomes = controller.BatchXsection()
            self.assertEqual(len(xsection_runner.calls), 2)
            self.assertTrue(all(value["complete"] for value in first_outcomes.values()))

            executable_path = Path(controller.CPREXEPath)
            removed_path = executable_path.with_name(
                executable_path.name + ".removed"
            )
            executable_path.rename(removed_path)
            self.assertFalse(Path(controller.CPREXEPath).exists())
            xsection_runner.calls = []

            resumed_outcomes = controller.BatchXsection()

            self.assertEqual(xsection_runner.calls, [])
            self.assertEqual(len(resumed_outcomes), 2)
            self.assertTrue(
                all(value["complete"] for value in resumed_outcomes.values())
            )
            self.assertTrue(
                all(value["skipped"] for value in resumed_outcomes.values())
            )
            record = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(record["LCx_xsection_CTA_complete"], "1")
            self.assertEqual(record["LCx_xsection_seg_complete"], "1")
            self.assertEqual(record["complete_xsection_CTA"], "1")
            self.assertEqual(record["complete_xsection_seg"], "1")

    def test_filter_failure_skips_by_default_and_checkbox_retries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
            )
            case_path = case_paths[0]
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            failing_runner = mock.Mock(
                side_effect=RuntimeError("injected Filter Mask failure")
            )
            controller._filter_mask_runner = failing_runner

            first = controller.BatchFilterMask()[str(case_path)]
            self.assertFalse(first["complete"])
            self.assertEqual(failing_runner.call_count, 1)

            retry_runner = RecordingFilterMaskRunner()
            controller._filter_mask_runner = retry_runner
            skipped = controller.BatchFilterMask()[str(case_path)]
            self.assertEqual(retry_runner.calls, [])
            self.assertTrue(skipped["skipped"])
            self.assertTrue(skipped["failed_skipped"])

            controller.RerunFailedFlag = True
            retried = controller.BatchFilterMask()[str(case_path)]
            self.assertEqual(len(retry_runner.calls), 1)
            self.assertTrue(retried["complete"])

    def test_legacy_filter_failure_without_fingerprint_stays_skipped(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = RecordingFilterMaskRunner()
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
                filter_mask_runner=runner,
            )
            case_path = case_paths[0]
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            record_map = controller._load_xsection_record_map()
            record = controller._ensure_case_record(record_map, case_path)
            record["filter_mask_complete"] = 0
            record["filter_mask_er_msg"] = "legacy case-level failure"
            record["filter_mask_fingerprint"] = ""
            controller._save_xsection_record_map(record_map)

            outcome = controller.BatchFilterMask()[str(case_path)]

            self.assertEqual(runner.calls, [])
            self.assertFalse(outcome["complete"])
            self.assertTrue(outcome["skipped"])
            self.assertTrue(outcome["failed_skipped"])
            saved = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(
                saved["filter_mask_er_msg"], "legacy case-level failure"
            )
            self.assertEqual(saved["filter_mask_fingerprint"], "")

    def test_xsection_failure_policy_and_changed_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            failing_runner = RecordingXsectionRunner(
                {"LCxCTALI.nii.gz": "missing"}
            )
            controller, case_paths, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
                xsection_runner=failing_runner,
            )
            case_path = case_paths[0]
            controller.MaskXsectionFlag = False
            controller.MaskXsectionOnlyFlag = False
            controller.GenerateTable(maskXsection=False, maskXsectionOnly=False)

            first = controller.BatchXsection()[("CASE_001", "LCx", "CTA")]
            self.assertFalse(first["complete"])
            self.assertEqual(len(failing_runner.calls), 1)

            retry_runner = RecordingXsectionRunner()
            controller._xsection_runner = retry_runner
            skipped = controller.BatchXsection()[
                ("CASE_001", "LCx", "CTA")
            ]
            self.assertEqual(retry_runner.calls, [])
            self.assertTrue(skipped["failed_skipped"])

            checked_runner = RecordingXsectionRunner(
                {"LCxCTALI.nii.gz": "missing"}
            )
            controller._xsection_runner = checked_runner
            controller.RerunFailedFlag = True
            controller.BatchXsection()
            self.assertEqual(len(checked_runner.calls), 1)

            controller.RerunFailedFlag = False
            centerline_path = (
                case_path / "Preproc" / "CSV" / "LCxExt.csv"
            )
            centerline_path.write_text(
                "0,0,0\n1,1,1\n2,2,2\n",
                encoding="utf-8",
            )
            changed_runner = RecordingXsectionRunner()
            controller._xsection_runner = changed_runner
            changed = controller.BatchXsection()[
                ("CASE_001", "LCx", "CTA")
            ]
            self.assertEqual(len(changed_runner.calls), 1)
            self.assertTrue(changed["complete"])

    def test_lumen_failure_skips_by_default_and_checkbox_retries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            xsection_runner = RecordingXsectionRunner()
            failing_lumen = RecordingLumenRunner(
                {"LCxValNNFixVal.nii.gz": "failure"}
            )
            controller, _, _, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
                xsection_runner=xsection_runner,
                lumen_runner=failing_lumen,
            )
            controller.MaskXsectionFlag = True
            controller.MaskXsectionOnlyFlag = False
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            controller.BatchXsection()

            first = controller.BatchLumenCorrection()[
                ("CASE_001", "LCx")
            ]
            self.assertFalse(first["complete"])
            self.assertEqual(len(failing_lumen.calls), 1)

            retry_runner = RecordingLumenRunner()
            controller._lumen_runner = retry_runner
            skipped = controller.BatchLumenCorrection()[
                ("CASE_001", "LCx")
            ]
            self.assertEqual(retry_runner.calls, [])
            self.assertTrue(skipped["failed_skipped"])

            controller.RerunFailedFlag = True
            retried = controller.BatchLumenCorrection()[
                ("CASE_001", "LCx")
            ]
            self.assertEqual(len(retry_runner.calls), 1)
            self.assertTrue(retried["complete"])


class PerCaseArteryApplicabilityTests(unittest.TestCase):
    def test_all_selected_arteries_use_false_are_not_applicable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            filter_runner = RecordingFilterMaskRunner()
            controller, case_paths, xsection_runner, lumen_runner = make_controller(
                temporary_directory,
                filter_mask_runner=filter_runner,
            )
            case_path = case_paths[0]
            set_case_artery_use(case_path, ())
            events = []
            controller._progress_callback = events.append
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)

            self.assertEqual(
                read_csv(case_path / "Preproc" / "CSV" / "case_BSRJZ.csv"),
                [],
            )
            self.assertEqual(read_csv(controller.FilterMaskTablePath), [])
            self.assertEqual(read_csv(controller.XsectionFixTablePath), [])
            self.assertEqual(len(read_csv(controller.XsectionTablePath)), 1)

            filter_outcome = controller.BatchFilterMask()[str(case_path)]
            xsection_outcomes = controller.BatchXsection()
            lumen_outcomes = controller.BatchLumenCorrection()

            self.assertTrue(filter_outcome["complete"])
            self.assertTrue(filter_outcome["skipped"])
            self.assertEqual(filter_runner.calls, [])
            self.assertEqual(xsection_runner.calls, [])
            self.assertEqual(lumen_runner.calls, [])
            self.assertTrue(all(
                outcome.get("not_applicable")
                for outcome in xsection_outcomes.values()
            ))
            self.assertTrue(all(
                outcome.get("not_applicable")
                for outcome in lumen_outcomes.values()
            ))

            saved = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(saved["filter_mask_complete"], "1")
            self.assertEqual(saved["complete_xsection_CTA"], "1")
            self.assertEqual(saved["complete_xsection_seg"], "1")
            self.assertEqual(saved["complete_lumen_correction"], "1")
            for artery in ("LCx", "mLAD", "mRCA"):
                self.assertEqual(saved[artery + "_filter_mask_complete"], "0")
                self.assertEqual(saved[artery + "_xsection_CTA_complete"], "0")
                self.assertEqual(saved[artery + "_xsection_seg_complete"], "0")
                self.assertEqual(saved[artery + "_lumen_complete"], "0")
                self.assertEqual(saved[artery + "_xsection_CTA_exe_er_msg"], "")
                self.assertEqual(saved[artery + "_xsection_seg_exe_er_msg"], "")
                self.assertEqual(saved[artery + "_lumen_er_msg"], "")

            stage_events = [event for event in events if event.kind == "case_stage"]
            self.assertEqual(
                [(event.stage, event.state) for event in stage_events],
                [
                    ("Filter Mask", "skipped"),
                    ("Xsection", "skipped"),
                    ("Lumen Correction", "skipped"),
                ],
            )

    def test_mixed_applicability_runs_only_use_true_arteries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            filter_runner = RecordingFilterMaskRunner()
            controller, case_paths, xsection_runner, lumen_runner = make_controller(
                temporary_directory,
                filter_mask_runner=filter_runner,
            )
            case_path = case_paths[0]
            set_case_artery_use(case_path, ("LCx",))
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)

            controller.BatchFilterMask()
            controller.BatchXsection()
            controller.BatchLumenCorrection()

            self.assertEqual(len(filter_runner.calls), 1)
            self.assertEqual(
                filter_runner.calls[0]["file_names"], ["LCx_OriB_"]
            )
            self.assertEqual(
                {call["output_name"] for call in xsection_runner.calls},
                {"LCxCTALI.nii.gz", "LCxValNN.nii.gz"},
            )
            self.assertEqual(len(lumen_runner.calls), 1)
            self.assertEqual(
                lumen_runner.calls[0]["output_name"],
                "LCxValNNFixVal.nii.gz",
            )

            saved = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(saved["complete_xsection_CTA"], "1")
            self.assertEqual(saved["complete_xsection_seg"], "1")
            self.assertEqual(saved["complete_lumen_correction"], "1")
            for artery in ("mLAD", "mRCA"):
                self.assertEqual(saved[artery + "_xsection_CTA_complete"], "0")
                self.assertEqual(saved[artery + "_xsection_seg_complete"], "0")
                self.assertEqual(saved[artery + "_lumen_complete"], "0")
                self.assertEqual(saved[artery + "_xsection_CTA_exe_er_msg"], "")
                self.assertEqual(saved[artery + "_xsection_seg_exe_er_msg"], "")
                self.assertEqual(saved[artery + "_lumen_er_msg"], "")

    def test_missing_row_for_use_true_remains_a_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths, xsection_runner, _ = make_controller(
                temporary_directory,
            )
            case_path = case_paths[0]
            set_case_artery_use(case_path, ("LCx",))
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            case_table_path = case_path / "Preproc" / "CSV" / "case_BSRJZ.csv"
            case_rows = [
                row for row in read_csv(case_table_path)
                if row["outputImagePath"] != "LCxValNN.nii.gz"
            ]
            controller._write_table(
                case_table_path, controller.CASE_HEADERS, case_rows
            )

            outcomes = controller.BatchXsection()

            missing = outcomes[("CASE_001", "LCx", "seg")]
            self.assertFalse(missing["complete"])
            self.assertFalse(missing["skipped"])
            self.assertIn("No seg Xsection table row for LCx", missing["error_message"])
            self.assertEqual(
                [call["output_name"] for call in xsection_runner.calls],
                ["LCxCTALI.nii.gz"],
            )
            saved = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(saved["LCx_xsection_seg_complete"], "0")
            self.assertEqual(saved["complete_xsection_seg"], "0")
            self.assertEqual(saved["mLAD_xsection_seg_exe_er_msg"], "")
            self.assertEqual(saved["mRCA_xsection_seg_exe_er_msg"], "")


class ApplicabilityResumeAndSafetyTests(unittest.TestCase):
    def test_stale_false_failures_are_repaired_with_rerun_failed_off(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            filter_runner = RecordingFilterMaskRunner()
            controller, case_paths, xsection_runner, lumen_runner = make_controller(
                temporary_directory,
                filter_mask_runner=filter_runner,
            )
            case_path = case_paths[0]
            set_case_artery_use(case_path, ())
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            record_map = controller._load_xsection_record_map()
            record = controller._ensure_case_record(record_map, case_path)
            for artery in ("LCx", "mLAD", "mRCA"):
                record[artery + "_filter_mask_complete"] = 0
                record[artery + "_filter_mask_er_msg"] = (
                    "No Filter Mask row for " + artery
                )
                record[artery + "_filter_mask_fingerprint"] = "old-filter"
                for artifact_type in ("CTA", "seg"):
                    prefix = artery + "_xsection_" + artifact_type
                    record[prefix + "_exe_failed"] = 0
                    record[prefix + "_complete"] = 0
                    record[prefix + "_exe_er_msg"] = (
                        "No " + artifact_type + " Xsection table row for " + artery
                    )
                    record[prefix + "_fingerprint"] = "old-xsection"
                record[artery + "_lumen_complete"] = 0
                record[artery + "_lumen_er_msg"] = (
                    "No Lumen Correction table row for " + artery
                )
                record[artery + "_lumen_fingerprint"] = "old-lumen"
            controller._save_xsection_record_map(record_map)
            controller.RerunFailedFlag = False

            filter_outcome = controller.BatchFilterMask()[str(case_path)]
            xsection_outcomes = controller.BatchXsection()
            lumen_outcomes = controller.BatchLumenCorrection()

            self.assertTrue(filter_outcome["complete"])
            self.assertFalse(filter_outcome["failed_skipped"])
            self.assertEqual(filter_runner.calls, [])
            self.assertEqual(xsection_runner.calls, [])
            self.assertEqual(lumen_runner.calls, [])
            self.assertFalse(any(
                outcome.get("failed_skipped")
                for outcome in xsection_outcomes.values()
            ))
            self.assertFalse(any(
                outcome.get("failed_skipped")
                for outcome in lumen_outcomes.values()
            ))
            saved = read_csv(controller.XsectionRecordPath)[0]
            for artery in ("LCx", "mLAD", "mRCA"):
                self.assertEqual(saved[artery + "_filter_mask_er_msg"], "")
                self.assertEqual(saved[artery + "_xsection_CTA_exe_er_msg"], "")
                self.assertEqual(saved[artery + "_xsection_seg_exe_er_msg"], "")
                self.assertEqual(saved[artery + "_lumen_er_msg"], "")
            self.assertEqual(saved["complete_xsection_CTA"], "1")
            self.assertEqual(saved["complete_xsection_seg"], "1")
            self.assertEqual(saved["complete_lumen_correction"], "1")

    def test_unreliable_applicability_remains_strict(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths, xsection_runner, _ = make_controller(
                temporary_directory,
                selected_arteries=("LCx",),
            )
            case_path = case_paths[0]
            use_json_path = (
                case_path / "Preproc" / "CSV" /
                "coronarySegmentsUse_BSRJZ.json"
            )
            use_json_path.write_text("{invalid json", encoding="utf-8")

            self.assertEqual(
                controller.GenerateTable(
                    maskXsection=True, maskXsectionOnly=False
                ),
                [],
            )
            outcomes = controller.BatchXsection()

            self.assertEqual(xsection_runner.calls, [])
            for artifact_type in ("CTA", "seg"):
                outcome = outcomes[("CASE_001", "LCx", artifact_type)]
                self.assertFalse(outcome["complete"])
                self.assertIn(
                    "Cannot verify per-case artery applicability",
                    outcome["error_message"],
                )
            saved = read_csv(controller.XsectionRecordPath)[0]
            self.assertEqual(saved["complete_xsection_CTA"], "0")
            self.assertEqual(saved["complete_xsection_seg"], "0")

    def test_generate_table_off_reconstructs_only_with_durable_marker(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths, _, _ = make_controller(temporary_directory)
            case_path = case_paths[0]
            set_case_artery_use(case_path, ())
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            controller._case_applicability = {}

            verified = controller._case_artery_applicability(case_path)

            self.assertTrue(verified.reliable)
            self.assertEqual(verified.applicable_arteries, ())
            controller._write_table(
                controller.XsectionTablePath, controller.XSECTION_HEADERS, []
            )
            controller._case_applicability = {}

            unverified = controller._case_artery_applicability(case_path)

            self.assertFalse(unverified.reliable)
            self.assertEqual(
                unverified.applicable_arteries, tuple(controller.SelectedArteries)
            )

    def test_stale_case_filter_error_does_not_block_applicable_artery(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths, xsection_runner, lumen_runner = make_controller(
                temporary_directory,
            )
            case_path = case_paths[0]
            set_case_artery_use(case_path, ("LCx",))
            controller.GenerateTable(maskXsection=True, maskXsectionOnly=False)
            record_map = controller._load_xsection_record_map()
            record = controller._ensure_case_record(record_map, case_path)
            record["filter_mask_complete"] = 0
            record["filter_mask_er_msg"] = (
                "mLAD: No Filter Mask row for mLAD; "
                "mRCA: No Filter Mask row for mRCA"
            )
            record["filter_mask_fingerprint"] = "legacy-case-fingerprint"
            controller._save_xsection_record_map(record_map)

            xsection_outcomes = controller.BatchXsection()
            lumen_outcomes = controller.BatchLumenCorrection()

            self.assertTrue(
                xsection_outcomes[("CASE_001", "LCx", "seg")]["complete"]
            )
            self.assertTrue(lumen_outcomes[("CASE_001", "LCx")]["complete"])
            self.assertEqual(len(xsection_runner.calls), 2)
            self.assertEqual(len(lumen_runner.calls), 1)

    def test_worker_initializer_ignores_sigint_only_in_child_processes(self):
        with mock.patch.object(
                BatchSliceRadiomicsMain.multiprocessing,
                "current_process",
                return_value=SimpleNamespace(name="SpawnProcess-1")), mock.patch.object(
                    BatchSliceRadiomicsMain.signal, "signal"
                ) as signal_mock:
            BatchSliceRadiomicsMain._initialize_batch_slice_worker()
            signal_mock.assert_called_once_with(
                BatchSliceRadiomicsMain.signal.SIGINT,
                BatchSliceRadiomicsMain.signal.SIG_IGN,
            )

        with mock.patch.object(
                BatchSliceRadiomicsMain.multiprocessing,
                "current_process",
                return_value=SimpleNamespace(name="MainProcess")), mock.patch.object(
                    BatchSliceRadiomicsMain.signal, "signal"
                ) as signal_mock:
            BatchSliceRadiomicsMain._initialize_batch_slice_worker()
            signal_mock.assert_not_called()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    unittest.main()
