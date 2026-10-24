import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from BatchNIIDataPrepareMain import (  # noqa: E402
    CANONICAL_CTA_RELATIVE_PATH,
    MANIFEST_FILE_NAME,
    OUTPUT_FOLDER_NAME,
    PIPELINE_CTA_RELATIVE_PATHS,
    build_case_plan,
    prepare_batch_data,
    sanitize_case_id,
    sha256_file,
)


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


class BatchNIIDataPrepareTests(unittest.TestCase):
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

