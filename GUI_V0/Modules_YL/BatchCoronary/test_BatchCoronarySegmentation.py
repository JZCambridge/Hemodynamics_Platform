import csv
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import xml.etree.ElementTree as ET


MODULE_DIRECTORY = Path(__file__).resolve().parent
GUI_ROOT = MODULE_DIRECTORY.parents[1]
FUNCTIONS_ROOT = GUI_ROOT.parent / "Functions_JZ"
for import_path in (GUI_ROOT, FUNCTIONS_ROOT):
    import_string = str(import_path)
    if import_string not in sys.path:
        sys.path.insert(0, import_string)


from Modules_YL.BatchCoronary import BatchCoronaryMain  # noqa: E402


STATUS_FIELDS = BatchCoronaryMain.SEGMENTATION_STATUS_FIELDS


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def segmentation_record(case_id, batch_name, **updates):
    record = {
        "case_id": case_id,
        "batch_name": batch_name,
        "case_relative_path": "{0}/{1}".format(batch_name, case_id),
        "run_segmentation": 1,
        "seg_exe_failed": 0,
        "seg_exe_er_msg": "",
        "coronary_segment_failed": 0,
        "complete_segmentation": 1,
    }
    record.update(updates)
    return record


def merge_worker(manifest_path, initial_root, result_queue):
    """Top-level worker so Windows' spawn multiprocessing can pickle it."""

    try:
        result = BatchCoronaryMain.merge_batch_segmentation_records(
            manifest_path, initial_root
        )
        result_queue.put((True, result["record_count"]))
    except Exception as error:
        result_queue.put((False, repr(error)))


def make_case(batch_path, case_id):
    case_path = Path(batch_path) / case_id
    ori_data_path = case_path / "Preproc" / "OriData"
    ori_data_path.mkdir(parents=True)
    (ori_data_path / "CTA.nii.gz").write_bytes(b"cta")
    (case_path / "0.dcm").write_bytes(b"")
    return case_path


def centerline_document(branch_name, coordinates=None):
    if coordinates is None:
        coordinates = [{"X": 1, "Y": 2, "Z": 3}]
    return [{"branch_name": branch_name, "coordinates": coordinates}]


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def make_centerline_sources(case_path, left_suffix=".json", right_suffix=".json"):
    ori_data_path = Path(case_path) / "Preproc" / "OriData"
    left_path = ori_data_path / (
        "L_coronary_centerline_network" + left_suffix
    )
    right_path = ori_data_path / (
        "R_coronary_centerline_network" + right_suffix
    )
    write_json(left_path, centerline_document("LM"))
    write_json(right_path, centerline_document("pRCA", coordinates=[]))
    return left_path, right_path


def make_complete_centerline_csv(case_path):
    case_path = Path(case_path)
    csv_path = case_path / "Preproc" / "CSV"
    csv_path.mkdir(parents=True, exist_ok=True)
    left_document = centerline_document("LM")
    right_document = centerline_document("pRCA", coordinates=[])
    write_json(
        csv_path / "L_coronary_centerline_network.json", left_document
    )
    write_json(
        csv_path / "R_coronary_centerline_network.json", right_document
    )
    for side, document in (("L", left_document), ("R", right_document)):
        aggregate_path = csv_path / (
            side + "_coronary_centerline_network.csv"
        )
        with aggregate_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("branch_name", "coordinates")
            )
            writer.writeheader()
            writer.writerows(document)
    (csv_path / "LM.csv").write_text("1,2,3\n", encoding="utf-8")
    return csv_path


def make_batch_controller(temporary_path, case_ids):
    initial_root = Path(temporary_path) / "0_Initial_Segmentation"
    batch_path = initial_root / "batch_0"
    table_path = batch_path / "Table"
    table_path.mkdir(parents=True)
    case_paths = [make_case(batch_path, case_id) for case_id in case_ids]
    executable_path = Path(temporary_path) / "fake_segmentation.exe"
    executable_path.write_bytes(b"fake")

    controller = BatchCoronaryMain.BatchCoronary()
    controller.WorkingDir = str(batch_path)
    controller.TableDir = str(table_path)
    controller.BatchName = "batch_0"
    controller.InitialSegmentationRoot = str(initial_root)
    controller.ManifestPath = str(
        initial_root / "BatchNIIDataPrepare_manifest.csv"
    )
    controller.FailedRootPath = str(Path(temporary_path))
    controller.CGEXEPath = str(executable_path)
    controller.SegmentationTimeoutSeconds = 60
    controller.DCMDirList = [str(path) for path in case_paths]
    controller.CTADirList = [
        str(path / "CTA" / "CTA.nii.gz") for path in case_paths
    ]
    controller.StoreListInformation()
    controller.UpdateMsgLog = lambda _message="": None
    return controller, case_paths


class FakeSafeRunner:
    def __init__(self, successful_case_ids):
        self.successful_case_ids = set(successful_case_ids)
        self.case_ids = []

    def __call__(self, destination, exe_path, timeout_seconds, output_limit):
        del exe_path, timeout_seconds, output_limit
        destination_path = Path(destination)
        case_id = destination_path.parents[1].name
        self.case_ids.append(case_id)
        if case_id in self.successful_case_ids:
            mask_path = (
                destination_path / "sc_labelled_coronary_segment.nii.gz"
            )
            mask_path.write_bytes(b"generated-mask")
            return {"exe_failed": False, "error_message": ""}
        return {
            "exe_failed": True,
            "error_message": "Part1 failed for {0}".format(case_id),
        }


class StaticTextWidget:
    def __init__(self, value):
        self.value = value

    def toPlainText(self):
        return self.value


class TableGenerationUI:
    def __init__(self, base_path):
        self.AVCPath_Txt_BPP = StaticTextWidget(str(base_path))
        self.MeshExePath_Txt_BPP = StaticTextWidget("mesh.exe")
        self.AUIPath_Txt_BPP = StaticTextWidget("aui.exe")


def write_filter_mask_table(table_path, case_paths):
    rows = []
    for case_path in case_paths:
        case_path = Path(case_path)
        input_path = case_path / "Preproc" / "OriData" / "input_mask.nii.gz"
        input_path.write_bytes(b"filter-input-" + case_path.name.encode("ascii"))
        output_folder = (
            case_path / "Preproc" / "OriData" / "Branches"
        )
        output_folder.mkdir(parents=True, exist_ok=True)
        rows.append(
            {
                "MaskPath": str(input_path),
                "MaskVal": "1",
                "NewMaskVal": "1",
                "OutputFolder": str(output_folder),
                "FileNameRef": "Aorta",
            }
        )
    table_path = Path(table_path)
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=BatchCoronaryMain.FILTER_MASK_REQUIRED_COLUMNS
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


class FakeFilterMask:
    def __init__(self, failing_case_ids=()):
        self.failing_case_ids = set(failing_case_ids)
        self.case_ids = []
        self.table_paths = []

    def BatchProcessing(self, table_path):
        self.table_paths.append(str(table_path))
        rows = read_csv(table_path)
        self.assert_one_case(rows)
        output_folder = Path(rows[0]["OutputFolder"])
        case_id = output_folder.parents[2].name
        self.case_ids.append(case_id)
        if case_id in self.failing_case_ids:
            raise RuntimeError("injected Filter Mask failure for " + case_id)
        for row in rows:
            for output_path in BatchCoronaryMain._filter_output_file_paths(
                row["OutputFolder"], row["FileNameRef"]
            ):
                output_path.write_bytes(b"generated-filter-mask")

    @staticmethod
    def assert_one_case(rows):
        case_ids = {
            Path(row["OutputFolder"]).parents[2].name for row in rows
        }
        if len(case_ids) != 1:
            raise AssertionError("Filter Mask was not isolated per case")


class SegmentationRecordTests(unittest.TestCase):
    def test_complete_segmentation_requires_a_nonempty_regular_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            case_path = temporary_path / "CASE_001"
            mask_path = (
                case_path
                / "Preproc"
                / "OriData"
                / "sc_labelled_coronary_segment.nii.gz"
            )

            self.assertFalse(BatchCoronaryMain.is_complete_segmentation(case_path))

            mask_path.parent.mkdir(parents=True)
            mask_path.write_bytes(b"")
            self.assertFalse(BatchCoronaryMain.is_complete_segmentation(case_path))

            mask_path.write_bytes(b"valid-mask")
            self.assertTrue(BatchCoronaryMain.is_complete_segmentation(case_path))

            mask_path.unlink()
            mask_path.mkdir()
            self.assertFalse(BatchCoronaryMain.is_complete_segmentation(case_path))

    def test_batch_record_is_one_atomic_csv_and_replaces_prior_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            record_path = (
                Path(temporary_directory)
                / "batch_0"
                / "Table"
                / "seg_record"
                / "batch_seg.csv"
            )
            first_records = [
                segmentation_record("CASE_001", "batch_0"),
                segmentation_record(
                    "CASE_002",
                    "batch_0",
                    seg_exe_failed=1,
                    seg_exe_er_msg="Part1: exit code 3",
                    coronary_segment_failed=1,
                    complete_segmentation=0,
                ),
            ]

            BatchCoronaryMain.write_batch_segmentation_records(
                record_path, first_records
            )
            self.assertEqual(len(read_csv(record_path)), 2)

            updated_records = list(first_records)
            updated_records[1] = segmentation_record("CASE_002", "batch_0")
            BatchCoronaryMain.write_batch_segmentation_records(
                record_path, updated_records
            )

            rows = read_csv(record_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["case_id"], "CASE_002")
            self.assertEqual(rows[1]["complete_segmentation"], "1")
            self.assertEqual(rows[1]["seg_exe_er_msg"], "")
            self.assertEqual(
                sorted(path.name for path in record_path.parent.iterdir()),
                ["batch_seg.csv", "batch_seg.csv.lock"],
            )

    def test_manifest_merge_preserves_columns_and_adds_missing_cases(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            initial_root = Path(temporary_directory) / "0_Initial_Segmentation"
            initial_root.mkdir()
            manifest_path = initial_root / "BatchNIIDataPrepare_manifest.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=(
                        "case_id",
                        "batch_name",
                        "case_relative_path",
                        "source_sha256",
                    ),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": "CASE_001",
                        "batch_name": "batch_0",
                        "case_relative_path": "batch_0/CASE_001",
                        "source_sha256": "keep-this-value",
                    }
                )

            record_0 = (
                initial_root
                / "batch_0"
                / "Table"
                / "seg_record"
                / "batch_seg.csv"
            )
            record_1 = (
                initial_root
                / "batch_1"
                / "Table"
                / "seg_record"
                / "batch_seg.csv"
            )
            BatchCoronaryMain.write_batch_segmentation_records(
                record_0,
                [segmentation_record("CASE_001", "batch_0")],
            )
            BatchCoronaryMain.write_batch_segmentation_records(
                record_1,
                [
                    segmentation_record(
                        "CASE_101",
                        "batch_1",
                        seg_exe_failed=1,
                        seg_exe_er_msg="timed out",
                        coronary_segment_failed=1,
                        complete_segmentation=0,
                    )
                ],
            )

            merge_result = BatchCoronaryMain.merge_batch_segmentation_records(
                manifest_path, initial_root
            )

            self.assertEqual(Path(merge_result["manifest_path"]), manifest_path)
            self.assertEqual(merge_result["record_count"], 2)
            self.assertEqual(merge_result["row_count"], 2)
            rows = read_csv(manifest_path)
            self.assertEqual(len(rows), 2)
            rows_by_key = {
                (row["batch_name"], row["case_id"]): row for row in rows
            }
            self.assertEqual(
                rows_by_key[("batch_0", "CASE_001")]["source_sha256"],
                "keep-this-value",
            )
            self.assertEqual(
                rows_by_key[("batch_1", "CASE_101")]["complete_segmentation"],
                "0",
            )
            self.assertEqual(
                rows_by_key[("batch_1", "CASE_101")]["seg_exe_er_msg"],
                "timed out",
            )
            self.assertTrue(all(field in rows[0] for field in STATUS_FIELDS))

    def test_missing_manifest_is_created_from_all_batch_records(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            initial_root = Path(temporary_directory) / "0_Initial_Segmentation"
            manifest_path = initial_root / "BatchNIIDataPrepare_manifest.csv"

            for batch_number in range(2):
                batch_name = "batch_{0}".format(batch_number)
                record_path = (
                    initial_root
                    / batch_name
                    / "Table"
                    / "seg_record"
                    / "batch_seg.csv"
                )
                BatchCoronaryMain.write_batch_segmentation_records(
                    record_path,
                    [
                        segmentation_record(
                            "CASE_{0:03d}".format(batch_number), batch_name
                        )
                    ],
                )

            BatchCoronaryMain.merge_batch_segmentation_records(
                manifest_path, initial_root
            )

            rows = read_csv(manifest_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {(row["batch_name"], row["case_id"]) for row in rows},
                {("batch_0", "CASE_000"), ("batch_1", "CASE_001")},
            )
            self.assertTrue(all(field in rows[0] for field in STATUS_FIELDS))

    def test_five_processes_can_merge_the_same_batch_ledgers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            initial_root = Path(temporary_directory) / "0_Initial_Segmentation"
            manifest_path = initial_root / "BatchNIIDataPrepare_manifest.csv"
            for batch_number in range(5):
                batch_name = "batch_{0}".format(batch_number)
                record_path = (
                    initial_root
                    / batch_name
                    / "Table"
                    / "seg_record"
                    / "batch_seg.csv"
                )
                records = [
                    segmentation_record(
                        "CASE_{0}_{1:03d}".format(batch_number, case_number),
                        batch_name,
                    )
                    for case_number in range(20)
                ]
                BatchCoronaryMain.write_batch_segmentation_records(
                    record_path, records
                )

            context = multiprocessing.get_context("spawn")
            result_queue = context.Queue()
            processes = [
                context.Process(
                    target=merge_worker,
                    args=(str(manifest_path), str(initial_root), result_queue),
                )
                for _ in range(5)
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
            self.assertTrue(all(success for success, _ in results), results)
            self.assertEqual(len(read_csv(manifest_path)), 100)

    def test_failed_first_case_does_not_stop_later_cases_and_is_moved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_FAIL", "CASE_OK_1", "CASE_OK_2"),
            )
            runner = FakeSafeRunner(("CASE_OK_1", "CASE_OK_2"))
            messages = []
            controller.UpdateMsgLog = messages.append

            records = controller._run_tracked_segmentation(runner)

            self.assertEqual(
                runner.case_ids,
                ["CASE_FAIL", "CASE_OK_1", "CASE_OK_2"],
            )
            failed_destination = (
                Path(temporary_directory)
                / "0_Failed_Segmentation"
                / "batch_0"
                / "CASE_FAIL"
            )
            self.assertTrue(failed_destination.is_dir())
            self.assertFalse(case_paths[0].exists())
            self.assertTrue(case_paths[1].is_dir())
            self.assertTrue(case_paths[2].is_dir())
            self.assertEqual(
                controller.DCMDirList,
                [str(case_paths[1]), str(case_paths[2])],
            )

            self.assertEqual(len(records), 3)
            records_by_id = {record["case_id"]: record for record in records}
            self.assertEqual(records_by_id["CASE_FAIL"]["complete_segmentation"], 0)
            self.assertEqual(records_by_id["CASE_OK_1"]["complete_segmentation"], 1)
            self.assertEqual(records_by_id["CASE_OK_2"]["complete_segmentation"], 1)

            batch_records = read_csv(controller.BatchSegmentationRecordPath)
            self.assertEqual(len(batch_records), 3)
            manifest_records = read_csv(controller.ManifestPath)
            self.assertEqual(len(manifest_records), 3)
            self.assertEqual(
                {row["case_id"] for row in manifest_records},
                {"CASE_FAIL", "CASE_OK_1", "CASE_OK_2"},
            )
            progress_messages = [
                message.splitlines()[-1]
                for message in messages
                if "Batch progress:" in message
            ]
            self.assertEqual(
                progress_messages,
                [
                    "Batch progress: completed 0, failed 1, processed 1/3.",
                    "Batch progress: completed 1, failed 1, processed 2/3.",
                    "Batch progress: completed 2, failed 1, processed 3/3.",
                ],
            )

    def test_blank_failed_root_skips_failure_until_retry_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_RETRY", "CASE_COMPLETE"),
            )
            controller.FailedRootPath = None
            messages = []
            controller.UpdateMsgLog = messages.append

            first_runner = FakeSafeRunner(("CASE_COMPLETE",))
            controller._run_tracked_segmentation(first_runner)

            self.assertEqual(
                first_runner.case_ids, ["CASE_RETRY", "CASE_COMPLETE"]
            )
            self.assertTrue(case_paths[0].is_dir())
            self.assertTrue(case_paths[1].is_dir())
            self.assertFalse(
                (
                    Path(temporary_directory)
                    / "0_Failed_Segmentation"
                    / "batch_0"
                    / "CASE_RETRY"
                ).exists()
            )
            self.assertTrue(
                any("future retry" in message for message in messages),
                messages,
            )

            # mainFunction rescans the batch on every invocation. Recreate that
            # fixed case snapshot to verify both persisted outcomes are skipped.
            controller.DCMDirList = [str(path) for path in case_paths]
            controller.StoreListInformation()
            second_runner = FakeSafeRunner(("CASE_RETRY", "CASE_COMPLETE"))

            second_records = controller._run_tracked_segmentation(second_runner)

            self.assertEqual(second_runner.case_ids, [])
            records_by_id = {
                record["case_id"]: record for record in second_records
            }
            self.assertEqual(
                str(records_by_id["CASE_RETRY"]["complete_segmentation"]), "0"
            )
            self.assertEqual(
                str(records_by_id["CASE_COMPLETE"]["complete_segmentation"]),
                "1",
            )
            self.assertEqual(
                controller.DCMDirList,
                [str(case_paths[1])],
            )

            controller.DCMDirList = [str(path) for path in case_paths]
            controller.StoreListInformation()
            controller.RerunFailedCasesFlag = True
            retry_runner = FakeSafeRunner(("CASE_RETRY", "CASE_COMPLETE"))

            retry_records = controller._run_tracked_segmentation(retry_runner)

            self.assertEqual(retry_runner.case_ids, ["CASE_RETRY"])
            retry_records_by_id = {
                record["case_id"]: record for record in retry_records
            }
            self.assertEqual(
                retry_records_by_id["CASE_RETRY"]["complete_segmentation"], 1
            )
            self.assertEqual(
                controller.DCMDirList,
                [str(case_paths[0]), str(case_paths[1])],
            )

    def test_known_failure_is_not_executed_and_moves_after_pending_work(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_KNOWN_FAIL", "CASE_PENDING", "CASE_DONE"),
            )
            controller.FailedRootPath = None
            controller.DCMDirList = [str(case_paths[0]), str(case_paths[2])]
            controller.StoreListInformation()
            controller._run_tracked_segmentation(
                FakeSafeRunner(("CASE_DONE",))
            )

            controller.FailedRootPath = str(Path(temporary_directory))
            controller.DCMDirList = [str(path) for path in case_paths]
            controller.StoreListInformation()
            pending_runner = FakeSafeRunner(("CASE_PENDING",))

            def runner(destination, exe_path, timeout_seconds, output_limit):
                self.assertTrue(case_paths[0].is_dir())
                return pending_runner(
                    destination, exe_path, timeout_seconds, output_limit
                )

            controller._run_tracked_segmentation(runner)

            self.assertEqual(pending_runner.case_ids, ["CASE_PENDING"])
            self.assertFalse(case_paths[0].exists())
            self.assertTrue(case_paths[1].is_dir())
            self.assertTrue(case_paths[2].is_dir())
            self.assertTrue(
                (
                    Path(temporary_directory)
                    / "0_Failed_Segmentation"
                    / "batch_0"
                    / "CASE_KNOWN_FAIL"
                ).is_dir()
            )
            self.assertEqual(
                controller.DCMDirList,
                [str(case_paths[1]), str(case_paths[2])],
            )

    def test_legacy_interrupted_segmentation_resumes_with_clean_running_bits(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory, ("CASE_INTERRUPTED",)
            )
            case_path = case_paths[0]
            record_map = controller._load_batch_segmentation_records()
            record = controller._ensure_case_record(record_map, case_path)
            record.update({
                "source_sha256": BatchCoronaryMain._sha256_file(
                    case_path / "Preproc" / "OriData" / "CTA.nii.gz"
                ),
                "run_segmentation": 1,
                "seg_exe_failed": 1,
                "seg_exe_er_msg": (
                    BatchCoronaryMain.SEGMENTATION_INTERRUPTED_MESSAGE
                ),
                "coronary_segment_failed": 1,
                "complete_segmentation": 0,
            })
            controller._save_batch_segmentation_record_map(record_map)

            calls = []

            def runner(destination, exe_path, timeout_seconds, output_limit):
                del exe_path, timeout_seconds, output_limit
                calls.append(Path(destination).parents[1].name)
                running_record = read_csv(
                    controller.BatchSegmentationRecordPath
                )[0]
                self.assertEqual(running_record["seg_exe_failed"], "0")
                self.assertEqual(
                    running_record["coronary_segment_failed"], "0"
                )
                self.assertEqual(
                    running_record["seg_exe_er_msg"],
                    BatchCoronaryMain.SEGMENTATION_INTERRUPTED_MESSAGE,
                )
                (
                    Path(destination)
                    / "sc_labelled_coronary_segment.nii.gz"
                ).write_bytes(b"resumed-mask")
                return {"exe_failed": False, "error_message": ""}

            controller._run_tracked_segmentation(runner)

            self.assertEqual(calls, ["CASE_INTERRUPTED"])
            self.assertEqual(controller.DCMDirList, [str(case_path)])

    def test_changed_cta_retries_a_recorded_failure_while_unchecked(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory, ("CASE_CHANGED_FAILURE",)
            )
            controller.FailedRootPath = None
            controller._run_tracked_segmentation(FakeSafeRunner(()))

            source_path = (
                case_paths[0] / "Preproc" / "OriData" / "CTA.nii.gz"
            )
            source_path.write_bytes(b"changed-after-failure")
            controller.DCMDirList = [str(case_paths[0])]
            controller.StoreListInformation()
            changed_runner = FakeSafeRunner(("CASE_CHANGED_FAILURE",))

            controller._run_tracked_segmentation(changed_runner)

            self.assertEqual(changed_runner.case_ids, ["CASE_CHANGED_FAILURE"])
            self.assertEqual(controller.DCMDirList, [str(case_paths[0])])

    def test_failed_destination_collision_is_preserved_and_does_not_abort(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_COLLISION", "CASE_OK"),
            )
            collision_path = (
                Path(temporary_directory)
                / "0_Failed_Segmentation"
                / "batch_0"
                / "CASE_COLLISION"
            )
            collision_path.mkdir(parents=True)
            sentinel_path = collision_path / "do_not_overwrite.txt"
            sentinel_path.write_text("preserve", encoding="utf-8")
            runner = FakeSafeRunner(("CASE_OK",))

            records = controller._run_tracked_segmentation(runner)

            self.assertEqual(runner.case_ids, ["CASE_COLLISION", "CASE_OK"])
            self.assertFalse(case_paths[0].exists())
            self.assertEqual(sentinel_path.read_text(encoding="utf-8"), "preserve")
            retry_paths = list(collision_path.parent.glob("CASE_COLLISION__retry_*"))
            self.assertEqual(len(retry_paths), 1)
            self.assertTrue(retry_paths[0].is_dir())
            self.assertTrue(case_paths[1].is_dir())
            records_by_id = {record["case_id"]: record for record in records}
            self.assertIn(
                "Existing failed case was preserved",
                records_by_id["CASE_COLLISION"]["seg_exe_er_msg"],
            )
            self.assertEqual(controller.DCMDirList, [str(case_paths[1])])
            self.assertEqual(len(read_csv(controller.BatchSegmentationRecordPath)), 2)
            self.assertEqual(len(read_csv(controller.ManifestPath)), 2)

    def test_resume_skips_unchanged_complete_case_but_reruns_changed_cta(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_RESUME",),
            )
            first_runner = FakeSafeRunner(("CASE_RESUME",))
            controller._run_tracked_segmentation(first_runner)
            self.assertEqual(first_runner.case_ids, ["CASE_RESUME"])

            unchanged_runner = FakeSafeRunner(("CASE_RESUME",))
            controller._run_tracked_segmentation(unchanged_runner)
            self.assertEqual(unchanged_runner.case_ids, [])

            source_cta = case_paths[0] / "Preproc" / "OriData" / "CTA.nii.gz"
            source_cta.write_bytes(b"changed-cta")
            changed_runner = FakeSafeRunner(("CASE_RESUME",))
            controller._run_tracked_segmentation(changed_runner)
            self.assertEqual(changed_runner.case_ids, ["CASE_RESUME"])

    def test_executable_failure_with_nonempty_mask_is_failed_and_later_case_runs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_EXIT_FAIL_WITH_MASK", "CASE_AFTER_FAILURE"),
            )
            calls = []

            def runner(destination, exe_path, timeout_seconds, output_limit):
                del exe_path, timeout_seconds, output_limit
                destination_path = Path(destination)
                case_id = destination_path.parents[1].name
                calls.append(case_id)
                (
                    destination_path / "sc_labelled_coronary_segment.nii.gz"
                ).write_bytes(b"nonempty-mask")
                if case_id == "CASE_EXIT_FAIL_WITH_MASK":
                    return {
                        "exe_failed": True,
                        "error_message": "segmentation executable exited with code 7",
                    }
                return {"exe_failed": False, "error_message": ""}

            records = controller._run_tracked_segmentation(runner)

            self.assertEqual(
                calls, ["CASE_EXIT_FAIL_WITH_MASK", "CASE_AFTER_FAILURE"]
            )
            records_by_id = {record["case_id"]: record for record in records}
            failed_record = records_by_id["CASE_EXIT_FAIL_WITH_MASK"]
            self.assertEqual(failed_record["seg_exe_failed"], 1)
            self.assertEqual(failed_record["complete_segmentation"], 0)
            self.assertIn("code 7", failed_record["seg_exe_er_msg"])
            self.assertFalse(case_paths[0].exists())
            self.assertTrue(
                (
                    Path(temporary_directory)
                    / "0_Failed_Segmentation"
                    / "batch_0"
                    / "CASE_EXIT_FAIL_WITH_MASK"
                    / "Preproc"
                    / "OriData"
                    / "sc_labelled_coronary_segment.nii.gz"
                ).is_file()
            )
            self.assertTrue(case_paths[1].is_dir())
            self.assertEqual(controller.DCMDirList, [str(case_paths[1])])

    def test_all_complete_resume_skips_when_executable_is_now_missing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_RESUME_WITHOUT_EXE",),
            )
            first_runner = FakeSafeRunner(("CASE_RESUME_WITHOUT_EXE",))
            controller._run_tracked_segmentation(first_runner)
            self.assertEqual(first_runner.case_ids, ["CASE_RESUME_WITHOUT_EXE"])

            controller.CGEXEPath = str(
                Path(temporary_directory) / "removed_segmentation.exe"
            )
            skipped_runner = mock.Mock(
                side_effect=AssertionError("completed case must not execute")
            )

            records = controller._run_tracked_segmentation(skipped_runner)

            skipped_runner.assert_not_called()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["complete_segmentation"], "1")
            self.assertEqual(controller.DCMDirList, [str(case_paths[0])])

    def test_table_failure_rolls_back_case_and_later_case_still_runs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_TABLE_FAIL", "CASE_TABLE_OK"),
            )
            for case_path in case_paths:
                controller.CreateFolders(str(case_path))
                use_json_path = (
                    case_path / "Preproc" / "CSV" / "coronarySegmentsUse.json"
                )
                use_json_path.write_text("[]", encoding="utf-8")

            controller.DCMDirList = [str(path) for path in case_paths]
            controller.StoreListInformation()
            controller.ui = TableGenerationUI(temporary_directory)
            controller.CPREXEPath = "cpr.exe"
            controller.STLEXEPath = "stl.exe"
            controller.getFFRresult = False
            controller.getIFRresult = False
            messages = []
            controller.UpdateMsgLog = messages.append

            failed_target = (
                case_paths[0] / "CFD" / "MurraysLaw" / "AVCBranM.csv"
            )
            failed_target.write_text("existing-data", encoding="utf-8")
            original_replace = BatchCoronaryMain.os.replace

            def fail_first_case_replace(source, destination):
                if Path(destination) == failed_target:
                    raise OSError("injected table write failure")
                return original_replace(source, destination)

            with mock.patch.object(
                    BatchCoronaryMain.os,
                    "replace",
                    side_effect=fail_first_case_replace):
                successful_cases = controller.GenerateTable()

            self.assertEqual(successful_cases, [str(case_paths[1])])
            self.assertEqual(controller.DCMDirList, [str(case_paths[1])])
            self.assertEqual(
                controller.CTADirList,
                [str(case_paths[1] / "CTA" / "CTA.nii.gz")],
            )
            self.assertEqual(
                failed_target.read_text(encoding="utf-8"),
                "existing-data",
            )
            self.assertEqual(
                list(failed_target.parent.glob(".AVCBranM.csv.*.tmp")),
                [],
            )
            self.assertTrue(
                (case_paths[1] / "CFD" / "MurraysLaw" / "AVCBranM.csv").is_file()
            )
            self.assertTrue(
                (case_paths[1] / "CFD" / "MurraysLaw" / "AVCParam.csv").is_file()
            )
            self.assertTrue(
                any(
                    "CASE_TABLE_FAIL" in message
                    and "injected table write failure" in message
                    for message in messages
                )
            )

            failed_case_text = str(case_paths[0])
            for attribute_name in BatchCoronaryMain.GENERATED_TABLE_ATTRIBUTES:
                self.assertNotIn(
                    failed_case_text,
                    repr(getattr(controller, attribute_name)),
                    msg=attribute_name,
                )

            aggregate_text = Path(controller.FilterMaskTablePath).read_text(
                encoding="utf-8"
            )
            self.assertIn("CASE_TABLE_OK", aggregate_text)
            self.assertNotIn("CASE_TABLE_FAIL", aggregate_text)

    def test_generated_tables_reuse_the_single_ori_data_cta(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_SINGLE_CTA",),
            )
            case_path = case_paths[0]
            controller.CreateFolders(str(case_path))
            branch = {
                "branch_name": "mLAD",
                "label": [7],
                "branch_label": "2,1,1",
                "use": True,
                "original_csv": "mLADExt.csv",
                "3D_csv": "mLAD3D.csv",
                "original_branch": "mLAD_OriB_",
                "cross_section_image_LI": "mLADCTALI.nii.gz",
                "cross_section_mask_NN": "mLADValNN.nii.gz",
                "cross_section_mask_fix": "mLADValNNFixVal.nii.gz",
                "cross_section_mask_fix_extension": "mLADValNNFE.nii.gz",
                "original_volume_mask_fix": "mLADValNNFEOV.nii.gz",
            }
            use_json_path = (
                case_path / "Preproc" / "CSV" / "coronarySegmentsUse.json"
            )
            use_json_path.write_text(json.dumps([branch]), encoding="utf-8")

            controller.DCMDirList = [str(case_path)]
            controller.StoreListInformation()
            controller.ui = TableGenerationUI(temporary_directory)
            controller.CPREXEPath = "cpr.exe"
            controller.STLEXEPath = "stl.exe"
            controller.getFFRresult = False
            controller.getIFRresult = False

            successful_cases = controller.GenerateTable()

            self.assertEqual(successful_cases, [str(case_path)])
            ori_data_path = case_path / "Preproc" / "OriData"
            authoritative_cta = ori_data_path / "CTA.nii.gz"
            self.assertTrue(authoritative_cta.is_file())
            self.assertFalse((case_path / "CTA" / "CTA.nii.gz").exists())
            self.assertFalse((ori_data_path / "Branches" / "CTA.nii.gz").exists())
            self.assertFalse(
                (case_path / "Preproc" / "XsectionFixExt" / "CTA.nii.gz").exists()
            )
            self.assertFalse(
                (case_path / "Preproc" / "OriVolume" / "CTA.nii.gz").exists()
            )

            self.assertEqual(
                Path(controller.XsectionTable[0][0]),
                ori_data_path,
            )
            case_rows = read_csv(case_path / "Preproc" / "CSV" / "case.csv")
            self.assertEqual(len(case_rows), 2)
            self.assertEqual(
                Path(controller.XsectionTable[0][0])
                / case_rows[0]["inputImagePath"],
                authoritative_cta,
            )
            self.assertEqual(
                Path(controller.XsectionTable[0][0])
                / case_rows[1]["inputImagePath"],
                ori_data_path / "Branches" / "mLAD_OriB_Val.nii.gz",
            )

            ori_volume_row = controller.OriVolumeTable[0]
            self.assertEqual(
                Path(ori_volume_row[0]) / ori_volume_row[1],
                authoritative_cta,
            )
            self.assertEqual(
                Path(ori_volume_row[0]) / ori_volume_row[2],
                case_path
                / "Preproc"
                / "XsectionFixExt"
                / "mLADValNNFE.nii.gz",
            )
            self.assertEqual(
                Path(controller.OriVolumeFixTable[0][0]),
                authoritative_cta,
            )


class CenterlineCsvValidationTests(unittest.TestCase):
    def test_branch_names_are_unique_case_sensitively(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "centerline.json"
            document = [
                {
                    "branch_name": "LCx",
                    "coordinates": [{"X": 1, "Y": 2, "Z": 3}],
                },
                {
                    "branch_name": "LCX",
                    "coordinates": [{"X": 4, "Y": 5, "Z": 6}],
                },
            ]
            write_json(json_path, document)

            self.assertEqual(
                BatchCoronaryMain.validate_centerline_json(json_path),
                document,
            )

            document[1]["branch_name"] = "LCx"
            write_json(json_path, document)
            with self.assertRaisesRegex(ValueError, "duplicate centerline branch LCx"):
                BatchCoronaryMain.validate_centerline_json(json_path)

    def test_staging_accepts_mixed_json_aliases_and_publishes_exact_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            case_path = Path(temporary_directory) / "CASE_MIXED"
            left_path, right_path = make_centerline_sources(
                case_path, left_suffix=".json", right_suffix=".json.txt"
            )

            fingerprint = BatchCoronaryMain.stage_centerline_json_pair(case_path)

            csv_path = case_path / "Preproc" / "CSV"
            staged_left = csv_path / "L_coronary_centerline_network.json"
            staged_right = csv_path / "R_coronary_centerline_network.json"
            self.assertTrue(fingerprint)
            self.assertEqual(staged_left.read_bytes(), left_path.read_bytes())
            self.assertEqual(staged_right.read_bytes(), right_path.read_bytes())
            self.assertTrue(left_path.is_file())
            self.assertTrue(right_path.is_file())
            self.assertEqual(list(csv_path.glob(".*.tmp")), [])

    def test_conflicting_json_aliases_fail_without_partial_replacement(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            case_path = Path(temporary_directory) / "CASE_CONFLICT"
            ori_data_path = case_path / "Preproc" / "OriData"
            csv_path = case_path / "Preproc" / "CSV"
            write_json(
                ori_data_path / "L_coronary_centerline_network.json",
                centerline_document("LM"),
            )
            write_json(
                ori_data_path / "L_coronary_centerline_network.json.txt",
                centerline_document("pLAD"),
            )
            write_json(
                ori_data_path / "R_coronary_centerline_network.json",
                centerline_document("pRCA"),
            )
            csv_path.mkdir(parents=True)
            staged_left = csv_path / "L_coronary_centerline_network.json"
            staged_right = csv_path / "R_coronary_centerline_network.json"
            staged_left.write_text("old-left", encoding="utf-8")
            staged_right.write_text("old-right", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "conflicting L"):
                BatchCoronaryMain.stage_centerline_json_pair(case_path)

            self.assertEqual(staged_left.read_text(encoding="utf-8"), "old-left")
            self.assertEqual(staged_right.read_text(encoding="utf-8"), "old-right")

    def test_identical_json_aliases_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            case_path = Path(temporary_directory) / "CASE_IDENTICAL"
            ori_data_path = case_path / "Preproc" / "OriData"
            document = centerline_document("LM")
            write_json(
                ori_data_path / "L_coronary_centerline_network.json", document
            )
            write_json(
                ori_data_path / "L_coronary_centerline_network.json.txt", document
            )
            write_json(
                ori_data_path / "R_coronary_centerline_network.json",
                centerline_document("pRCA"),
            )

            fingerprint = BatchCoronaryMain.stage_centerline_json_pair(case_path)

            self.assertTrue(fingerprint)
            self.assertTrue(
                (
                    case_path
                    / "Preproc"
                    / "CSV"
                    / "L_coronary_centerline_network.json"
                ).is_file()
            )

    def test_semantic_csv_check_rejects_missing_and_header_only_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            case_path = Path(temporary_directory) / "CASE_INCOMPLETE"
            csv_path = make_complete_centerline_csv(case_path)
            self.assertTrue(
                BatchCoronaryMain.inspect_centerline_csv_case(case_path)["complete"]
            )

            aggregate_path = csv_path / "L_coronary_centerline_network.csv"
            aggregate_path.unlink()
            missing = BatchCoronaryMain.inspect_centerline_csv_case(case_path)
            self.assertFalse(missing["complete"])
            self.assertIn("missing or empty aggregate", missing["error_message"])

            aggregate_path.write_text(
                "branch_name,coordinates\n", encoding="utf-8"
            )
            header_only = BatchCoronaryMain.inspect_centerline_csv_case(case_path)
            self.assertFalse(header_only["complete"])
            self.assertIn("no data rows", header_only["error_message"])

    def test_only_nonempty_json_branches_require_numeric_branch_csv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            case_path = Path(temporary_directory) / "CASE_BRANCHES"
            csv_path = make_complete_centerline_csv(case_path)

            # pRCA has an empty coordinates list, so no pRCA.csv is required.
            self.assertFalse((csv_path / "pRCA.csv").exists())
            self.assertTrue(
                BatchCoronaryMain.is_complete_centerline_csv(case_path)
            )

            (csv_path / "LM.csv").write_text("not,numeric,data\n", encoding="utf-8")
            malformed = BatchCoronaryMain.inspect_centerline_csv_case(case_path)
            self.assertFalse(malformed["complete"])
            self.assertIn("float", malformed["error_message"])

            (csv_path / "LM.csv").write_bytes(b"")
            empty = BatchCoronaryMain.inspect_centerline_csv_case(case_path)
            self.assertFalse(empty["complete"])
            self.assertIn("missing or empty branch", empty["error_message"])

    def test_malformed_json_is_rejected_with_a_useful_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            json_path = Path(temporary_directory) / "centerline.json"
            write_json(
                json_path,
                [
                    {
                        "branch_name": "LM",
                        "coordinates": [{"X": 1, "Y": 2}],
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "invalid Z"):
                BatchCoronaryMain.validate_centerline_json(json_path)


class CenterlineCsvOrchestrationTests(unittest.TestCase):
    def test_zero_success_safety_stop_keeps_entire_batch_in_place(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_ALL_FAIL_1", "CASE_ALL_FAIL_2"),
            )
            for case_path in case_paths:
                make_centerline_sources(case_path)
            messages = []
            controller.UpdateMsgLog = messages.append

            controller.BeginCenterlineCsvRun()
            staged = controller.CopyCenterlineJson()
            outcomes = controller.FinalizeCenterlineCsvRun()

            self.assertTrue(all(result["complete"] for result in staged.values()))
            self.assertTrue(
                all(not result["complete"] for result in outcomes.values())
            )
            self.assertEqual(controller.DCMDirList, [])
            self.assertTrue(all(case_path.is_dir() for case_path in case_paths))
            failed_batch_path = (
                Path(temporary_directory)
                / "0_Failed_Segmentation"
                / "batch_0"
            )
            self.assertTrue(
                all(
                    not (failed_batch_path / case_path.name).exists()
                    for case_path in case_paths
                )
            )

            records = {
                row["case_id"]: row
                for row in read_csv(controller.BatchSegmentationRecordPath)
            }
            for case_path in case_paths:
                record = records[case_path.name]
                self.assertEqual(record["centerline_csv_failed"], "1")
                self.assertEqual(record["complete_centerline_csv"], "0")
            self.assertTrue(
                any(
                    "safety stop" in message.casefold()
                    and "no case folders were moved" in message.casefold()
                    for message in messages
                ),
                messages,
            )

    def test_safeguard_checks_every_case_then_moves_all_failures(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_FAIL_FIRST", "CASE_OK_MIDDLE", "CASE_FAIL_LAST"),
            )
            for case_path in case_paths:
                make_centerline_sources(case_path)
            messages = []
            controller.UpdateMsgLog = messages.append

            resume = controller.BeginCenterlineCsvRun()
            staged = controller.CopyCenterlineJson()
            make_complete_centerline_csv(case_paths[1])
            outcomes = controller.FinalizeCenterlineCsvRun()

            self.assertEqual(
                resume,
                {"skipped": 0, "rerun": 3, "failed": 0, "total": 3},
            )
            self.assertTrue(all(result["complete"] for result in staged.values()))
            self.assertFalse(outcomes[str(case_paths[0])]["complete"])
            self.assertTrue(outcomes[str(case_paths[1])]["complete"])
            self.assertFalse(outcomes[str(case_paths[2])]["complete"])

            self.assertEqual(controller.DCMDirList, [str(case_paths[1])])
            self.assertTrue(case_paths[1].is_dir())
            for failed_index in (0, 2):
                destination = (
                    Path(temporary_directory)
                    / "0_Failed_Segmentation"
                    / "batch_0"
                    / case_paths[failed_index].name
                )
                self.assertTrue(destination.is_dir())
                self.assertFalse(case_paths[failed_index].exists())

            rows = {
                row["case_id"]: row
                for row in read_csv(controller.BatchSegmentationRecordPath)
            }
            self.assertEqual(rows["CASE_FAIL_FIRST"]["centerline_csv_failed"], "1")
            self.assertEqual(rows["CASE_FAIL_FIRST"]["complete_centerline_csv"], "0")
            self.assertEqual(rows["CASE_OK_MIDDLE"]["centerline_csv_failed"], "0")
            self.assertEqual(rows["CASE_OK_MIDDLE"]["complete_centerline_csv"], "1")
            self.assertEqual(rows["CASE_FAIL_LAST"]["centerline_csv_failed"], "1")

            progress_messages = [
                message for message in messages if message.startswith("CSV safeguard:")
            ]
            self.assertEqual(
                progress_messages,
                [
                    "CSV safeguard: ready 0, failed 1, checked 1/3.",
                    "CSV safeguard: ready 1, failed 1, checked 2/3.",
                    "CSV safeguard: ready 1, failed 2, checked 3/3.",
                ],
            )
            manifest_rows = {
                row["case_id"]: row for row in read_csv(controller.ManifestPath)
            }
            self.assertEqual(
                manifest_rows["CASE_OK_MIDDLE"]["complete_centerline_csv"], "1"
            )

    def test_known_centerline_failure_skips_execution_then_moves(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_CSV_FAILED", "CASE_CSV_PENDING"),
            )
            for case_path in case_paths:
                make_centerline_sources(case_path)

            controller.FailedRootPath = None
            controller.DCMDirList = [str(case_paths[0])]
            controller.StoreListInformation()
            controller.BeginCenterlineCsvRun()
            controller.CopyCenterlineJson()
            controller.FinalizeCenterlineCsvRun()
            self.assertTrue(case_paths[0].is_dir())

            controller.FailedRootPath = str(Path(temporary_directory))
            controller.DCMDirList = [str(path) for path in case_paths]
            controller.StoreListInformation()
            resume = controller.BeginCenterlineCsvRun()

            self.assertEqual(
                resume,
                {"skipped": 0, "rerun": 1, "failed": 1, "total": 2},
            )
            self.assertEqual(controller.DCMDirList, [str(case_paths[1])])
            self.assertTrue(case_paths[0].is_dir())
            staged = controller.CopyCenterlineJson()
            self.assertEqual(list(staged), [str(case_paths[1])])
            make_complete_centerline_csv(case_paths[1])

            outcomes = controller.FinalizeCenterlineCsvRun()

            self.assertTrue(outcomes[str(case_paths[0])]["skipped_failed"])
            self.assertTrue(outcomes[str(case_paths[1])]["complete"])
            self.assertFalse(case_paths[0].exists())
            self.assertTrue(
                (
                    Path(temporary_directory)
                    / "0_Failed_Segmentation"
                    / "batch_0"
                    / "CASE_CSV_FAILED"
                ).is_dir()
            )
            self.assertEqual(controller.DCMDirList, [str(case_paths[1])])

    def test_checked_centerline_failure_is_retried(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory, ("CASE_CSV_RETRY",)
            )
            case_path = case_paths[0]
            make_centerline_sources(case_path)
            controller.FailedRootPath = None
            controller.BeginCenterlineCsvRun()
            controller.CopyCenterlineJson()
            controller.FinalizeCenterlineCsvRun()

            controller.FailedRootPath = str(Path(temporary_directory))
            controller.RerunFailedCasesFlag = True
            controller.DCMDirList = [str(case_path)]
            controller.StoreListInformation()
            resume = controller.BeginCenterlineCsvRun()

            self.assertEqual(
                resume,
                {"skipped": 0, "rerun": 1, "failed": 0, "total": 1},
            )
            running_record = read_csv(
                controller.BatchSegmentationRecordPath
            )[0]
            self.assertEqual(running_record["centerline_csv_failed"], "0")
            self.assertEqual(
                running_record["centerline_csv_er_msg"],
                BatchCoronaryMain.CENTERLINE_CSV_INTERRUPTED_MESSAGE,
            )
            controller.CopyCenterlineJson()
            make_complete_centerline_csv(case_path)
            outcomes = controller.FinalizeCenterlineCsvRun()

            self.assertTrue(outcomes[str(case_path)]["complete"])
            self.assertTrue(case_path.is_dir())
            self.assertEqual(controller.DCMDirList, [str(case_path)])

    def test_resume_requires_matching_source_hash_and_valid_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory, ("CASE_RESUME_CSV",)
            )
            case_path = case_paths[0]
            left_source, _right_source = make_centerline_sources(case_path)

            controller.BeginCenterlineCsvRun()
            controller.CopyCenterlineJson()
            csv_path = make_complete_centerline_csv(case_path)
            controller.FinalizeCenterlineCsvRun()

            controller.DCMDirList = [str(case_path)]
            controller.StoreListInformation()
            unchanged = controller.BeginCenterlineCsvRun()
            self.assertEqual(
                unchanged,
                {"skipped": 1, "rerun": 0, "failed": 0, "total": 1},
            )
            self.assertEqual(controller.DCMDirList, [])

            controller.DCMDirList = [str(case_path)]
            controller.StoreListInformation()
            left_source.write_text(
                json.dumps(centerline_document("mLAD")), encoding="utf-8"
            )
            changed_source = controller.BeginCenterlineCsvRun()
            self.assertEqual(
                changed_source,
                {"skipped": 0, "rerun": 1, "failed": 0, "total": 1},
            )

            # Restore and record the changed source/output state as complete.
            controller.CopyCenterlineJson()
            write_json(
                csv_path / "L_coronary_centerline_network.json",
                centerline_document("mLAD"),
            )
            (csv_path / "mLAD.csv").write_text("1,2,3\n", encoding="utf-8")
            (csv_path / "L_coronary_centerline_network.csv").write_text(
                "branch_name,coordinates\nmLAD,[]\n", encoding="utf-8"
            )
            controller.FinalizeCenterlineCsvRun()
            (csv_path / "L_coronary_centerline_network.csv").write_bytes(b"")

            controller.DCMDirList = [str(case_path)]
            controller.StoreListInformation()
            truncated_output = controller.BeginCenterlineCsvRun()
            self.assertEqual(
                truncated_output,
                {"skipped": 0, "rerun": 1, "failed": 0, "total": 1},
            )

    def test_downstream_check_creates_a_ledger_without_segmentation_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory, ("CASE_NO_SEG_RUN",)
            )
            make_centerline_sources(case_paths[0])
            record_path = (
                Path(controller.TableDir)
                / "seg_record"
                / "batch_seg.csv"
            )
            self.assertFalse(record_path.exists())

            controller.BeginCenterlineCsvRun()

            rows = read_csv(record_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["case_id"], "CASE_NO_SEG_RUN")
            self.assertEqual(rows[0]["run_segmentation"], "0")
            self.assertEqual(rows[0]["run_centerline_csv_check"], "1")
            self.assertEqual(rows[0]["centerline_csv_failed"], "0")
            self.assertEqual(
                rows[0]["centerline_csv_er_msg"],
                BatchCoronaryMain.CENTERLINE_CSV_INTERRUPTED_MESSAGE,
            )
            self.assertEqual(rows[0]["complete_centerline_csv"], "0")


class FilterMaskResumeTests(unittest.TestCase):
    def test_failure_does_not_stop_later_cases_and_old_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory, ("CASE_FILTER_FAIL", "CASE_FILTER_OK")
            )
            controller.FilterMaskTablePath = str(
                Path(controller.TableDir) / "FilterMask.csv"
            )
            rows = write_filter_mask_table(
                controller.FilterMaskTablePath, case_paths
            )
            failed_expected = BatchCoronaryMain._filter_output_file_paths(
                rows[0]["OutputFolder"], rows[0]["FileNameRef"]
            )
            for expected_path in failed_expected:
                expected_path.write_bytes(b"preserved-old-output")

            fake_filter = FakeFilterMask(("CASE_FILTER_FAIL",))
            controller.model = SimpleNamespace(filterMask=fake_filter)
            messages = []
            controller.UpdateMsgLog = messages.append

            outcomes = controller.BatchFilterMask()

            self.assertEqual(
                fake_filter.case_ids, ["CASE_FILTER_FAIL", "CASE_FILTER_OK"]
            )
            self.assertFalse(outcomes[str(case_paths[0])]["complete"])
            self.assertTrue(outcomes[str(case_paths[1])]["complete"])
            self.assertEqual(controller.DCMDirList, [str(case_paths[1])])

            failed_destination = (
                Path(temporary_directory)
                / "0_Failed_Segmentation"
                / "batch_0"
                / "CASE_FILTER_FAIL"
            )
            self.assertTrue(failed_destination.is_dir())
            preserved_outputs = list(
                (failed_destination / "Preproc" / "OriData" / "Branches").glob(
                    "*.pre_run_*"
                )
            )
            self.assertEqual(len(preserved_outputs), 2)
            self.assertTrue(
                all(path.read_bytes() == b"preserved-old-output" for path in preserved_outputs)
            )

            successful_expected = BatchCoronaryMain._filter_output_file_paths(
                rows[1]["OutputFolder"], rows[1]["FileNameRef"]
            )
            self.assertTrue(
                BatchCoronaryMain._filter_mask_outputs_complete(
                    successful_expected
                )
            )
            self.assertEqual(
                list(Path(controller.TableDir).glob(".filter_mask_*.csv")), []
            )

            records = {
                row["case_id"]: row
                for row in read_csv(controller.BatchSegmentationRecordPath)
            }
            self.assertEqual(records["CASE_FILTER_FAIL"]["filter_mask_failed"], "1")
            self.assertEqual(records["CASE_FILTER_FAIL"]["complete_filter_mask"], "0")
            self.assertIn(
                "injected Filter Mask failure",
                records["CASE_FILTER_FAIL"]["filter_mask_er_msg"],
            )
            self.assertEqual(records["CASE_FILTER_OK"]["filter_mask_failed"], "0")
            self.assertEqual(records["CASE_FILTER_OK"]["complete_filter_mask"], "1")
            self.assertEqual(
                [
                    message
                    for message in messages
                    if message.startswith("Filter Mask progress:")
                ],
                [
                    "Filter Mask progress: completed 0, failed 1, processed 1/2.",
                    "Filter Mask progress: completed 1, failed 1, processed 2/2.",
                ],
            )

    def test_known_filter_failure_skips_execution_then_moves(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory, ("CASE_FILTER_RECORDED_FAIL",)
            )
            controller.FailedRootPath = None
            controller.FilterMaskTablePath = str(
                Path(controller.TableDir) / "FilterMask.csv"
            )
            write_filter_mask_table(controller.FilterMaskTablePath, case_paths)
            first_filter = FakeFilterMask(("CASE_FILTER_RECORDED_FAIL",))
            controller.model = SimpleNamespace(filterMask=first_filter)
            controller.BatchFilterMask()
            self.assertTrue(case_paths[0].is_dir())

            controller.FailedRootPath = str(Path(temporary_directory))
            controller.DCMDirList = [str(case_paths[0])]
            controller.StoreListInformation()
            skipped_filter = FakeFilterMask()
            controller.model = SimpleNamespace(filterMask=skipped_filter)

            outcomes = controller.BatchFilterMask()

            self.assertEqual(skipped_filter.case_ids, [])
            self.assertTrue(
                outcomes[str(case_paths[0])]["skipped_failed"]
            )
            self.assertFalse(case_paths[0].exists())
            self.assertTrue(
                (
                    Path(temporary_directory)
                    / "0_Failed_Segmentation"
                    / "batch_0"
                    / "CASE_FILTER_RECORDED_FAIL"
                ).is_dir()
            )
            self.assertEqual(controller.DCMDirList, [])

    def test_checked_filter_failure_is_retried(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory, ("CASE_FILTER_RETRY",)
            )
            controller.FailedRootPath = None
            controller.FilterMaskTablePath = str(
                Path(controller.TableDir) / "FilterMask.csv"
            )
            write_filter_mask_table(controller.FilterMaskTablePath, case_paths)
            controller.model = SimpleNamespace(
                filterMask=FakeFilterMask(("CASE_FILTER_RETRY",))
            )
            controller.BatchFilterMask()

            controller.RerunFailedCasesFlag = True
            controller.DCMDirList = [str(case_paths[0])]
            controller.StoreListInformation()
            retry_filter = FakeFilterMask()
            controller.model = SimpleNamespace(filterMask=retry_filter)

            outcomes = controller.BatchFilterMask()

            self.assertEqual(retry_filter.case_ids, ["CASE_FILTER_RETRY"])
            self.assertTrue(outcomes[str(case_paths[0])]["complete"])
            self.assertEqual(controller.DCMDirList, [str(case_paths[0])])

    def test_legacy_interrupted_filter_resumes_with_clean_running_bit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory, ("CASE_FILTER_INTERRUPTED",)
            )
            controller.FilterMaskTablePath = str(
                Path(controller.TableDir) / "FilterMask.csv"
            )
            write_filter_mask_table(controller.FilterMaskTablePath, case_paths)
            plans = BatchCoronaryMain._load_filter_mask_case_plans(
                controller.FilterMaskTablePath, [str(case_paths[0])]
            )
            case_key = BatchCoronaryMain._normalized_path_key(case_paths[0])
            fingerprint, _expected_paths = (
                BatchCoronaryMain._filter_mask_case_fingerprint(
                    case_paths[0], plans[case_key]
                )
            )
            record_map = controller._load_batch_segmentation_records()
            record = controller._ensure_case_record(record_map, case_paths[0])
            record.update({
                "run_filter_mask": 1,
                "filter_mask_failed": 1,
                "filter_mask_er_msg": (
                    BatchCoronaryMain.FILTER_MASK_INTERRUPTED_MESSAGE
                ),
                "complete_filter_mask": 0,
                "filter_mask_input_sha256": fingerprint,
            })
            controller._save_batch_segmentation_record_map(record_map)
            test_case = self

            class InspectingFilterMask(FakeFilterMask):
                def BatchProcessing(self, table_path):
                    running_record = read_csv(
                        controller.BatchSegmentationRecordPath
                    )[0]
                    test_case.assertEqual(
                        running_record["filter_mask_failed"], "0"
                    )
                    test_case.assertEqual(
                        running_record["filter_mask_er_msg"],
                        BatchCoronaryMain.FILTER_MASK_INTERRUPTED_MESSAGE,
                    )
                    super().BatchProcessing(table_path)

            inspecting_filter = InspectingFilterMask()
            controller.model = SimpleNamespace(filterMask=inspecting_filter)

            outcomes = controller.BatchFilterMask()

            self.assertEqual(
                inspecting_filter.case_ids, ["CASE_FILTER_INTERRUPTED"]
            )
            self.assertTrue(outcomes[str(case_paths[0])]["complete"])

    def test_resume_skips_valid_outputs_but_changed_input_forces_rerun(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory, ("CASE_FILTER_RESUME",)
            )
            controller.FilterMaskTablePath = str(
                Path(controller.TableDir) / "FilterMask.csv"
            )
            rows = write_filter_mask_table(
                controller.FilterMaskTablePath, case_paths
            )
            fake_filter = FakeFilterMask()
            controller.model = SimpleNamespace(filterMask=fake_filter)

            first = controller.BatchFilterMask()
            second = controller.BatchFilterMask()

            self.assertTrue(first[str(case_paths[0])]["complete"])
            self.assertFalse(first[str(case_paths[0])]["skipped"])
            self.assertTrue(second[str(case_paths[0])]["complete"])
            self.assertTrue(second[str(case_paths[0])]["skipped"])
            self.assertEqual(fake_filter.case_ids, ["CASE_FILTER_RESUME"])

            Path(rows[0]["MaskPath"]).write_bytes(b"changed-filter-input")
            third = controller.BatchFilterMask()
            self.assertTrue(third[str(case_paths[0])]["complete"])
            self.assertFalse(third[str(case_paths[0])]["skipped"])
            self.assertEqual(
                fake_filter.case_ids,
                ["CASE_FILTER_RESUME", "CASE_FILTER_RESUME"],
            )
            branch_folder = Path(rows[0]["OutputFolder"])
            self.assertEqual(list(branch_folder.glob("*.pre_run_*")), [])

    def test_output_naming_matches_filter_mask_sanitization(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = Path(temporary_directory)
            output_paths = BatchCoronaryMain._filter_output_file_paths(
                output_folder,
                r"C:\source\L-M bad:name.nii.gz",
            )

            self.assertEqual(
                [path.name for path in output_paths],
                [
                    "L-M badname.nii.gzNVl.nii.gz",
                    "L-M badname.nii.gzVal.nii.gz",
                ],
            )

    def test_unrelated_rows_are_ignored_and_zero_row_case_fails_individually(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            controller, case_paths = make_batch_controller(
                temporary_directory,
                ("CASE_FILTER_VALID", "CASE_FILTER_NO_ROWS"),
            )
            controller.FilterMaskTablePath = str(
                Path(controller.TableDir) / "FilterMask.csv"
            )
            generated_rows = write_filter_mask_table(
                controller.FilterMaskTablePath, case_paths
            )
            table_rows = [
                generated_rows[0],
                {
                    "MaskPath": generated_rows[0]["MaskPath"],
                    "MaskVal": "1",
                    "NewMaskVal": "1",
                    "OutputFolder": str(
                        Path(temporary_directory) / "unrelated-output"
                    ),
                    "FileNameRef": "Unrelated",
                },
                {
                    "MaskPath": "",
                    "MaskVal": "",
                    "NewMaskVal": "",
                    "OutputFolder": "",
                    "FileNameRef": "",
                },
            ]
            with Path(controller.FilterMaskTablePath).open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=BatchCoronaryMain.FILTER_MASK_REQUIRED_COLUMNS,
                )
                writer.writeheader()
                writer.writerows(table_rows)

            fake_filter = FakeFilterMask()
            controller.model = SimpleNamespace(filterMask=fake_filter)
            messages = []
            controller.UpdateMsgLog = messages.append

            outcomes = controller.BatchFilterMask()

            self.assertTrue(outcomes[str(case_paths[0])]["complete"])
            self.assertFalse(outcomes[str(case_paths[1])]["complete"])
            self.assertIn(
                "no rows for this case",
                outcomes[str(case_paths[1])]["error_message"],
            )
            self.assertEqual(fake_filter.case_ids, ["CASE_FILTER_VALID"])
            self.assertEqual(controller.DCMDirList, [str(case_paths[0])])
            self.assertTrue(
                any(
                    "ignored rows" in message
                    and "unrelated-output" in message
                    and "<blank>" in message
                    for message in messages
                )
            )
            self.assertTrue(
                (
                    Path(temporary_directory)
                    / "0_Failed_Segmentation"
                    / "batch_0"
                    / "CASE_FILTER_NO_ROWS"
                ).is_dir()
            )


class ResumeUiTests(unittest.TestCase):
    def test_rerun_failed_checkbox_exists_and_defaults_unchecked(self):
        ui_path = GUI_ROOT / "ui" / "BatchPreProcessing.ui"
        root = ET.parse(str(ui_path)).getroot()
        checkbox = root.find(
            ".//widget[@name='RerunFailedCases_CB_BPP']"
        )

        self.assertIsNotNone(checkbox)
        properties = {
            property_node.get("name"): property_node
            for property_node in checkbox.findall("property")
        }
        self.assertEqual(
            properties["checked"].findtext("bool"), "false"
        )
        self.assertEqual(
            properties["text"].findtext("string"),
            "Also rerun failed cases",
        )


if __name__ == "__main__":
    unittest.main()
