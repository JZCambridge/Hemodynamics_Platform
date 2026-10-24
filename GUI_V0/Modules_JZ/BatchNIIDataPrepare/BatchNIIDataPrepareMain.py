# -*- coding: utf-8 -*-
"""Prepare ranked NIfTI cases for the Batch Coronary workflow.

The filesystem functions in this module deliberately do not depend on Qt.  The
``BatchNIIDataPrepare`` class is only a thin controller used by the main HEDYS
window, while :func:`prepare_batch_data` can also be called from tests or the
command line.
"""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime

try:
    from PySide2.QtWidgets import QFileDialog
except ImportError:  # Core preparation remains usable without Qt.
    QFileDialog = None


DEFAULT_INPUT_FOLDER = r"G:\FishAndChips_C2\test_codex\temp_nii"
DEFAULT_OUTPUT_FOLDER = r"G:\FishAndChips_C2\test_codex"
OUTPUT_FOLDER_NAME = "0_Initial_Segmentation"
MANIFEST_FILE_NAME = "BatchNIIDataPrepare_manifest.csv"
DEFAULT_CASES_PER_BATCH = 100
MAX_CASES_PER_BATCH = 100
HASH_BLOCK_SIZE = 1024 * 1024

WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5",
    "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
    "LPT6", "LPT7", "LPT8", "LPT9",
}

CANONICAL_CTA_RELATIVE_PATH = Path("CTA") / "CTA.nii.gz"
PIPELINE_CTA_RELATIVE_PATHS = (
    ("ori_data", Path("Preproc") / "OriData" / "CTA.nii.gz"),
    ("branches", Path("Preproc") / "OriData" / "Branches" / "CTA.nii.gz"),
    ("xsection_fix_ext", Path("Preproc") / "XsectionFixExt" / "CTA.nii.gz"),
    ("ori_volume", Path("Preproc") / "OriVolume" / "CTA.nii.gz"),
)

MANIFEST_FIELDS = (
    "rank",
    "source_file",
    "source_path",
    "case_id",
    "batch_index",
    "batch_name",
    "batch_position",
    "cases_per_batch",
    "case_relative_path",
    "source_size_bytes",
    "source_sha256",
    "canonical_cta_path",
    "canonical_cta_size_bytes",
    "canonical_cta_sha256",
    "canonical_verified",
    "ori_data_cta_path",
    "ori_data_cta_sha256",
    "branches_cta_path",
    "branches_cta_sha256",
    "xsection_fix_ext_cta_path",
    "xsection_fix_ext_cta_sha256",
    "ori_volume_cta_path",
    "ori_volume_cta_sha256",
    "pipeline_copies_requested",
    "pipeline_copies_verified",
    "all_verified",
)


@dataclass(frozen=True)
class CasePlan:
    """One deterministic source-to-case assignment."""

    rank: int
    source_path: Path
    case_id: str
    batch_index: int
    batch_name: str
    batch_position: int
    case_path: Path
    case_relative_path: str


@dataclass(frozen=True)
class PreparationResult:
    """Concise result returned to the UI and command-line caller."""

    case_count: int
    batch_count: int
    copied_file_count: int
    unchanged_file_count: int
    manifest_path: Path
    pipeline_ready: bool
    cases_per_batch: int

    def summary(self):
        readiness = "pipeline-ready" if self.pipeline_ready else "canonical CTA only"
        return (
            "Prepared {0} cases in {1} batches (up to {2} cases each, {3}); "
            "copied {4}, unchanged {5}.\nManifest: {6}"
        ).format(
            self.case_count,
            self.batch_count,
            self.cases_per_batch,
            readiness,
            self.copied_file_count,
            self.unchanged_file_count,
            self.manifest_path,
        )


def sanitize_case_id(file_name):
    """Return a Windows-safe case ID derived from a ``.nii.gz`` filename."""

    if not file_name.lower().endswith(".nii.gz"):
        raise ValueError("Only .nii.gz input files are supported: {0}".format(file_name))

    base_name = file_name[:-7]
    case_id = re.sub(r"[^A-Za-z0-9_]", "", base_name)
    if not case_id:
        raise ValueError(
            "The filename has no usable letters, digits, or underscores: {0}".format(file_name)
        )
    if case_id.upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError(
            "The sanitized case ID is a reserved Windows name: {0}".format(case_id)
        )
    return case_id


def discover_nifti_files(input_folder):
    """Return top-level ``.nii.gz`` files in deterministic filename order."""

    if not str(input_folder).strip():
        raise ValueError("Input folder is required.")
    input_path = Path(input_folder).expanduser()
    if not input_path.exists():
        raise FileNotFoundError("Input folder does not exist: {0}".format(input_path))
    if not input_path.is_dir():
        raise NotADirectoryError("Input path is not a folder: {0}".format(input_path))

    files = [
        path
        for path in input_path.iterdir()
        if path.is_file() and path.name.lower().endswith(".nii.gz")
    ]
    files.sort(key=lambda path: (path.name.casefold(), path.name))
    if not files:
        raise ValueError("No top-level .nii.gz files were found in: {0}".format(input_path))
    empty_files = [path.name for path in files if path.stat().st_size == 0]
    if empty_files:
        raise ValueError(
            "Empty .nii.gz input files are not allowed: {0}".format(
                ", ".join(empty_files)
            )
        )
    return files


def _validate_cases_per_batch(cases_per_batch):
    if isinstance(cases_per_batch, bool) or not isinstance(cases_per_batch, int):
        raise TypeError("Cases per batch must be an integer.")
    if not 1 <= cases_per_batch <= MAX_CASES_PER_BATCH:
        raise ValueError(
            "Cases per batch must be between 1 and {0}.".format(
                MAX_CASES_PER_BATCH
            )
        )
    return cases_per_batch


def build_case_plan(
    input_folder,
    output_folder,
    cases_per_batch=DEFAULT_CASES_PER_BATCH,
):
    """Preflight names and return all deterministic case assignments."""

    cases_per_batch = _validate_cases_per_batch(cases_per_batch)
    source_files = discover_nifti_files(input_folder)
    if not str(output_folder).strip():
        raise ValueError("Output folder is required.")
    output_root = Path(output_folder).expanduser()
    data_root = output_root / OUTPUT_FOLDER_NAME
    seen_case_ids = {}
    plans = []

    for zero_index, source_path in enumerate(source_files):
        case_id = sanitize_case_id(source_path.name)
        collision_key = case_id.casefold()
        if collision_key in seen_case_ids:
            raise ValueError(
                "Case-ID collision after sanitizing '{0}' and '{1}': {2}".format(
                    seen_case_ids[collision_key], source_path.name, case_id
                )
            )
        seen_case_ids[collision_key] = source_path.name

        batch_index = zero_index // cases_per_batch
        batch_name = "batch_{0}".format(batch_index)
        case_relative = Path(batch_name) / case_id
        plans.append(
            CasePlan(
                rank=zero_index + 1,
                source_path=source_path.resolve(),
                case_id=case_id,
                batch_index=batch_index,
                batch_name=batch_name,
                batch_position=(zero_index % cases_per_batch) + 1,
                case_path=data_root / case_relative,
                case_relative_path=case_relative.as_posix(),
            )
        )
    return plans


def _validate_existing_case_layout(plans, output_folder, cases_per_batch):
    """Prevent one case being discoverable in both its old and new batch."""

    data_root = Path(output_folder).expanduser() / OUTPUT_FOLDER_NAME
    if not data_root.is_dir():
        return

    expected_by_case_id = {plan.case_id.casefold(): plan.case_path for plan in plans}
    conflicts = []
    batch_paths = sorted(data_root.iterdir(), key=lambda path: path.name.casefold())
    for batch_path in batch_paths:
        valid_batch_name = re.fullmatch(r"batch_[0-9]+", batch_path.name)
        if not batch_path.is_dir() or valid_batch_name is None:
            continue
        for case_path in batch_path.iterdir():
            if not case_path.is_dir() or not (case_path / "0.dcm").is_file():
                continue
            expected_path = expected_by_case_id.get(case_path.name.casefold())
            if expected_path is None:
                continue
            actual_key = os.path.normcase(os.path.abspath(str(case_path)))
            expected_key = os.path.normcase(os.path.abspath(str(expected_path)))
            if actual_key != expected_key:
                conflicts.append((case_path.name, case_path, expected_path))

    if conflicts:
        case_id, current_path, planned_path = conflicts[0]
        message = (
            "Existing batch layout conflicts with cases_per_batch={0}: case "
            "'{1}' is already in '{2}' but would move to '{3}'. Choose an "
            "empty output parent or keep the existing batch size. Existing "
            "case folders are never moved or deleted automatically."
        )
        raise ValueError(
            message.format(cases_per_batch, case_id, current_path, planned_path)
        )


def sha256_file(file_path):
    """Calculate SHA-256 using bounded memory (compatible with Python 3.7)."""

    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file_object:
        while True:
            block = file_object.read(HASH_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_existing_targets(plans, output_folder, create_pipeline_folders):
    """Reject known target conflicts before creating or replacing anything."""

    output_root = Path(output_folder).expanduser()
    data_root = output_root / OUTPUT_FOLDER_NAME
    for path, description in (
        (output_root, "output path"),
        (data_root, "initial-segmentation path"),
    ):
        if path.exists() and not path.is_dir():
            raise NotADirectoryError("Existing {0} is not a folder: {1}".format(description, path))

    for plan in plans:
        batch_path = plan.case_path.parent
        directory_targets = [
            (batch_path, "batch path"),
            (plan.case_path, "case path"),
            (plan.case_path / "CTA", "CTA path"),
        ]
        if create_pipeline_folders:
            directory_targets.append((batch_path / "Table", "batch Table path"))
        for path, description in directory_targets:
            if path.exists() and not path.is_dir():
                raise NotADirectoryError(
                    "Existing {0} is not a folder: {1}".format(description, path)
                )

        marker_path = plan.case_path / "0.dcm"
        if marker_path.exists():
            if not marker_path.is_file():
                raise ValueError("Existing DICOM marker is not a file: {0}".format(marker_path))
            if marker_path.stat().st_size != 0:
                raise ValueError(
                    "Refusing to replace non-empty DICOM marker: {0}".format(marker_path)
                )

        destination_paths = [plan.case_path / CANONICAL_CTA_RELATIVE_PATH]
        if create_pipeline_folders:
            destination_paths.extend(
                plan.case_path / relative_path
                for _, relative_path in PIPELINE_CTA_RELATIVE_PATHS
            )
        for destination in destination_paths:
            if destination.exists() and not destination.is_file():
                raise ValueError("Existing CTA destination is not a file: {0}".format(destination))
            parent = destination.parent
            if parent.exists() and not parent.is_dir():
                raise NotADirectoryError("CTA parent is not a folder: {0}".format(parent))


def _copy_and_verify(source, destination, expected_size, expected_sha256):
    """Synchronize one destination and return ``(hash, was_copied)``."""

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if destination.stat().st_size == expected_size:
            destination_hash = sha256_file(destination)
            if destination_hash == expected_sha256:
                return destination_hash, False

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{0}.".format(destination.name),
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(str(source), str(temporary_path))
        temporary_size = temporary_path.stat().st_size
        temporary_hash = sha256_file(temporary_path)
        if temporary_size != expected_size or temporary_hash != expected_sha256:
            raise IOError(
                "Copied CTA verification failed for temporary file: {0}".format(temporary_path)
            )

        os.replace(str(temporary_path), str(destination))
        final_size = destination.stat().st_size
        final_hash = sha256_file(destination)
        if final_size != expected_size or final_hash != expected_sha256:
            raise IOError("Copied CTA verification failed: {0}".format(destination))
        return final_hash, True
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_manifest_atomic(manifest_path, rows):
    """Publish a complete deterministic CSV without exposing a partial file."""

    manifest_path = Path(manifest_path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{0}.".format(manifest_path.name),
        suffix=".tmp",
        dir=str(manifest_path.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=MANIFEST_FIELDS,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary_path), str(manifest_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def prepare_batch_data(
    input_folder,
    output_folder,
    create_pipeline_folders=True,
    batch_preproc=None,
    cases_per_batch=DEFAULT_CASES_PER_BATCH,
):
    """Prepare, checksum, and record all cases.

    ``batch_preproc`` must expose the existing ``CreateFolders(CasePath=...)``
    method when ``create_pipeline_folders`` is true.
    """

    cases_per_batch = _validate_cases_per_batch(cases_per_batch)
    plans = build_case_plan(
        input_folder,
        output_folder,
        cases_per_batch=cases_per_batch,
    )
    if create_pipeline_folders:
        if batch_preproc is None or not callable(getattr(batch_preproc, "CreateFolders", None)):
            raise ValueError(
                "BatchCoronary CreateFolders is required when pipeline folders are requested."
            )

    _validate_existing_case_layout(plans, output_folder, cases_per_batch)
    _validate_existing_targets(plans, output_folder, create_pipeline_folders)

    # Hash every source before any output is created.  This catches unreadable
    # inputs and also gives every subsequent copy one immutable expected value.
    source_information = []
    for plan in plans:
        source_information.append(
            (plan.source_path.stat().st_size, sha256_file(plan.source_path))
        )

    output_root = Path(output_folder).expanduser()
    data_root = output_root / OUTPUT_FOLDER_NAME
    data_root.mkdir(parents=True, exist_ok=True)

    rows = []
    copied_file_count = 0
    unchanged_file_count = 0

    for plan, source_info in zip(plans, source_information):
        source_size, source_hash = source_info
        plan.case_path.mkdir(parents=True, exist_ok=True)

        marker_path = plan.case_path / "0.dcm"
        if not marker_path.exists():
            marker_path.touch(exist_ok=False)

        if create_pipeline_folders:
            # BatchCoronary writes shared CSV files relative to its working
            # directory.  Each batch is run independently, so each batch must
            # have its own Table directory rather than one at data_root.
            batch_preproc.TableDir = str(plan.case_path.parent / "Table")
            batch_preproc.CreateFolders(CasePath=str(plan.case_path))

        destination_results = {}
        canonical_path = plan.case_path / CANONICAL_CTA_RELATIVE_PATH
        canonical_hash, copied = _copy_and_verify(
            plan.source_path, canonical_path, source_size, source_hash
        )
        destination_results["canonical"] = (canonical_path, canonical_hash)
        copied_file_count += int(copied)
        unchanged_file_count += int(not copied)

        if create_pipeline_folders:
            for destination_key, relative_path in PIPELINE_CTA_RELATIVE_PATHS:
                destination_path = plan.case_path / relative_path
                destination_hash, copied = _copy_and_verify(
                    plan.source_path, destination_path, source_size, source_hash
                )
                destination_results[destination_key] = (destination_path, destination_hash)
                copied_file_count += int(copied)
                unchanged_file_count += int(not copied)

        canonical_verified = canonical_hash == source_hash
        pipeline_verified = create_pipeline_folders and all(
            destination_results[key][1] == source_hash
            for key, _ in PIPELINE_CTA_RELATIVE_PATHS
        )
        all_verified = canonical_verified and (
            pipeline_verified if create_pipeline_folders else True
        )
        if not all_verified:
            raise IOError("CTA verification failed for case: {0}".format(plan.case_id))

        row = {
            "rank": plan.rank,
            "source_file": plan.source_path.name,
            "source_path": str(plan.source_path),
            "case_id": plan.case_id,
            "batch_index": plan.batch_index,
            "batch_name": plan.batch_name,
            "batch_position": plan.batch_position,
            "cases_per_batch": cases_per_batch,
            "case_relative_path": plan.case_relative_path,
            "source_size_bytes": source_size,
            "source_sha256": source_hash,
            "canonical_cta_path": str(canonical_path.resolve()),
            "canonical_cta_size_bytes": canonical_path.stat().st_size,
            "canonical_cta_sha256": canonical_hash,
            "canonical_verified": canonical_verified,
            "ori_data_cta_path": "",
            "ori_data_cta_sha256": "",
            "branches_cta_path": "",
            "branches_cta_sha256": "",
            "xsection_fix_ext_cta_path": "",
            "xsection_fix_ext_cta_sha256": "",
            "ori_volume_cta_path": "",
            "ori_volume_cta_sha256": "",
            "pipeline_copies_requested": create_pipeline_folders,
            "pipeline_copies_verified": pipeline_verified,
            "all_verified": all_verified,
        }
        for destination_key, _ in PIPELINE_CTA_RELATIVE_PATHS:
            if destination_key in destination_results:
                destination_path, destination_hash = destination_results[destination_key]
                row["{0}_cta_path".format(destination_key)] = str(destination_path.resolve())
                row["{0}_cta_sha256".format(destination_key)] = destination_hash
        rows.append(row)

    manifest_path = data_root / MANIFEST_FILE_NAME
    _write_manifest_atomic(manifest_path, rows)

    return PreparationResult(
        case_count=len(plans),
        batch_count=(len(plans) + cases_per_batch - 1) // cases_per_batch,
        copied_file_count=copied_file_count,
        unchanged_file_count=unchanged_file_count,
        manifest_path=manifest_path.resolve(),
        pipeline_ready=create_pipeline_folders,
        cases_per_batch=cases_per_batch,
    )


def _widget_text(widget):
    if hasattr(widget, "text"):
        return widget.text().strip()
    return widget.toPlainText().strip()


def _set_widget_text(widget, text):
    if hasattr(widget, "setPlainText"):
        widget.setPlainText(str(text))
    else:
        widget.setText(str(text))


class BatchNIIDataPrepare:
    """Qt controller for the Batch NII Data Prepare page."""

    def __init__(self, UI=None, Hedys=None):
        self.ui = UI
        self.modelui = getattr(Hedys, "ui", None) if Hedys is not None else None
        self.batch_preproc = getattr(Hedys, "BatchPreproc", None) if Hedys is not None else None

        if self.ui is not None:
            self.ui.ChooseInputFolder_Btn_BNDP.clicked.connect(
                lambda: self.ChooseInputFolder()
            )
            self.ui.ChooseOutputFolder_Btn_BNDP.clicked.connect(
                lambda: self.ChooseOutputFolder()
            )
            self.ui.Prepare_Btn_BNDP.clicked.connect(
                lambda: self.PrepareDataFromUI()
            )

    def ChooseInputFolder(self):
        if QFileDialog is None:
            raise RuntimeError("PySide2 is required to open a folder dialog.")
        start_path = _widget_text(self.ui.InputFolderPath_Txt_BNDP)
        selected = QFileDialog.getExistingDirectory(
            self.ui, "Choose input NIfTI folder", start_path
        )
        if selected:
            _set_widget_text(self.ui.InputFolderPath_Txt_BNDP, selected)

    def ChooseOutputFolder(self):
        if QFileDialog is None:
            raise RuntimeError("PySide2 is required to open a folder dialog.")
        start_path = _widget_text(self.ui.OutputFolderPath_Txt_BNDP)
        selected = QFileDialog.getExistingDirectory(
            self.ui, "Choose output folder", start_path
        )
        if selected:
            _set_widget_text(self.ui.OutputFolderPath_Txt_BNDP, selected)

    def PrepareDataFromUI(self):
        """Run from the Qt signal while keeping validation errors in the UI."""

        try:
            return self.PrepareData()
        except Exception:
            return None

    def PrepareData(
        self,
        input_folder=None,
        output_folder=None,
        create_folders=None,
        cases_per_batch=None,
    ):
        if input_folder is None:
            input_folder = _widget_text(self.ui.InputFolderPath_Txt_BNDP)
        if output_folder is None:
            output_folder = _widget_text(self.ui.OutputFolderPath_Txt_BNDP)
        if create_folders is None:
            create_folders = self.ui.CreateFolders_CB_BNDP.isChecked()
        if cases_per_batch is None:
            cases_per_batch = (
                self.ui.CasesPerBatch_SB_BNDP.value()
                if self.ui is not None
                else DEFAULT_CASES_PER_BATCH
            )

        try:
            result = prepare_batch_data(
                input_folder=input_folder,
                output_folder=output_folder,
                create_pipeline_folders=create_folders,
                batch_preproc=self.batch_preproc,
                cases_per_batch=cases_per_batch,
            )
            message = result.summary()
        except Exception as error:
            message = "Preparation failed: {0}".format(error)
            if self.ui is not None:
                _set_widget_text(self.ui.Summary_Txt_BNDP, message)
            self.UpdateMsgLog(message)
            raise

        if self.ui is not None:
            _set_widget_text(self.ui.Summary_Txt_BNDP, message)
        self.UpdateMsgLog(message)
        return result

    def UpdateMsgLog(self, msg=""):
        now_string = datetime.now().strftime("%d/%b/%y - %H:%M:%S")
        display = "##############{0}##############\n{1}\n############################\n".format(
            now_string, msg
        )
        if self.modelui is not None:
            self.modelui.plainTextEdit_Message.setPlainText(display)
            self.modelui.plainTextEdit_Log.appendPlainText(display)
        print(display)


def _load_standalone_batch_preproc():
    """Load the explicitly required BatchCoronary folder creator for CLI use."""

    gui_root = Path(__file__).resolve().parents[2]
    platform_root = gui_root.parent
    for import_path in (gui_root, platform_root / "Functions_JZ"):
        import_string = str(import_path)
        if import_string not in sys.path:
            sys.path.insert(0, import_string)
    from Modules_YL.BatchCoronary import BatchCoronaryMain

    return BatchCoronaryMain.BatchCoronary()


def main(arguments=None):
    parser = argparse.ArgumentParser(
        description="Prepare ranked .nii.gz cases for Batch Coronary."
    )
    parser.add_argument("--input-folder", default=DEFAULT_INPUT_FOLDER)
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument(
        "--cases-per-batch",
        type=int,
        default=DEFAULT_CASES_PER_BATCH,
        help="Maximum cases per batch folder (1-{0}).".format(MAX_CASES_PER_BATCH),
    )
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Create only 0.dcm and CTA/CTA.nii.gz, without the pipeline folders.",
    )
    options = parser.parse_args(arguments)
    create_pipeline_folders = not options.canonical_only
    batch_preproc = (
        _load_standalone_batch_preproc() if create_pipeline_folders else None
    )
    result = prepare_batch_data(
        input_folder=options.input_folder,
        output_folder=options.output_folder,
        create_pipeline_folders=create_pipeline_folders,
        batch_preproc=batch_preproc,
        cases_per_batch=options.cases_per_batch,
    )
    print(result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())

