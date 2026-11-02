# -*- coding: utf-8 -*-
"""Prepare ranked NIfTI cases for the Batch Coronary workflow.

The filesystem functions in this module deliberately do not depend on Qt.  The
``BatchNIIDataPrepare`` class is only a thin controller used by the main HEDYS
window, while :func:`prepare_batch_data` can also be called from tests or the
command line.
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

try:
    from PySide2.QtWidgets import QApplication, QFileDialog
except ImportError:  # Core preparation remains usable without Qt.
    QApplication = None
    QFileDialog = None


DEFAULT_DATA_ROOT = Path(r"S:\04_JZ\FishAndChips_C2")
DEFAULT_INPUT_FOLDER = str(DEFAULT_DATA_ROOT / "0_Raw_Data")
DEFAULT_OUTPUT_FOLDER = r"S:\04_JZ\FishAndChips_C2_round2"
OUTPUT_FOLDER_NAME = "0_Initial_Segmentation"
MANIFEST_FILE_NAME = "BatchNIIDataPrepare_manifest.csv"
FORMAT_AUDIT_FILE_NAME = "BatchNIIDataPrepare_format_audit.csv"
DEFAULT_CASES_PER_BATCH = 100
MAX_CASES_PER_BATCH = 100
DEFAULT_CASE_LIMIT = 0
DEFAULT_PARALLEL_WORKERS = 1
MAX_PARALLEL_WORKERS = 8
HASH_BLOCK_SIZE = 1024 * 1024
TARGET_CTA_ORIENTATION = "LPS"
SEGMENTATION_FORMAT_PROFILE = "Hymo segmentation input profile v1"
SEGMENTATION_FORMAT_REQUIREMENTS = (
    "empirically validated from successful C1 inputs: readable NIfTI; "
    "3-D scalar image; finite integral decoded HU values "
    "within signed-int16 range; prepared LPS orientation; prepared signed "
    "int16; physical geometry and voxel values preserved without resampling; "
    "destination SHA-256 verified"
)

FORMAT_CHECK_FIELDS = (
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
    "check_destination_sha256",
)
CHECK_PASS = "PASS"
CHECK_FAIL = "FAIL"
CHECK_NOT_RUN = "NOT_RUN"
CHECK_NOT_CHECKED = "NOT_CHECKED"

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
CTA_DESTINATION_RELATIVE_PATHS = (
    ("canonical", CANONICAL_CTA_RELATIVE_PATH),
) + PIPELINE_CTA_RELATIVE_PATHS
CTA_DESTINATION_KEYS = tuple(
    destination_key for destination_key, _ in CTA_DESTINATION_RELATIVE_PATHS
)
DEFAULT_CTA_DESTINATIONS = ("ori_data",)
CTA_DESTINATION_WIDGET_NAMES = (
    ("canonical", "CopyCanonicalCTA_CB_BNDP"),
    ("ori_data", "CopyOriDataCTA_CB_BNDP"),
    ("branches", "CopyBranchesCTA_CB_BNDP"),
    ("xsection_fix_ext", "CopyXsectionFixExtCTA_CB_BNDP"),
    ("ori_volume", "CopyOriVolumeCTA_CB_BNDP"),
)

SEGMENTATION_FIELDS = (
    "run_segmentation",
    "seg_exe_failed",
    "seg_exe_er_msg",
    "coronary_segment_failed",
    "complete_segmentation",
)

DOWNSTREAM_STATUS_FIELDS = (
    "run_centerline_csv_check",
    "centerline_csv_failed",
    "centerline_csv_er_msg",
    "complete_centerline_csv",
    "run_filter_mask",
    "filter_mask_failed",
    "filter_mask_er_msg",
    "complete_filter_mask",
)

XSECTION_STATUS_FIELDS = (
    "run_xsection",
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

PROCESSING_STATUS_FIELDS = (
    SEGMENTATION_FIELDS + DOWNSTREAM_STATUS_FIELDS + XSECTION_STATUS_FIELDS
)

PROCESSING_STATUS_DEFAULTS = {
    "run_segmentation": 0,
    "seg_exe_failed": 0,
    "seg_exe_er_msg": "",
    "coronary_segment_failed": 0,
    "complete_segmentation": 0,
    "run_centerline_csv_check": 0,
    "centerline_csv_failed": 0,
    "centerline_csv_er_msg": "",
    "complete_centerline_csv": 0,
    "run_filter_mask": 0,
    "filter_mask_failed": 0,
    "filter_mask_er_msg": "",
    "complete_filter_mask": 0,
    "run_xsection": 0,
    "LCx_xsection_CTA_exe_failed": 0,
    "LCx_xsection_CTA_exe_er_msg": "",
    "LCx_xsection_CTA_complete": 0,
    "mLAD_xsection_CTA_exe_failed": 0,
    "mLAD_xsection_CTA_exe_er_msg": "",
    "mLAD_xsection_CTA_complete": 0,
    "mRCA_xsection_CTA_exe_failed": 0,
    "mRCA_xsection_CTA_exe_er_msg": "",
    "mRCA_xsection_CTA_complete": 0,
    "complete_xsection_CTA": 0,
    "LCx_xsection_seg_exe_failed": 0,
    "LCx_xsection_seg_exe_er_msg": "",
    "LCx_xsection_seg_complete": 0,
    "mLAD_xsection_seg_exe_failed": 0,
    "mLAD_xsection_seg_exe_er_msg": "",
    "mLAD_xsection_seg_complete": 0,
    "mRCA_xsection_seg_exe_failed": 0,
    "mRCA_xsection_seg_exe_er_msg": "",
    "mRCA_xsection_seg_complete": 0,
    "complete_xsection_seg": 0,
    "LCx_lumen_complete": 0,
    "LCx_lumen_er_msg": "",
    "mLAD_lumen_complete": 0,
    "mLAD_lumen_er_msg": "",
    "mRCA_lumen_complete": 0,
    "mRCA_lumen_er_msg": "",
    "complete_lumen_correction": 0,
}

FORMAT_RESULT_FIELDS = (
    "segmentation_format_profile",
    "segmentation_format_requirements",
    "preparation_parallel_workers",
    "format_check_status",
    "segmentation_software_compatible",
    "format_checks_performed",
    "format_changes",
    "format_check_message",
    "source_dimension",
    "source_components",
    "source_image_size",
    "prepared_image_size",
) + FORMAT_CHECK_FIELDS

FORMAT_AUDIT_FIELDS = (
    "audit_timestamp",
    "rank",
    "source_file",
    "source_path",
    "case_id",
    "batch_index",
    "batch_name",
    "batch_position",
    "case_relative_path",
    "status",
    "processed_so_far",
    "selected_total",
    "passed_so_far",
    "changed_so_far",
    "unchanged_so_far",
    "failed_so_far",
    "excluded_so_far",
) + FORMAT_RESULT_FIELDS + (
    "source_size_bytes",
    "source_sha256",
    "source_orientation",
    "source_pixel_type",
    "source_spacing",
    "prepared_orientation",
    "prepared_pixel_type",
    "prepared_spacing",
    "prepared_cta_size_bytes",
    "prepared_cta_sha256",
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
    "normalize_for_segmentation",
    *FORMAT_RESULT_FIELDS,
    "source_orientation",
    "source_pixel_type",
    "source_spacing",
    "prepared_orientation",
    "prepared_pixel_type",
    "prepared_spacing",
    "prepared_cta_size_bytes",
    "prepared_cta_sha256",
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
) + PROCESSING_STATUS_FIELDS


class CTAFormatRequirementError(ValueError):
    """A case that cannot safely satisfy the segmentation input profile."""

    def __init__(self, message, checks=None, metadata=None):
        super().__init__(message)
        self.checks = dict(checks or {})
        self.metadata = dict(metadata or {})


class CTAPreparationError(IOError):
    """An operational per-case failure with the completed checks attached."""

    def __init__(self, message, checks=None, metadata=None):
        super().__init__(message)
        self.checks = dict(checks or {})
        self.metadata = dict(metadata or {})


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
    cta_destinations: tuple
    preserved_unselected_file_count: int
    normalized_case_count: int
    total_case_count: int
    processed_case_count: int
    passed_case_count: int
    changed_case_count: int
    unchanged_case_count: int
    failed_case_count: int
    excluded_case_count: int
    audit_path: Path
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS
    elapsed_seconds: float = 0.0

    def summary(self):
        destinations = ", ".join(self.cta_destinations)
        message = (
            "Processed {0}/{1} selected cases: passed {2} "
            "(changed {3}, unchanged {4}), failed {5}, excluded {6}.\n"
            "Prepared successful cases in {7} batches (up to {8} cases each); "
            "CTA destinations: {9}; copied {10}, unchanged files {11}.\n"
            "Segmentation format profile: {12}.\n"
            "Requirements: {13}.\n"
            "Parallel preparation workers: {14}.\n"
            "Elapsed time: {15:.1f} seconds; throughput: {16:.2f} cases/minute.\n"
            "Format audit (all selected cases): {17}\n"
            "Preparation manifest (successful cases only): {18}"
        ).format(
            self.processed_case_count,
            self.total_case_count,
            self.passed_case_count,
            self.changed_case_count,
            self.unchanged_case_count,
            self.failed_case_count,
            self.excluded_case_count,
            self.batch_count,
            self.cases_per_batch,
            destinations,
            self.copied_file_count,
            self.unchanged_file_count,
            SEGMENTATION_FORMAT_PROFILE,
            SEGMENTATION_FORMAT_REQUIREMENTS,
            self.parallel_workers,
            self.elapsed_seconds,
            (
                self.processed_case_count * 60.0 / self.elapsed_seconds
                if self.elapsed_seconds > 0 else 0.0
            ),
            self.audit_path,
            self.manifest_path,
        )
        if self.preserved_unselected_file_count:
            message += (
                "\nWarning: preserved {0} pre-existing CTA file(s) in "
                "unselected legacy destinations."
            ).format(self.preserved_unselected_file_count)
        if self.normalized_case_count:
            message += (
                "\nNormalized {0} CTA(s) losslessly to LPS int16 for the "
                "segmentation executable."
            ).format(self.normalized_case_count)
        return message


@dataclass(frozen=True)
class PreparedCTA:
    """One source representation used for all selected case destinations."""

    path: Path
    size: int
    sha256: str
    source_orientation: str
    source_pixel_type: str
    source_spacing: str
    prepared_orientation: str
    prepared_pixel_type: str
    prepared_spacing: str
    normalized: bool
    source_dimension: object
    source_components: object
    source_image_size: str
    prepared_image_size: str
    checks: tuple
    changes: str
    message: str


@dataclass(frozen=True)
class CasePreparationOutcome:
    """Worker result consumed by the coordinator thread."""

    plan: CasePlan
    audit_row: dict
    manifest_row: object
    error: object
    copied_file_count: int = 0
    unchanged_file_count: int = 0
    preserved_unselected_file_count: int = 0
    normalized_case_count: int = 0
    layout_prepared: bool = False


def normalize_cta_destinations(cta_destinations, create_pipeline_folders):
    """Return validated destination keys in deterministic layout order.

    ``None`` retains the historical programmatic behavior: create every CTA
    location with the full pipeline layout, or only ``CTA/CTA.nii.gz`` when
    pipeline folders are disabled.  GUI and normal CLI calls pass an explicit
    selection so their default can remain the single authoritative OriData CTA.
    """

    if cta_destinations is None:
        if create_pipeline_folders:
            return CTA_DESTINATION_KEYS
        return ("canonical",)

    if isinstance(cta_destinations, str):
        requested = (cta_destinations,)
    else:
        try:
            requested = tuple(cta_destinations)
        except TypeError:
            raise TypeError("CTA destinations must be a key or an iterable of keys.")

    normalized_requested = []
    for destination_key in requested:
        if not isinstance(destination_key, str):
            raise TypeError("CTA destination keys must be text.")
        normalized_requested.append(destination_key.strip().lower())

    if not normalized_requested or not any(normalized_requested):
        raise ValueError("Select at least one CTA destination.")

    unknown_keys = sorted(
        set(normalized_requested).difference(CTA_DESTINATION_KEYS)
    )
    if unknown_keys:
        raise ValueError(
            "Unknown CTA destination(s): {0}. Choose from: {1}.".format(
                ", ".join(unknown_keys), ", ".join(CTA_DESTINATION_KEYS)
            )
        )

    requested_set = set(normalized_requested)
    return tuple(
        destination_key
        for destination_key in CTA_DESTINATION_KEYS
        if destination_key in requested_set
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


def _validate_case_limit(case_limit):
    if isinstance(case_limit, bool) or not isinstance(case_limit, int):
        raise TypeError("Case limit must be an integer.")
    if case_limit < 0:
        raise ValueError("Case limit must be 0 (all cases) or a positive integer.")
    return case_limit


def _validate_parallel_workers(parallel_workers):
    if isinstance(parallel_workers, bool) or not isinstance(
            parallel_workers, int):
        raise TypeError("Parallel workers must be an integer.")
    if not 1 <= parallel_workers <= MAX_PARALLEL_WORKERS:
        raise ValueError(
            "Parallel workers must be between 1 and {0}.".format(
                MAX_PARALLEL_WORKERS
            )
        )
    return parallel_workers


def discover_nifti_files(input_folder, case_limit=DEFAULT_CASE_LIMIT):
    """Return sorted top-level NIfTIs, optionally limited to the first cases.

    ``case_limit=0`` preserves the historical behavior and returns every
    top-level ``.nii.gz`` file.  A positive limit is applied only after the
    deterministic filename sort.
    """

    case_limit = _validate_case_limit(case_limit)
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
    if case_limit:
        files = files[:case_limit]
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
    case_limit=DEFAULT_CASE_LIMIT,
):
    """Preflight names and return all deterministic case assignments."""

    cases_per_batch = _validate_cases_per_batch(cases_per_batch)
    source_files = discover_nifti_files(input_folder, case_limit=case_limit)
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


def _format_vector(values):
    return "x".join("{0:.9g}".format(float(value)) for value in values)


def _format_size(values):
    return "x".join(str(int(value)) for value in values)


def _new_format_checks(value=CHECK_NOT_RUN):
    return {field: value for field in FORMAT_CHECK_FIELDS}


def _format_checks_summary(checks):
    return "; ".join(
        "{0}={1}".format(field.replace("check_", ""), checks.get(field, CHECK_NOT_RUN))
        for field in FORMAT_CHECK_FIELDS
    )


def _checks_as_tuple(checks):
    return tuple((field, checks.get(field, CHECK_NOT_RUN)) for field in FORMAT_CHECK_FIELDS)


def _physical_corner_points(image, numpy_module):
    """Return sorted physical corner coordinates for geometry comparison."""

    index_choices = [
        (0, max(int(size) - 1, 0)) for size in image.GetSize()
    ]
    points = [
        image.TransformIndexToPhysicalPoint(tuple(int(value) for value in index))
        for index in itertools.product(*index_choices)
    ]
    return numpy_module.asarray(sorted(points), dtype=numpy_module.float64)


def _physical_geometry_preserved(source_image, prepared_image, numpy_module):
    source_corners = _physical_corner_points(source_image, numpy_module)
    prepared_corners = _physical_corner_points(prepared_image, numpy_module)
    return bool(
        source_corners.shape == prepared_corners.shape
        and numpy_module.allclose(
            source_corners, prepared_corners, rtol=0.0, atol=1e-5
        )
    )


def _voxel_values_equal(first_image, second_image, numpy_module, sitk_module):
    """Compare two decoded images exactly, one plane at a time."""

    if first_image.GetSize() != second_image.GetSize():
        return False
    first = numpy_module.asarray(sitk_module.GetArrayViewFromImage(first_image))
    second = numpy_module.asarray(sitk_module.GetArrayViewFromImage(second_image))
    if first.shape != second.shape:
        return False
    return all(
        numpy_module.array_equal(first[index], second[index])
        for index in range(first.shape[0])
    )


def _direction_is_orthonormal(image, numpy_module):
    dimension = image.GetDimension()
    matrix = numpy_module.asarray(
        image.GetDirection(), dtype=numpy_module.float64
    ).reshape((dimension, dimension))
    identity = numpy_module.eye(dimension, dtype=numpy_module.float64)
    return bool(
        numpy_module.allclose(
            matrix.dot(matrix.T), identity, rtol=0.0, atol=1e-5
        )
        and numpy_module.isclose(
            abs(numpy_module.linalg.det(matrix)), 1.0, rtol=0.0, atol=1e-5
        )
    )


def _load_normalization_dependencies():
    """Load optional image dependencies only when normalization is requested."""

    try:
        import numpy
        import SimpleITK
    except ImportError as error:
        raise RuntimeError(
            "CTA normalization requires NumPy and SimpleITK in the active "
            "environment."
        ) from error
    return numpy, SimpleITK


@contextmanager
def _simpleitk_thread_limit(sitk_module, parallel_workers):
    """Limit nested ITK threads while case-level workers are active.

    SimpleITK's setting is process-global, so only the coordinator changes it
    and always restores the caller's value.  One-worker runs retain the exact
    historical SimpleITK setting.
    """

    if sitk_module is None or parallel_workers <= 1:
        yield
        return

    process_object = sitk_module.ProcessObject
    previous_threads = int(
        process_object.GetGlobalDefaultNumberOfThreads()
    )
    available_cpus = max(1, int(os.cpu_count() or 1))
    worker_thread_limit = max(1, available_cpus // parallel_workers)
    if previous_threads > 0:
        worker_thread_limit = min(previous_threads, worker_thread_limit)

    changed = worker_thread_limit != previous_threads
    if changed:
        process_object.SetGlobalDefaultNumberOfThreads(worker_thread_limit)
    try:
        yield
    finally:
        if changed:
            process_object.SetGlobalDefaultNumberOfThreads(previous_threads)


def _validate_lossless_int16_values(
        image, numpy_module, sitk_module, source_path, checks, metadata):
    """Record and enforce finite, integral, signed-int16 decoded values."""

    data = numpy_module.asarray(sitk_module.GetArrayViewFromImage(image))
    type_limits = numpy_module.iinfo(numpy_module.int16)
    finite = True
    integral = True
    within_range = True
    for plane_index in range(data.shape[0]):
        plane = numpy_module.asarray(data[plane_index])
        if not numpy_module.isfinite(plane).all():
            finite = False
            continue
        if plane.size:
            plane_min = float(numpy_module.min(plane))
            plane_max = float(numpy_module.max(plane))
            if plane_min < type_limits.min or plane_max > type_limits.max:
                within_range = False
            if not numpy_module.array_equal(plane, numpy_module.rint(plane)):
                integral = False

    checks["check_finite"] = CHECK_PASS if finite else CHECK_FAIL
    if finite:
        checks["check_integral_hu"] = CHECK_PASS if integral else CHECK_FAIL
        checks["check_int16_range"] = CHECK_PASS if within_range else CHECK_FAIL
    if not finite:
        raise CTAFormatRequirementError(
            "CTA contains non-finite decoded values.", checks, metadata
        )
    if not integral:
        raise CTAFormatRequirementError(
            "CTA contains fractional values in decoded HU; refusing a lossy "
            "signed-int16 conversion.",
            checks,
            metadata,
        )
    if not within_range:
        raise CTAFormatRequirementError(
            "CTA decoded values exceed signed-int16 limits.", checks, metadata
        )


@contextmanager
def prepared_cta_source(
        source_path,
        temporary_directory,
        normalize_for_segmentation=True,
        source_size=None,
        source_sha256=None):
    """Yield one verified CTA representation for every selected destination.

    ``SimpleITK.DICOMOrient`` only permutes and flips voxel axes; it does not
    interpolate.  The subsequent int16 conversion is allowed only when every
    decoded voxel is finite, integral, and in range.  Thus NIfTI scaling (for
    example ``scl_inter=-1024``) is baked into the stored HU values without a
    numerical loss.
    """

    source_path = Path(source_path)
    checks = _new_format_checks()
    metadata = {
        "source_dimension": "",
        "source_components": "",
        "source_image_size": "",
        "source_orientation": "",
        "source_pixel_type": "",
        "source_spacing": "",
        "prepared_image_size": "",
        "prepared_orientation": "",
        "prepared_pixel_type": "",
        "prepared_spacing": "",
        "prepared_cta_size_bytes": "",
        "prepared_cta_sha256": "",
    }
    if source_size is None:
        source_size = source_path.stat().st_size
    if source_sha256 is None:
        source_sha256 = sha256_file(source_path)
    if not normalize_for_segmentation:
        checks = _new_format_checks(CHECK_NOT_CHECKED)
        yield PreparedCTA(
            path=source_path,
            size=source_size,
            sha256=source_sha256,
            source_orientation="",
            source_pixel_type="",
            source_spacing="",
            prepared_orientation="",
            prepared_pixel_type="",
            prepared_spacing="",
            normalized=False,
            source_dimension="",
            source_components="",
            source_image_size="",
            prepared_image_size="",
            checks=_checks_as_tuple(checks),
            changes="COPIED UNCHECKED: source representation preserved by request",
            message=(
                "Segmentation format compatibility was not checked because "
                "normalization was disabled."
            ),
        )
        return

    numpy_module, sitk = _load_normalization_dependencies()
    try:
        source_image = sitk.ReadImage(str(source_path))
    except Exception as error:
        checks["check_readable_nifti"] = CHECK_FAIL
        raise CTAFormatRequirementError(
            "Input is not a readable NIfTI image: {0}".format(error),
            checks,
            metadata,
        ) from error
    checks["check_readable_nifti"] = CHECK_PASS

    source_dimension = source_image.GetDimension()
    source_components = source_image.GetNumberOfComponentsPerPixel()
    metadata.update({
        "source_dimension": source_dimension,
        "source_components": source_components,
        "source_image_size": _format_size(source_image.GetSize()),
        "source_pixel_type": source_image.GetPixelIDTypeAsString(),
        "source_spacing": _format_vector(source_image.GetSpacing()),
    })
    if source_dimension != 3:
        checks["check_3d"] = CHECK_FAIL
        raise CTAFormatRequirementError(
            "Segmentation requires a 3-D image; input dimension is {0}.".format(
                source_dimension
            ),
            checks,
            metadata,
        )
    checks["check_3d"] = CHECK_PASS
    if source_components != 1:
        checks["check_scalar"] = CHECK_FAIL
        raise CTAFormatRequirementError(
            "Segmentation requires a scalar image; input has {0} components."
            .format(source_components),
            checks,
            metadata,
        )
    checks["check_scalar"] = CHECK_PASS

    direction_ok = _direction_is_orthonormal(source_image, numpy_module)
    checks["check_direction_orthonormal"] = (
        CHECK_PASS if direction_ok else CHECK_FAIL
    )
    if not direction_ok:
        raise CTAFormatRequirementError(
            "Input direction matrix is not orthonormal and cannot be safely "
            "converted by axis permutation/flip alone.",
            checks,
            metadata,
        )

    source_orientation = (
        sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            source_image.GetDirection()
        )
    )
    source_pixel_type = source_image.GetPixelIDTypeAsString()
    source_spacing = _format_vector(source_image.GetSpacing())
    metadata["source_orientation"] = source_orientation
    _validate_lossless_int16_values(
        source_image,
        numpy_module,
        sitk,
        source_path,
        checks,
        metadata,
    )
    already_normalized = (
        source_orientation == TARGET_CTA_ORIENTATION
        and source_image.GetPixelID() == sitk.sitkInt16
    )
    if already_normalized:
        checks["check_output_lps"] = CHECK_PASS
        checks["check_output_int16"] = CHECK_PASS
        checks["check_geometry_preserved"] = CHECK_PASS
        checks["check_values_preserved"] = CHECK_PASS
        metadata.update({
            "prepared_image_size": metadata["source_image_size"],
            "prepared_orientation": source_orientation,
            "prepared_pixel_type": source_pixel_type,
            "prepared_spacing": source_spacing,
            "prepared_cta_size_bytes": source_size,
            "prepared_cta_sha256": source_sha256,
        })
        yield PreparedCTA(
            path=source_path,
            size=source_size,
            sha256=source_sha256,
            source_orientation=source_orientation,
            source_pixel_type=source_pixel_type,
            source_spacing=source_spacing,
            prepared_orientation=source_orientation,
            prepared_pixel_type=source_pixel_type,
            prepared_spacing=source_spacing,
            normalized=False,
            source_dimension=source_dimension,
            source_components=source_components,
            source_image_size=metadata["source_image_size"],
            prepared_image_size=metadata["prepared_image_size"],
            checks=_checks_as_tuple(checks),
            changes="PASSED UNCHANGED: already LPS signed int16; no resampling",
            message="All segmentation format requirements passed unchanged.",
        )
        return

    try:
        oriented_image = sitk.DICOMOrient(source_image, TARGET_CTA_ORIENTATION)
    except Exception as error:
        checks["check_output_lps"] = CHECK_FAIL
        raise CTAFormatRequirementError(
            "Could not convert orientation to LPS by axis permutation/flip: "
            "{0}".format(error),
            checks,
            metadata,
        ) from error
    prepared_image = sitk.Cast(oriented_image, sitk.sitkInt16)
    prepared_orientation = (
        sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            prepared_image.GetDirection()
        )
    )
    checks["check_output_lps"] = (
        CHECK_PASS
        if prepared_orientation == TARGET_CTA_ORIENTATION
        else CHECK_FAIL
    )
    checks["check_output_int16"] = (
        CHECK_PASS
        if prepared_image.GetPixelID() == sitk.sitkInt16
        else CHECK_FAIL
    )
    checks["check_geometry_preserved"] = (
        CHECK_PASS
        if _physical_geometry_preserved(
            source_image, prepared_image, numpy_module
        )
        else CHECK_FAIL
    )
    checks["check_values_preserved"] = (
        CHECK_PASS
        if _voxel_values_equal(
            oriented_image, prepared_image, numpy_module, sitk
        )
        else CHECK_FAIL
    )
    if any(
            checks[field] != CHECK_PASS
            for field in (
                "check_output_lps",
                "check_output_int16",
                "check_geometry_preserved",
                "check_values_preserved",
            )):
        raise CTAFormatRequirementError(
            "The prepared CTA failed an LPS/int16/geometry/value-preservation "
            "requirement.",
            checks,
            metadata,
        )

    temporary_directory = Path(temporary_directory)
    temporary_directory.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".CTA.normalized.",
        suffix=".nii.gz",
        dir=str(temporary_directory),
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        # Let SimpleITK create a complete NIfTI rather than opening the empty
        # reservation made by mkstemp on all supported library versions.
        temporary_path.unlink()
        try:
            sitk.WriteImage(prepared_image, str(temporary_path), True)
            verified_image = sitk.ReadImage(str(temporary_path))
        except Exception as error:
            raise CTAPreparationError(
                "Could not write/read the prepared CTA: {0}".format(error),
                checks,
                metadata,
            ) from error
        verified_orientation = (
            sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
                verified_image.GetDirection()
            )
        )
        if verified_orientation != TARGET_CTA_ORIENTATION:
            checks["check_output_lps"] = CHECK_FAIL
            raise CTAPreparationError(
                "Normalized CTA orientation verification failed: {0}".format(
                    temporary_path
                ),
                checks,
                metadata,
            )
        if verified_image.GetPixelID() != sitk.sitkInt16:
            checks["check_output_int16"] = CHECK_FAIL
            raise CTAPreparationError(
                "Normalized CTA pixel-type verification failed: {0}".format(
                    temporary_path
                ),
                checks,
                metadata,
            )
        if verified_image.GetSize() != prepared_image.GetSize():
            checks["check_geometry_preserved"] = CHECK_FAIL
            raise CTAPreparationError(
                "Normalized CTA size verification failed: {0}".format(
                    temporary_path
                ),
                checks,
                metadata,
            )
        checks["check_geometry_preserved"] = (
            CHECK_PASS
            if _physical_geometry_preserved(
                source_image, verified_image, numpy_module
            )
            else CHECK_FAIL
        )
        checks["check_values_preserved"] = (
            CHECK_PASS
            if _voxel_values_equal(
                prepared_image, verified_image, numpy_module, sitk
            )
            else CHECK_FAIL
        )
        if checks["check_geometry_preserved"] != CHECK_PASS:
            raise CTAPreparationError(
                "Prepared CTA physical geometry changed after writing.",
                checks,
                metadata,
            )
        if checks["check_values_preserved"] != CHECK_PASS:
            raise CTAPreparationError(
                "Prepared CTA voxel values changed after writing.",
                checks,
                metadata,
            )
        prepared_size = temporary_path.stat().st_size
        prepared_hash = sha256_file(temporary_path)
        prepared_spacing = _format_vector(verified_image.GetSpacing())
        prepared_image_size = _format_size(verified_image.GetSize())
        metadata.update({
            "prepared_image_size": prepared_image_size,
            "prepared_orientation": verified_orientation,
            "prepared_pixel_type": verified_image.GetPixelIDTypeAsString(),
            "prepared_spacing": prepared_spacing,
            "prepared_cta_size_bytes": prepared_size,
            "prepared_cta_sha256": prepared_hash,
        })
        changes = []
        if source_orientation != TARGET_CTA_ORIENTATION:
            changes.append(
                "orientation {0}->LPS by axis permutation/flip".format(
                    source_orientation
                )
            )
        if source_image.GetPixelID() != sitk.sitkInt16:
            changes.append(
                "decoded {0} HU->signed int16 direct HU".format(
                    source_pixel_type
                )
            )
        changes.append("no resampling")
        yield PreparedCTA(
            path=temporary_path,
            size=prepared_size,
            sha256=prepared_hash,
            source_orientation=source_orientation,
            source_pixel_type=source_pixel_type,
            source_spacing=source_spacing,
            prepared_orientation=verified_orientation,
            prepared_pixel_type=verified_image.GetPixelIDTypeAsString(),
            prepared_spacing=prepared_spacing,
            normalized=True,
            source_dimension=source_dimension,
            source_components=source_components,
            source_image_size=metadata["source_image_size"],
            prepared_image_size=prepared_image_size,
            checks=_checks_as_tuple(checks),
            changes="CHANGED: " + "; ".join(changes),
            message="All segmentation format requirements passed after conversion.",
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _validate_existing_targets(
    plans,
    output_folder,
    create_pipeline_folders,
    cta_destinations,
):
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
        ]
        if create_pipeline_folders:
            directory_targets.append((batch_path / "Table", "batch Table path"))
        selected_paths = dict(CTA_DESTINATION_RELATIVE_PATHS)
        for destination_key in cta_destinations:
            destination_parent = (plan.case_path / selected_paths[destination_key]).parent
            directory_targets.append(
                (destination_parent, "{0} CTA parent".format(destination_key))
            )
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

        destination_paths = [
            plan.case_path / selected_paths[destination_key]
            for destination_key in cta_destinations
        ]
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


def _copy_case_cta(
        plan,
        source_size,
        source_sha256,
        destination_keys,
        destination_relative_paths,
        normalize_for_segmentation,
        prepare_case=None):
    """Prepare once, copy to every requested destination, and return counts."""

    destination_results = {}
    copied_file_count = 0
    unchanged_file_count = 0
    with prepared_cta_source(
            plan.source_path,
            plan.case_path.parent,
            normalize_for_segmentation=normalize_for_segmentation,
            source_size=source_size,
            source_sha256=source_sha256) as prepared:
        try:
            if prepare_case is not None:
                prepare_case()
            for destination_key in destination_keys:
                destination_path = (
                    plan.case_path / destination_relative_paths[destination_key]
                )
                destination_hash, copied = _copy_and_verify(
                    prepared.path,
                    destination_path,
                    prepared.size,
                    prepared.sha256,
                )
                destination_results[destination_key] = (
                    destination_path,
                    destination_hash,
                )
                copied_file_count += int(copied)
                unchanged_file_count += int(not copied)
        except CTAFormatRequirementError:
            raise
        except Exception as error:
            checks = dict(prepared.checks)
            checks["check_destination_sha256"] = CHECK_FAIL
            metadata = {
                "source_dimension": prepared.source_dimension,
                "source_components": prepared.source_components,
                "source_image_size": prepared.source_image_size,
                "source_orientation": prepared.source_orientation,
                "source_pixel_type": prepared.source_pixel_type,
                "source_spacing": prepared.source_spacing,
                "prepared_image_size": prepared.prepared_image_size,
                "prepared_orientation": prepared.prepared_orientation,
                "prepared_pixel_type": prepared.prepared_pixel_type,
                "prepared_spacing": prepared.prepared_spacing,
                "prepared_cta_size_bytes": prepared.size,
                "prepared_cta_sha256": prepared.sha256,
                "format_changes": prepared.changes,
                "format_check_message": prepared.message,
            }
            raise CTAPreparationError(
                "Case destination preparation failed: {0}".format(error),
                checks,
                metadata,
            ) from error
        return (
            prepared,
            destination_results,
            copied_file_count,
            unchanged_file_count,
        )


def _write_manifest_atomic(manifest_path, rows, additional_fieldnames=()):
    """Publish a complete deterministic CSV without exposing a partial file."""

    manifest_path = Path(manifest_path)
    extra_fieldnames = []
    known_fieldnames = set(MANIFEST_FIELDS)
    for fieldname in additional_fieldnames:
        if (
            fieldname
            and fieldname not in known_fieldnames
            and fieldname not in extra_fieldnames
        ):
            extra_fieldnames.append(fieldname)
    fieldnames = MANIFEST_FIELDS + tuple(extra_fieldnames)
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
                fieldnames=fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary_path), str(manifest_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_format_audit_atomic(audit_path, rows):
    """Checkpoint the all-case format audit without exposing a partial CSV."""

    audit_path = Path(audit_path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{0}.".format(audit_path.name),
        suffix=".tmp",
        dir=str(audit_path.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
                file_descriptor, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=FORMAT_AUDIT_FIELDS,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary_path), str(audit_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _append_format_audit_row(audit_path, row):
    """Durably append one completed case record to the current audit."""

    with Path(audit_path).open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FORMAT_AUDIT_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writerow(row)
        csv_file.flush()
        os.fsync(csv_file.fileno())


def _read_manifest_state(manifest_path):
    """Return reusable row state and unknown columns from an existing manifest."""

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return {}, ()

    statuses = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        ordered_fieldnames = tuple(reader.fieldnames or ())
        fieldnames = set(ordered_fieldnames)
        if not {"batch_name", "case_id"}.issubset(fieldnames):
            return {}, ()
        additional_fieldnames = tuple(
            fieldname
            for fieldname in ordered_fieldnames
            if fieldname and fieldname not in MANIFEST_FIELDS
        )
        for row in reader:
            key = (
                row.get("batch_name", "").strip().casefold(),
                row.get("case_id", "").strip().casefold(),
            )
            if not all(key):
                continue
            status = {
                field: row.get(field, "")
                for field in PROCESSING_STATUS_FIELDS
                if field in fieldnames
            }
            status.update(
                {
                    field: row.get(field, "")
                    for field in additional_fieldnames
                }
            )
            status["_source_sha256"] = row.get("source_sha256", "")
            status["_prepared_cta_sha256"] = row.get(
                "prepared_cta_sha256", ""
            )
            statuses[key] = status
    return statuses, additional_fieldnames


def _read_segmentation_status(manifest_path):
    """Return previously recorded processing fields keyed by batch and case."""

    statuses, _ = _read_manifest_state(manifest_path)
    return {
        key: {
            field: value
            for field, value in status.items()
            if (
                field in PROCESSING_STATUS_FIELDS
                or field in ("_source_sha256", "_prepared_cta_sha256")
            )
        }
        for key, status in statuses.items()
    }


def _base_format_audit_row(plan):
    row = {field: "" for field in FORMAT_AUDIT_FIELDS}
    row.update({
        "audit_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rank": plan.rank,
        "source_file": plan.source_path.name,
        "source_path": str(plan.source_path),
        "case_id": plan.case_id,
        "batch_index": plan.batch_index,
        "batch_name": plan.batch_name,
        "batch_position": plan.batch_position,
        "case_relative_path": plan.case_relative_path,
        "segmentation_format_profile": SEGMENTATION_FORMAT_PROFILE,
        "segmentation_format_requirements": SEGMENTATION_FORMAT_REQUIREMENTS,
    })
    row.update(_new_format_checks())
    row["format_checks_performed"] = _format_checks_summary(row)
    return row


def _update_format_audit_row(
        row,
        status,
        checks=None,
        metadata=None,
        compatible="Unknown",
        changes="",
        message=""):
    checks = dict(checks or {})
    metadata = dict(metadata or {})
    for field in FORMAT_CHECK_FIELDS:
        if field in checks:
            row[field] = checks[field]
    for field in (
            "source_dimension",
            "source_components",
            "source_image_size",
            "prepared_image_size",
            "source_orientation",
            "source_pixel_type",
            "source_spacing",
            "prepared_orientation",
            "prepared_pixel_type",
            "prepared_spacing",
            "prepared_cta_size_bytes",
            "prepared_cta_sha256",
            "format_changes",
            "format_check_message"):
        if field in metadata and metadata[field] != "":
            row[field] = metadata[field]
    row.update({
        "status": status,
        "format_check_status": status,
        "segmentation_software_compatible": compatible,
        "format_changes": changes or row.get("format_changes", ""),
        "format_check_message": message or row.get("format_check_message", ""),
    })
    row["format_checks_performed"] = _format_checks_summary(row)
    return row


def _emit_preparation_progress(progress_callback, counts, plan=None, status="", message=""):
    if progress_callback is None:
        return
    event = dict(counts)
    event.update({
        "current_case": plan.case_id if plan is not None else "",
        "status": status,
        "message": message,
    })
    try:
        progress_callback(event)
    except Exception as error:
        print("Preparation progress callback failed: {0}".format(error))


def _audit_progress_message(row):
    compatible = row.get("segmentation_software_compatible", "Unknown")
    return (
        "profile compatible={0} | checks: {1} | action: {2} | {3}"
    ).format(
        compatible,
        row.get("format_checks_performed", ""),
        row.get("format_changes", "") or "none",
        row.get("format_check_message", ""),
    ).strip(" |")


def _prepare_case_worker(
        plan,
        create_pipeline_folders,
        batch_preproc,
        cases_per_batch,
        selected_destination_keys,
        destination_relative_paths,
        normalize_for_segmentation,
        previous_manifest_state,
        additional_manifest_fields,
        layout_lock):
    """Prepare one case without writing shared CSV or progress state."""

    audit_row = _base_format_audit_row(plan)
    marker_created = False
    layout_prepared = False
    source_size = ""
    source_hash = ""
    copied_file_count = 0
    unchanged_file_count = 0
    preserved_unselected_file_count = 0
    normalized_case_count = 0
    manifest_row = None
    pending_error = None

    try:
        source_size = plan.source_path.stat().st_size
        source_hash = sha256_file(plan.source_path)
        audit_row["source_size_bytes"] = source_size
        audit_row["source_sha256"] = source_hash

        def prepare_case_layout():
            nonlocal marker_created, layout_prepared
            plan.case_path.mkdir(parents=True, exist_ok=True)
            marker_path = plan.case_path / "0.dcm"
            if not marker_path.exists():
                marker_path.touch(exist_ok=False)
                marker_created = True
            if create_pipeline_folders:
                # BatchPreproc stores TableDir on the shared controller.  Keep
                # its state assignment and folder creation as one atomic step.
                with layout_lock:
                    batch_preproc.TableDir = str(
                        plan.case_path.parent / "Table"
                    )
                    batch_preproc.CreateFolders(CasePath=str(plan.case_path))
                    layout_prepared = True

        (
            prepared_cta,
            destination_results,
            copied_file_count,
            unchanged_file_count,
        ) = _copy_case_cta(
            plan,
            source_size,
            source_hash,
            selected_destination_keys,
            destination_relative_paths,
            normalize_for_segmentation,
            prepare_case=prepare_case_layout,
        )
        normalized_case_count = int(prepared_cta.normalized)

        for destination_key, relative_path in CTA_DESTINATION_RELATIVE_PATHS:
            if destination_key not in destination_results and (
                    plan.case_path / relative_path).is_file():
                preserved_unselected_file_count += 1

        canonical_result = destination_results.get("canonical")
        canonical_verified = bool(
            canonical_result and canonical_result[1] == prepared_cta.sha256
        )
        selected_pipeline_keys = tuple(
            key for key in selected_destination_keys if key != "canonical"
        )
        pipeline_verified = bool(selected_pipeline_keys) and all(
            destination_results[key][1] == prepared_cta.sha256
            for key in selected_pipeline_keys
        )
        all_verified = all(
            destination_results[key][1] == prepared_cta.sha256
            for key in selected_destination_keys
        )
        prepared_checks = dict(prepared_cta.checks)
        prepared_checks["check_destination_sha256"] = (
            CHECK_PASS if all_verified else CHECK_FAIL
        )
        prepared_metadata = {
            "source_dimension": prepared_cta.source_dimension,
            "source_components": prepared_cta.source_components,
            "source_image_size": prepared_cta.source_image_size,
            "source_orientation": prepared_cta.source_orientation,
            "source_pixel_type": prepared_cta.source_pixel_type,
            "source_spacing": prepared_cta.source_spacing,
            "prepared_image_size": prepared_cta.prepared_image_size,
            "prepared_orientation": prepared_cta.prepared_orientation,
            "prepared_pixel_type": prepared_cta.prepared_pixel_type,
            "prepared_spacing": prepared_cta.prepared_spacing,
            "prepared_cta_size_bytes": prepared_cta.size,
            "prepared_cta_sha256": prepared_cta.sha256,
        }
        if not all_verified:
            raise CTAPreparationError(
                "CTA destination SHA-256 verification failed.",
                prepared_checks,
                prepared_metadata,
            )

        if not normalize_for_segmentation:
            case_status = "copied_unchecked"
            compatible = "Not checked"
        elif prepared_cta.normalized:
            case_status = "passed_changed"
            compatible = True
        else:
            case_status = "passed_unchanged"
            compatible = True
        _update_format_audit_row(
            audit_row,
            case_status,
            checks=prepared_checks,
            metadata=prepared_metadata,
            compatible=compatible,
            changes=prepared_cta.changes,
            message=prepared_cta.message,
        )

        canonical_path = canonical_result[0] if canonical_result else None
        canonical_hash = canonical_result[1] if canonical_result else ""
        manifest_row = {
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
            "normalize_for_segmentation": bool(normalize_for_segmentation),
            "source_orientation": prepared_cta.source_orientation,
            "source_pixel_type": prepared_cta.source_pixel_type,
            "source_spacing": prepared_cta.source_spacing,
            "prepared_orientation": prepared_cta.prepared_orientation,
            "prepared_pixel_type": prepared_cta.prepared_pixel_type,
            "prepared_spacing": prepared_cta.prepared_spacing,
            "prepared_cta_size_bytes": prepared_cta.size,
            "prepared_cta_sha256": prepared_cta.sha256,
            "canonical_cta_path": (
                str(canonical_path.resolve())
                if canonical_path is not None else ""
            ),
            "canonical_cta_size_bytes": (
                canonical_path.stat().st_size
                if canonical_path is not None else ""
            ),
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
            "pipeline_copies_requested": bool(selected_pipeline_keys),
            "pipeline_copies_verified": pipeline_verified,
            "all_verified": all_verified,
        }
        manifest_row.update({
            field: audit_row.get(field, "")
            for field in FORMAT_RESULT_FIELDS
        })
        processing_status = dict(PROCESSING_STATUS_DEFAULTS)
        previous_status = previous_manifest_state.get(
            (plan.batch_name.casefold(), plan.case_id.casefold()),
            {},
        )
        previous_prepared_hash = previous_status.get(
            "_prepared_cta_sha256", ""
        )
        if (
            previous_status.get("_source_sha256", "") == source_hash
            and (
                previous_prepared_hash == prepared_cta.sha256
                or (
                    not normalize_for_segmentation
                    and not previous_prepared_hash
                    and source_hash == prepared_cta.sha256
                )
            )
        ):
            processing_status.update({
                field: previous_status[field]
                for field in PROCESSING_STATUS_FIELDS
                if field in previous_status
            })
        manifest_row.update({
            field: previous_status.get(field, "")
            for field in additional_manifest_fields
        })
        manifest_row.update(processing_status)
        for destination_key, _ in PIPELINE_CTA_RELATIVE_PATHS:
            if destination_key in destination_results:
                destination_path, destination_hash = destination_results[
                    destination_key
                ]
                manifest_row["{0}_cta_path".format(destination_key)] = str(
                    destination_path.resolve()
                )
                manifest_row[
                    "{0}_cta_sha256".format(destination_key)
                ] = destination_hash
    except CTAFormatRequirementError as error:
        pending_error = error
        _update_format_audit_row(
            audit_row,
            "excluded",
            checks=error.checks,
            metadata=error.metadata,
            compatible=False,
            changes="EXCLUDED: no downstream CTA was prepared",
            message=str(error),
        )
    except Exception as error:
        pending_error = error
        error_checks = getattr(error, "checks", {})
        error_metadata = getattr(error, "metadata", {})
        prior_changes = error_metadata.get("format_changes", "")
        failure_changes = "FAILED: operational preparation error"
        if prior_changes:
            failure_changes = prior_changes + "; " + failure_changes
        _update_format_audit_row(
            audit_row,
            "failed",
            checks=error_checks,
            metadata=error_metadata,
            compatible="Unknown",
            changes=failure_changes,
            message="{0}: {1}".format(type(error).__name__, error),
        )
        if marker_created:
            marker_path = plan.case_path / "0.dcm"
            if marker_path.is_file():
                try:
                    marker_path.unlink()
                except OSError as cleanup_error:
                    audit_row["format_check_message"] += (
                        " Marker cleanup also failed: {0}".format(
                            cleanup_error
                        )
                    )

    return CasePreparationOutcome(
        plan=plan,
        audit_row=audit_row,
        manifest_row=manifest_row,
        error=pending_error,
        copied_file_count=copied_file_count,
        unchanged_file_count=unchanged_file_count,
        preserved_unselected_file_count=preserved_unselected_file_count,
        normalized_case_count=normalized_case_count,
        layout_prepared=layout_prepared,
    )


def prepare_batch_data(
    input_folder,
    output_folder,
    create_pipeline_folders=True,
    batch_preproc=None,
    cases_per_batch=DEFAULT_CASES_PER_BATCH,
    cta_destinations=None,
    normalize_for_segmentation=True,
    case_limit=DEFAULT_CASE_LIMIT,
    progress_callback=None,
    continue_on_case_error=False,
    parallel_workers=DEFAULT_PARALLEL_WORKERS,
):
    """Prepare, checksum, and record all cases.

    ``batch_preproc`` must expose the existing ``CreateFolders(CasePath=...)``
    method when ``create_pipeline_folders`` is true.

    When ``normalize_for_segmentation`` is true, every readable 3-D CTA is
    losslessly reoriented to LPS and stored as signed int16 HU before it is
    copied to the selected destinations.  Set it false only for a downstream
    tool that explicitly supports the source NIfTI representation.

    ``progress_callback`` receives one dictionary before processing and after
    every case. With ``continue_on_case_error=True``, known format violations
    are excluded, operational errors are failed, and later cases continue.
    All selected cases are recorded in the format-audit CSV; only successful
    cases enter the downstream preparation manifest.

    ``parallel_workers`` runs independent cases concurrently (1-8). Shared
    BatchPreproc layout state and all CSV/progress updates remain serialized.
    Fail-fast runs stay single-threaded so no unrecorded case can finish after
    the first error.
    """

    cases_per_batch = _validate_cases_per_batch(cases_per_batch)
    case_limit = _validate_case_limit(case_limit)
    parallel_workers = _validate_parallel_workers(parallel_workers)
    selected_destination_keys = normalize_cta_destinations(
        cta_destinations,
        create_pipeline_folders,
    )
    destination_relative_paths = dict(CTA_DESTINATION_RELATIVE_PATHS)
    plans = build_case_plan(
        input_folder,
        output_folder,
        cases_per_batch=cases_per_batch,
        case_limit=case_limit,
    )
    if create_pipeline_folders:
        if batch_preproc is None or not callable(getattr(batch_preproc, "CreateFolders", None)):
            raise ValueError(
                "BatchCoronary CreateFolders is required when pipeline folders are requested."
            )
    sitk_module = None
    if normalize_for_segmentation:
        # Missing dependencies are a run-level configuration error, not a
        # property of one case, so fail before creating output.
        _, sitk_module = _load_normalization_dependencies()

    _validate_existing_case_layout(plans, output_folder, cases_per_batch)
    _validate_existing_targets(
        plans,
        output_folder,
        create_pipeline_folders,
        selected_destination_keys,
    )

    output_root = Path(output_folder).expanduser()
    data_root = output_root / OUTPUT_FOLDER_NAME
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / MANIFEST_FILE_NAME
    audit_path = data_root / FORMAT_AUDIT_FILE_NAME
    previous_manifest_state, additional_manifest_fields = _read_manifest_state(
        manifest_path
    )
    _write_format_audit_atomic(audit_path, [])

    rows = []
    copied_file_count = 0
    unchanged_file_count = 0
    preserved_unselected_file_count = 0
    normalized_case_count = 0
    counts = {
        "processed": 0,
        "total": len(plans),
        "passed": 0,
        "changed": 0,
        "unchanged": 0,
        "failed": 0,
        "excluded": 0,
    }
    effective_workers = min(parallel_workers, max(len(plans), 1))
    if not continue_on_case_error:
        # Preserve historical fail-fast behavior: no background case may
        # mutate output after the first error is reported.
        effective_workers = 1
    layout_lock = threading.Lock()
    layout_outcomes = []
    preparation_started_at = time.monotonic()

    _emit_preparation_progress(
        progress_callback,
        counts,
        status="starting",
        message=(
            "Checking every selected CTA against {0} with {1} case "
            "worker(s): {2}"
        ).format(
            SEGMENTATION_FORMAT_PROFILE,
            effective_workers,
            SEGMENTATION_FORMAT_REQUIREMENTS,
        ),
    )

    def prepare_one(plan):
        return _prepare_case_worker(
            plan=plan,
            create_pipeline_folders=create_pipeline_folders,
            batch_preproc=batch_preproc,
            cases_per_batch=cases_per_batch,
            selected_destination_keys=selected_destination_keys,
            destination_relative_paths=destination_relative_paths,
            normalize_for_segmentation=normalize_for_segmentation,
            previous_manifest_state=previous_manifest_state,
            additional_manifest_fields=additional_manifest_fields,
            layout_lock=layout_lock,
        )

    def record_outcome(outcome):
        nonlocal copied_file_count, unchanged_file_count
        nonlocal preserved_unselected_file_count, normalized_case_count

        audit_row = outcome.audit_row
        status = audit_row["status"]
        audit_row["preparation_parallel_workers"] = effective_workers
        if outcome.manifest_row is not None:
            outcome.manifest_row[
                "preparation_parallel_workers"
            ] = effective_workers
            rows.append(outcome.manifest_row)
            counts["passed"] += 1
            if status == "passed_changed":
                counts["changed"] += 1
            else:
                counts["unchanged"] += 1
        elif status == "excluded":
            counts["excluded"] += 1
        else:
            counts["failed"] += 1

        copied_file_count += outcome.copied_file_count
        unchanged_file_count += outcome.unchanged_file_count
        preserved_unselected_file_count += (
            outcome.preserved_unselected_file_count
        )
        normalized_case_count += outcome.normalized_case_count
        if outcome.layout_prepared:
            layout_outcomes.append(outcome.plan)

        counts["processed"] += 1
        audit_row.update({
            "processed_so_far": counts["processed"],
            "selected_total": counts["total"],
            "passed_so_far": counts["passed"],
            "changed_so_far": counts["changed"],
            "unchanged_so_far": counts["unchanged"],
            "failed_so_far": counts["failed"],
            "excluded_so_far": counts["excluded"],
        })
        _append_format_audit_row(audit_path, audit_row)
        _emit_preparation_progress(
            progress_callback,
            counts,
            plan=outcome.plan,
            status=status,
            message=_audit_progress_message(audit_row),
        )
        if outcome.error is not None and not continue_on_case_error:
            raise outcome.error

    with _simpleitk_thread_limit(sitk_module, effective_workers):
        if effective_workers == 1:
            for plan in plans:
                record_outcome(prepare_one(plan))
        else:
            with ThreadPoolExecutor(
                    max_workers=effective_workers,
                    thread_name_prefix="BatchNII") as executor:
                futures = [
                    executor.submit(prepare_one, plan)
                    for plan in plans
                ]
                for future in as_completed(futures):
                    record_outcome(future.result())

    # Thread completion order must not change the downstream rank order or the
    # shared BatchPreproc state a serial run would leave behind.
    rows.sort(key=lambda row: int(row["rank"]))
    if create_pipeline_folders and layout_outcomes:
        last_layout_plan = max(layout_outcomes, key=lambda plan: plan.rank)
        with layout_lock:
            batch_preproc.TableDir = str(
                last_layout_plan.case_path.parent / "Table"
            )

    _write_manifest_atomic(
        manifest_path,
        rows,
        additional_fieldnames=additional_manifest_fields,
    )

    return PreparationResult(
        case_count=counts["passed"],
        batch_count=len({row["batch_name"] for row in rows}),
        copied_file_count=copied_file_count,
        unchanged_file_count=unchanged_file_count,
        manifest_path=manifest_path.resolve(),
        pipeline_ready=(
            create_pipeline_folders
            and "ori_data" in selected_destination_keys
            and counts["passed"] > 0
        ),
        cases_per_batch=cases_per_batch,
        cta_destinations=selected_destination_keys,
        preserved_unselected_file_count=preserved_unselected_file_count,
        normalized_case_count=normalized_case_count,
        total_case_count=counts["total"],
        processed_case_count=counts["processed"],
        passed_case_count=counts["passed"],
        changed_case_count=counts["changed"],
        unchanged_case_count=counts["unchanged"],
        failed_case_count=counts["failed"],
        excluded_case_count=counts["excluded"],
        audit_path=audit_path.resolve(),
        parallel_workers=effective_workers,
        elapsed_seconds=time.monotonic() - preparation_started_at,
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
            limit_cases_widget = getattr(
                self.ui, "LimitCases_CB_BNDP", None
            )
            case_limit_widget = getattr(
                self.ui, "CaseLimit_SB_BNDP", None
            )
            if limit_cases_widget is not None and case_limit_widget is not None:
                limit_cases_widget.toggled.connect(
                    case_limit_widget.setEnabled
                )
                case_limit_widget.setEnabled(limit_cases_widget.isChecked())

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

        prepare_button = getattr(self.ui, "Prepare_Btn_BNDP", None)
        if prepare_button is not None:
            prepare_button.setEnabled(False)
        try:
            return self.PrepareData()
        except Exception:
            return None
        finally:
            if prepare_button is not None:
                prepare_button.setEnabled(True)

    def PrepareData(
        self,
        input_folder=None,
        output_folder=None,
        create_folders=None,
        cases_per_batch=None,
        cta_destinations=None,
        normalize_for_segmentation=None,
        case_limit=None,
        parallel_workers=None,
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
        if case_limit is None:
            limit_cases_widget = (
                getattr(self.ui, "LimitCases_CB_BNDP", None)
                if self.ui is not None else None
            )
            case_limit_widget = (
                getattr(self.ui, "CaseLimit_SB_BNDP", None)
                if self.ui is not None else None
            )
            if (
                    limit_cases_widget is not None
                    and not limit_cases_widget.isChecked()):
                case_limit = 0
            else:
                case_limit = (
                    case_limit_widget.value()
                    if case_limit_widget is not None else DEFAULT_CASE_LIMIT
                )
        if cta_destinations is None and self.ui is not None:
            destination_widgets = [
                (destination_key, getattr(self.ui, widget_name, None))
                for destination_key, widget_name in CTA_DESTINATION_WIDGET_NAMES
            ]
            if any(widget is not None for _, widget in destination_widgets):
                cta_destinations = tuple(
                    destination_key
                    for destination_key, widget in destination_widgets
                    if widget is not None and widget.isChecked()
                )
            else:
                cta_destinations = DEFAULT_CTA_DESTINATIONS
        if normalize_for_segmentation is None:
            normalize_widget = (
                getattr(self.ui, "NormalizeForSegmentation_CB_BNDP", None)
                if self.ui is not None else None
            )
            normalize_for_segmentation = (
                normalize_widget.isChecked()
                if normalize_widget is not None else True
            )
        if parallel_workers is None:
            parallel_workers_widget = (
                getattr(self.ui, "ParallelWorkers_SB_BNDP", None)
                if self.ui is not None else None
            )
            parallel_workers = (
                parallel_workers_widget.value()
                if parallel_workers_widget is not None
                else DEFAULT_PARALLEL_WORKERS
            )

        try:
            result = prepare_batch_data(
                input_folder=input_folder,
                output_folder=output_folder,
                create_pipeline_folders=create_folders,
                batch_preproc=self.batch_preproc,
                cases_per_batch=cases_per_batch,
                cta_destinations=cta_destinations,
                normalize_for_segmentation=normalize_for_segmentation,
                case_limit=case_limit,
                progress_callback=self._update_preparation_progress,
                continue_on_case_error=True,
                parallel_workers=parallel_workers,
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

    def _update_preparation_progress(self, event):
        """Display and log one structured preparation progress event."""

        event = event or {}
        counts = {
            key: int(event.get(key, 0) or 0)
            for key in (
                "processed",
                "total",
                "passed",
                "changed",
                "unchanged",
                "failed",
                "excluded",
            )
        }
        summary = (
            "Processed {processed}/{total} | passed {passed} | "
            "changed {changed} | unchanged {unchanged} | failed {failed} | "
            "excluded {excluded}"
        ).format(**counts)
        current_case = str(event.get("current_case", "") or "").strip()
        status = str(event.get("status", "") or "").strip()
        message = str(event.get("message", "") or "").strip()
        details = [value for value in (current_case, status, message) if value]
        display = summary
        if details:
            display += " | " + " | ".join(details)

        if self.ui is not None:
            progress_label = getattr(self.ui, "Progress_Label_BNDP", None)
            if progress_label is not None:
                _set_widget_text(progress_label, display)
            progress_bar = getattr(self.ui, "ProgressBar_BNDP", None)
            if progress_bar is not None:
                maximum = max(counts["total"], 1)
                progress_bar.setRange(0, maximum)
                progress_bar.setValue(min(counts["processed"], maximum))
                progress_bar.setFormat("%v/%m cases (%p%)")

        self.UpdateMsgLog(display)
        if QApplication is not None and QApplication.instance() is not None:
            QApplication.processEvents()

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


def _print_cli_progress(event):
    event = event or {}
    processed = int(event.get("processed", 0) or 0)
    total = int(event.get("total", 0) or 0)
    print(
        "[{0}/{1}] {2} | {3} | passed={4} changed={5} unchanged={6} "
        "failed={7} excluded={8} | {9}".format(
            processed,
            total,
            event.get("current_case", "") or "batch",
            event.get("status", ""),
            int(event.get("passed", 0) or 0),
            int(event.get("changed", 0) or 0),
            int(event.get("unchanged", 0) or 0),
            int(event.get("failed", 0) or 0),
            int(event.get("excluded", 0) or 0),
            event.get("message", ""),
        )
    )


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
        "--case-limit",
        type=int,
        default=DEFAULT_CASE_LIMIT,
        help=(
            "Prepare only the first N deterministically sorted top-level "
            ".nii.gz cases; 0 prepares all cases (default)."
        ),
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=DEFAULT_PARALLEL_WORKERS,
        help=(
            "Prepare independent cases concurrently (1-{0}); default {1}. "
            "For a single mechanical disk, start with 2."
        ).format(MAX_PARALLEL_WORKERS, DEFAULT_PARALLEL_WORKERS),
    )
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Create only 0.dcm and CTA/CTA.nii.gz, without the pipeline folders.",
    )
    parser.add_argument(
        "--cta-destination",
        action="append",
        choices=CTA_DESTINATION_KEYS,
        help=(
            "CTA destination key; repeat for multiple destinations. "
            "Defaults to ori_data."
        ),
    )
    parser.add_argument(
        "--preserve-source-representation",
        action="store_true",
        help=(
            "Do not normalize CTA axes and stored HU values before "
            "segmentation. The default is lossless LPS int16 normalization."
        ),
    )
    parser.add_argument(
        "--stop-on-case-error",
        action="store_true",
        help=(
            "Stop after recording the first excluded/failed case. The default "
            "records the case and continues with the remaining inputs."
        ),
    )
    options = parser.parse_args(arguments)
    if options.canonical_only and options.cta_destination:
        parser.error("--canonical-only cannot be combined with --cta-destination.")
    create_pipeline_folders = not options.canonical_only
    cta_destinations = (
        ("canonical",)
        if options.canonical_only
        else tuple(options.cta_destination or DEFAULT_CTA_DESTINATIONS)
    )
    batch_preproc = (
        _load_standalone_batch_preproc() if create_pipeline_folders else None
    )
    result = prepare_batch_data(
        input_folder=options.input_folder,
        output_folder=options.output_folder,
        create_pipeline_folders=create_pipeline_folders,
        batch_preproc=batch_preproc,
        cases_per_batch=options.cases_per_batch,
        cta_destinations=cta_destinations,
        normalize_for_segmentation=not options.preserve_source_representation,
        case_limit=options.case_limit,
        progress_callback=_print_cli_progress,
        continue_on_case_error=not options.stop_on_case_error,
        parallel_workers=options.parallel_workers,
    )
    print(result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())

