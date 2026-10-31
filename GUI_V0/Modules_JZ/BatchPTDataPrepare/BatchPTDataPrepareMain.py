# -*- coding: utf-8 -*-
"""Prepare cross-sectional coronary CTA tensors for model training.

The conversion functions are intentionally independent of the Hymo GUI.  This
file can be imported as a library, run from the command line, or connected to
the Qt page through the BatchPTDataPrepare controller.

The reference ViT representation is reproduced without image normalisation or
reorientation: SimpleITK's z-y-x array is transposed to x-y-z, the central
16-by-16 pixels are sampled every fifth longitudinal slice, 196 tiles are
retained or zero padded, and the row-major 14-by-14 mosaic is copied into three
identical channels.
"""

from __future__ import print_function

import argparse
import csv
import errno
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import SimpleITK as sitk

try:
    from PySide2.QtCore import QObject, QThread, Signal, Slot
    from PySide2.QtWidgets import QFileDialog, QMessageBox
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

    class QObject(object):
        """Small import-time fallback used by the non-Qt CLI."""

        def __init__(self, *args, **kwargs):
            del args, kwargs

    class QThread(object):
        pass

    class _FallbackSignal(object):
        def connect(self, callback):
            del callback

        def emit(self, *args):
            del args

    def Signal(*args):
        del args
        return _FallbackSignal()

    def Slot(*args):
        del args

        def decorator(function):
            return function

        return decorator

    QFileDialog = None
    QMessageBox = None


DEFAULT_INPUT_FOLDER = (
    r"S:\04_JZ\FishAndChips_C2\0_Initial_Segmentation"
)
DEFAULT_OUTPUT_PARENT = r"S:\04_JZ\FishAndChips_C2"
DEFAULT_MANIFEST_PATH = (
    r"S:\04_JZ\FishAndChips_C2\0_Initial_Segmentation"
    r"\BatchNIIDataPrepare_manifest.csv"
)
VIT_OUTPUT_FOLDER = "Input_Xsection_PT"
PATCH_OUTPUT_FOLDER = "Input_Xsection_PT_48x48x196"
FALLBACK_MANIFEST_NAME = "BatchPTDataPrepare_manifest.csv"
ARTERIES = ("RCA", "LAD", "LCX")
FORMATS = ("vit", "patch")
LAYOUTS = ("auto", "pipeline", "prepared", "flat")
DEFAULT_WORKERS = 4
DEFAULT_QC_COUNT = 3
DEFAULT_CHECKPOINT_CASES = 25
LOCK_TIMEOUT_SECONDS = 120
LOCK_POLL_SECONDS = 0.1
MAX_ERROR_LENGTH = 4000
ATOMIC_REPLACE_ATTEMPTS = 10
VIT_SHAPE = (3, 224, 224)
PATCH_SHAPE = (48, 48, 196)

PIPELINE_FILE_ARTERIES = {
    "mrcactali.nii.gz": "RCA",
    "mladctali.nii.gz": "LAD",
    "lcxctali.nii.gz": "LCX",
}
ARTERY_ORDER = {artery: index for index, artery in enumerate(ARTERIES)}


def _utc_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _clean_error(error, limit=MAX_ERROR_LENGTH):
    text = " ".join(str(error or "").replace("\x00", "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)] + " ...[truncated]"


def _natural_key(value):
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def _require_torch():
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "PyTorch is required for PT preparation. Activate the hymo "
            "environment and install torch==1.13.1+cpu."
        )
    return torch


def _normalise_arteries(values):
    normalised = []
    for value in values or ():
        artery = str(value).strip().upper()
        if artery not in ARTERIES:
            raise ValueError(
                "Unsupported artery {0!r}; choose from {1}.".format(
                    value, ", ".join(ARTERIES)
                )
            )
        if artery not in normalised:
            normalised.append(artery)
    if not normalised:
        raise ValueError("Select at least one artery.")
    return tuple(normalised)


def _normalise_layout(value):
    layout = str(value or "auto").strip().casefold()
    layout = layout.split(" - ", 1)[0].strip()
    aliases = {
        "autodetect": "auto",
        "auto-detect": "auto",
        "hymo": "pipeline",
        "nested": "pipeline",
        "prepared_cases": "prepared",
        "prepared-case": "prepared",
        "case": "prepared",
    }
    layout = aliases.get(layout, layout)
    if layout not in LAYOUTS:
        raise ValueError(
            "Unsupported layout {0!r}; choose from {1}.".format(
                value, ", ".join(LAYOUTS)
            )
        )
    return layout


@dataclass(frozen=True)
class PTPrepareConfig:
    """Immutable settings captured for one conversion run."""

    input_folder: str
    output_parent: str
    manifest_path: str = ""
    layout: str = "auto"
    selected_arteries: tuple = ARTERIES
    write_vit: bool = True
    write_patch: bool = False
    max_workers: int = DEFAULT_WORKERS
    save_qc: bool = False
    qc_count: int = DEFAULT_QC_COUNT
    checkpoint_cases: int = DEFAULT_CHECKPOINT_CASES

    @property
    def selected_formats(self):
        formats = []
        if self.write_vit:
            formats.append("vit")
        if self.write_patch:
            formats.append("patch")
        return tuple(formats)

    @property
    def vit_output_folder(self):
        return str(Path(self.output_parent) / VIT_OUTPUT_FOLDER)

    @property
    def patch_output_folder(self):
        return str(Path(self.output_parent) / PATCH_OUTPUT_FOLDER)

    @property
    def fallback_manifest_path(self):
        return str(
            Path(self.output_parent)
            / VIT_OUTPUT_FOLDER
            / FALLBACK_MANIFEST_NAME
        )


PTPreparationConfig = PTPrepareConfig


@dataclass(frozen=True)
class PTInputRecord:
    """One unique source NIfTI and its intended output identity."""

    source_path: str
    case_id: str
    artery: str
    batch_name: str
    layout: str
    case_relative_path: str = ""

    @property
    def key(self):
        return (self.case_id.casefold(), self.artery)

    @property
    def case_key(self):
        return (self.batch_name.casefold(), self.case_id.casefold())


@dataclass(frozen=True)
class TensorSamplingInfo:
    sampled_count: int
    used_count: int
    padded_count: int
    truncated_count: int


@dataclass(frozen=True)
class PTFormatOutcome:
    format_name: str
    output_path: str
    state: str
    error: str = ""

    @property
    def complete(self):
        return self.state in ("saved", "skipped")


@dataclass(frozen=True)
class PTSourceOutcome:
    record: PTInputRecord
    source_size_bytes: int
    source_mtime_ns: int
    source_state: str
    sampling: object
    format_outcomes: tuple
    error: str = ""
    qc_error: str = ""


@dataclass(frozen=True)
class PTProgressSnapshot:
    total_source_files: int
    completed_source_files: int
    total_cases: int
    completed_cases: int
    saved_outputs: int
    skipped_outputs: int
    failed_outputs: int
    missing_outputs: int
    deferred_outputs: int
    active_workers: int
    elapsed_seconds: float
    eta_seconds: float
    percent: int


@dataclass(frozen=True)
class PTRunResult:
    input_layout: str
    source_count: int
    case_count: int
    saved_count: int
    skipped_count: int
    failed_count: int
    missing_count: int
    deferred_count: int
    source_outcomes: tuple
    manifest_path: str
    fallback_manifest_path: str
    log_path: str
    duration_seconds: float

    def summary(self):
        return (
            "PT preparation finished: {0} sources across {1} cases; "
            "saved {2}, skipped {3}, missing {4}, deferred {5}, failed {6}; "
            "elapsed {7:.1f}s."
        ).format(
            self.source_count,
            self.case_count,
            self.saved_count,
            self.skipped_count,
            self.missing_count,
            self.deferred_count,
            self.failed_count,
            self.duration_seconds,
        )


def _coerce_config(config):
    if isinstance(config, PTPrepareConfig):
        result = config
    elif isinstance(config, dict):
        result = PTPrepareConfig(**config)
    else:
        raise TypeError("config must be PTPrepareConfig or a dictionary")

    input_folder = str(Path(result.input_folder).expanduser())
    output_parent = str(Path(result.output_parent).expanduser())
    if not input_folder.strip():
        raise ValueError("Input folder is required.")
    if not output_parent.strip():
        raise ValueError("Output parent folder is required.")
    selected_arteries = _normalise_arteries(result.selected_arteries)
    layout = _normalise_layout(result.layout)
    max_workers = max(1, int(result.max_workers))
    qc_count = max(0, int(result.qc_count))
    checkpoint_cases = max(1, int(result.checkpoint_cases))
    if not result.write_vit and not result.write_patch:
        raise ValueError("Select at least one tensor format.")
    return PTPrepareConfig(
        input_folder=input_folder,
        output_parent=output_parent,
        manifest_path=str(result.manifest_path or "").strip(),
        layout=layout,
        selected_arteries=selected_arteries,
        write_vit=bool(result.write_vit),
        write_patch=bool(result.write_patch),
        max_workers=max_workers,
        save_qc=bool(result.save_qc),
        qc_count=qc_count,
        checkpoint_cases=checkpoint_cases,
    )


def parse_prepared_filename(file_name):
    """Parse case and artery from the right of a prepared NIfTI name."""

    name = str(file_name)
    if not name.casefold().endswith(".nii.gz"):
        return None
    stem = name[:-7]
    match = re.match(r"^(?P<case>.+)_(?P<artery>RCA|LAD|LCX)$", stem, re.I)
    if match is None:
        return None
    case_id = match.group("case")
    if not case_id:
        return None
    return case_id, match.group("artery").upper()


def _pipeline_case_locations(root):
    locations = []
    direct_xsection = root / "Preproc" / "Xsection"
    if direct_xsection.is_dir():
        batch_name = (
            root.parent.name
            if root.parent.name.casefold().startswith("batch_")
            else ""
        )
        locations.append((root, batch_name))

    for child in sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: _natural_key(path.name)):
        if child.name.casefold() == "table":
            continue
        if child.name.casefold().startswith("batch_"):
            for case_path in sorted(
                    (path for path in child.iterdir() if path.is_dir()),
                    key=lambda path: _natural_key(path.name)):
                if (case_path / "Preproc" / "Xsection").is_dir():
                    locations.append((case_path, child.name))
        elif (child / "Preproc" / "Xsection").is_dir():
            batch_name = (
                root.name if root.name.casefold().startswith("batch_") else ""
            )
            locations.append((child, batch_name))
    return locations


def _discover_pipeline(root):
    records = []
    for case_path, batch_name in _pipeline_case_locations(root):
        xsection_path = case_path / "Preproc" / "Xsection"
        for source_path in sorted(
                (path for path in xsection_path.iterdir() if path.is_file()),
                key=lambda path: _natural_key(path.name)):
            artery = PIPELINE_FILE_ARTERIES.get(source_path.name.casefold())
            if artery is None:
                continue
            try:
                case_relative = str(case_path.relative_to(root))
            except ValueError:
                case_relative = case_path.name
            records.append(
                PTInputRecord(
                    source_path=str(source_path),
                    case_id=case_path.name,
                    artery=artery,
                    batch_name=batch_name,
                    layout="pipeline",
                    case_relative_path=case_relative,
                )
            )
    return records


def _discover_prepared(root):
    records = []
    for case_path in sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: _natural_key(path.name)):
        if case_path.name.casefold() in ("table", "_qc", "logs"):
            continue
        for source_path in sorted(
                (path for path in case_path.iterdir() if path.is_file()),
                key=lambda path: _natural_key(path.name)):
            parsed = parse_prepared_filename(source_path.name)
            if parsed is None:
                continue
            parsed_case, artery = parsed
            if parsed_case.casefold() != case_path.name.casefold():
                raise ValueError(
                    "Prepared filename case ID does not match its parent "
                    "folder: {0}".format(source_path)
                )
            records.append(
                PTInputRecord(
                    source_path=str(source_path),
                    case_id=case_path.name,
                    artery=artery,
                    batch_name="",
                    layout="prepared",
                    case_relative_path=case_path.name,
                )
            )
    return records


def _discover_flat(root):
    records = []
    for source_path in sorted(
            (path for path in root.iterdir() if path.is_file()),
            key=lambda path: _natural_key(path.name)):
        parsed = parse_prepared_filename(source_path.name)
        if parsed is None:
            continue
        case_id, artery = parsed
        records.append(
            PTInputRecord(
                source_path=str(source_path),
                case_id=case_id,
                artery=artery,
                batch_name="",
                layout="flat",
                case_relative_path="",
            )
        )
    return records


def _discover_all_inputs(config):
    config = _coerce_config(config)
    root = Path(config.input_folder)
    if not root.is_dir():
        raise NotADirectoryError(
            "Input folder does not exist or is not a directory: {0}".format(
                root
            )
        )
    discoverers = {
        "pipeline": _discover_pipeline,
        "prepared": _discover_prepared,
        "flat": _discover_flat,
    }
    if config.layout == "auto":
        candidates = {}
        discovery_errors = {}
        for layout, discoverer in discoverers.items():
            try:
                records = discoverer(root)
            except ValueError as error:
                discovery_errors[layout] = error
                records = []
            if records:
                candidates[layout] = records
        if len(candidates) > 1:
            raise ValueError(
                "Ambiguous mixed input layouts detected ({0}). Select an "
                "explicit layout or separate the inputs.".format(
                    ", ".join(sorted(candidates))
                )
            )
        if not candidates:
            if discovery_errors:
                raise next(iter(discovery_errors.values()))
            raise ValueError(
                "No supported cross-sectional CTA NIfTIs were found under "
                "{0}.".format(root)
            )
        layout, records = next(iter(candidates.items()))
    else:
        layout = config.layout
        records = discoverers[layout](root)
        if not records:
            raise ValueError(
                "No {0} layout cross-sectional CTA NIfTIs were found under "
                "{1}.".format(layout, root)
            )

    seen = {}
    for record in records:
        key = record.key
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "Duplicate output identity {0}_{1}: {2} and {3}".format(
                    record.case_id,
                    record.artery,
                    previous.source_path,
                    record.source_path,
                )
            )
        seen[key] = record
    ordered = sorted(
        records,
        key=lambda item: (
            _natural_key(item.batch_name),
            _natural_key(item.case_id),
            ARTERY_ORDER[item.artery],
            _natural_key(item.source_path),
        ),
    )
    return layout, tuple(ordered)


def discover_xsection_inputs(config):
    """Discover selected sources while enforcing one unambiguous layout."""

    config = _coerce_config(config)
    _layout, records = _discover_all_inputs(config)
    selected = set(config.selected_arteries)
    return tuple(record for record in records if record.artery in selected)


def _array_to_vit_tensor(array_zyx):
    torch = _require_torch()
    array_zyx = np.asarray(array_zyx)
    if array_zyx.ndim != 3:
        raise ValueError(
            "Expected a three-dimensional NIfTI array, got shape {0}.".format(
                array_zyx.shape
            )
        )
    image_xyz = np.asarray(
        np.transpose(array_zyx, (2, 1, 0)), dtype=np.float32, order="C"
    )
    size_x, size_y, _size_z = image_xyz.shape
    if size_x < 16 or size_y < 16:
        raise ValueError(
            "Cross-section is smaller than 16x16: {0}.".format(
                image_xyz.shape
            )
        )
    start_x = size_x // 2 - 8
    start_y = size_y // 2 - 8
    sampled = image_xyz[
        start_x:start_x + 16,
        start_y:start_y + 16,
        ::5,
    ]
    tiles = np.transpose(sampled, (2, 0, 1))
    sampled_count = int(tiles.shape[0])
    used_count = min(sampled_count, 196)
    padded_count = max(0, 196 - sampled_count)
    truncated_count = max(0, sampled_count - 196)

    mosaic = np.zeros((224, 224), dtype=np.float32)
    for index in range(used_count):
        row, column = divmod(index, 14)
        mosaic[
            row * 16:(row + 1) * 16,
            column * 16:(column + 1) * 16,
        ] = tiles[index]
    three_channels = np.repeat(mosaic[np.newaxis, :, :], 3, axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(three_channels)).float()
    tensor = tensor.contiguous()
    sampling = TensorSamplingInfo(
        sampled_count=sampled_count,
        used_count=used_count,
        padded_count=padded_count,
        truncated_count=truncated_count,
    )
    return tensor, sampling


def _read_nifti_to_vit(path):
    image = sitk.ReadImage(str(path))
    array_zyx = sitk.GetArrayFromImage(image)
    return _array_to_vit_tensor(array_zyx)


def nifti_to_vit_tensor(path):
    """Convert one NIfTI to a contiguous float32 tensor of shape 3x224x224."""

    tensor, _sampling = _read_nifti_to_vit(path)
    return tensor


def vit_tensor_to_patch_volume(tensor):
    """Reconstruct the publication model's 48x48x196 neighbour volume."""

    torch = _require_torch()
    if not torch.is_tensor(tensor):
        raise TypeError("Expected a PyTorch tensor.")
    if tuple(tensor.shape) != VIT_SHAPE:
        raise ValueError(
            "Expected tensor shape {0}, got {1}.".format(
                VIT_SHAPE, tuple(tensor.shape)
            )
        )
    if tensor.dtype != torch.float32:
        raise ValueError("Expected a float32 reference tensor.")
    mosaic = tensor[0].contiguous()
    tiles = (
        mosaic.reshape(14, 16, 14, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(196, 16, 16)
    )
    output = torch.zeros(
        PATCH_SHAPE, dtype=torch.float32, device=tensor.device
    )
    for center_index in range(196):
        for neighbour_position, offset in enumerate(range(-4, 5)):
            source_index = center_index + offset
            if source_index < 0 or source_index >= 196:
                continue
            row, column = divmod(neighbour_position, 3)
            output[
                row * 16:(row + 1) * 16,
                column * 16:(column + 1) * 16,
                center_index,
            ] = tiles[source_index]
    return output.contiguous()


def _source_signature(path):
    stat_result = Path(path).stat()
    return int(stat_result.st_size), int(stat_result.st_mtime_ns)


def _output_path(config, record, format_name):
    if Path(record.case_id).name != record.case_id:
        raise ValueError("Unsafe case ID: {0!r}".format(record.case_id))
    file_name = "{0}_{1}.pt".format(record.case_id, record.artery)
    if format_name == "vit":
        return Path(config.vit_output_folder) / file_name
    if format_name == "patch":
        return Path(config.patch_output_folder) / file_name
    raise ValueError("Unknown tensor format: {0}".format(format_name))


def _expected_shape(format_name):
    if format_name == "vit":
        return VIT_SHAPE
    if format_name == "patch":
        return PATCH_SHAPE
    raise ValueError("Unknown tensor format: {0}".format(format_name))


def _load_tensor(path):
    torch = _require_torch()
    return torch.load(str(path), map_location="cpu")


def validate_pt_output(path, format_name):
    """Return true only for a bare, contiguous float32 tensor contract."""

    torch = _require_torch()
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        tensor = _load_tensor(path)
    except Exception:
        return False
    if not torch.is_tensor(tensor):
        return False
    if tuple(tensor.shape) != _expected_shape(format_name):
        return False
    if tensor.dtype != torch.float32 or not tensor.is_contiguous():
        return False
    if format_name == "vit":
        if not torch.equal(tensor[0], tensor[1]):
            return False
        if not torch.equal(tensor[0], tensor[2]):
            return False
    return True


def _replace_with_retry(source_path, destination_path):
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    retryable_errnos = {errno.EACCES, errno.EBUSY, errno.EPERM}
    retryable_winerrors = {5, 32, 33}
    delay = 0.05
    last_error = None
    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(str(source_path), str(destination_path))
            return
        except OSError as error:
            last_error = error
            retryable = (
                isinstance(error, PermissionError)
                or error.errno in retryable_errnos
                or getattr(error, "winerror", None) in retryable_winerrors
            )
            if not retryable or attempt + 1 >= ATOMIC_REPLACE_ATTEMPTS:
                break
            time.sleep(delay)
            delay = min(0.5, delay * 2)
    raise last_error


def _atomic_save_tensor(tensor, destination, format_name):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{0}.".format(destination.name),
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch = _require_torch()
        torch.save(tensor.contiguous().cpu(), str(temporary_path))
        if not validate_pt_output(temporary_path, format_name):
            raise IOError(
                "Temporary tensor failed validation: {0}".format(
                    temporary_path
                )
            )
        _replace_with_retry(temporary_path, destination)
        if not validate_pt_output(destination, format_name):
            raise IOError(
                "Published tensor failed validation: {0}".format(destination)
            )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _flush_and_sync(file_object):
    file_object.flush()
    os.fsync(file_object.fileno())


@contextmanager
def interprocess_file_lock(lock_path, timeout_seconds=LOCK_TIMEOUT_SECONDS):
    """Hold the same one-byte OS lock convention used by Hymo manifests."""

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+b")
    acquired = False
    try:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()
        deadline = time.monotonic() + float(timeout_seconds)
        while True:
            try:
                lock_file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(
                        lock_file.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                acquired = True
                break
            except (IOError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for file lock: {0}".format(
                            lock_path
                        )
                    )
                time.sleep(LOCK_POLL_SECONDS)
        yield
    finally:
        if acquired:
            try:
                lock_file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except (IOError, OSError):
                pass
        lock_file.close()


def _read_csv_rows(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return [], []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader.fieldnames or []), list(reader)


def _write_csv_atomic(csv_path, fieldnames, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{0}.".format(csv_path.name),
        suffix=".tmp",
        dir=str(csv_path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
                file_descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            _flush_and_sync(handle)
        _replace_with_retry(temporary_path, csv_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _update_identity(update):
    return (
        str(update.get("batch_name", "")).strip().casefold(),
        str(update.get("case_id", "")).strip().casefold(),
    )


def _find_manifest_row(rows, update):
    batch_key, case_key = _update_identity(update)
    exact = []
    same_case = []
    for row in rows:
        row_case = str(row.get("case_id", "")).strip().casefold()
        if not row_case or row_case != case_key:
            continue
        same_case.append(row)
        row_batch = str(row.get("batch_name", "")).strip().casefold()
        if row_batch == batch_key:
            exact.append(row)
    if len(exact) == 1:
        return exact[0]
    if len(same_case) == 1:
        return same_case[0]
    return None


def _merge_updates_into_csv(csv_path, updates, allow_append):
    """Merge fresh updates under a lock and return unmatched updates."""

    csv_path = Path(csv_path)
    updates = list(updates)
    if not updates:
        return []
    if not csv_path.is_file() and not allow_append:
        return updates
    with interprocess_file_lock(str(csv_path) + ".lock"):
        fieldnames, rows = _read_csv_rows(csv_path)
        if rows and "case_id" not in fieldnames:
            return updates
        if not fieldnames:
            fieldnames = ["case_id", "batch_name", "case_relative_path"]
        for required in ("case_id", "batch_name", "case_relative_path"):
            if required not in fieldnames:
                fieldnames.append(required)

        unmatched = []
        changed = False
        for update in updates:
            case_id = str(update.get("case_id", "")).strip()
            if not case_id:
                continue
            row = _find_manifest_row(rows, update)
            if row is None and allow_append:
                row = {field_name: "" for field_name in fieldnames}
                row["case_id"] = case_id
                row["batch_name"] = str(
                    update.get("batch_name", "")
                ).strip()
                row["case_relative_path"] = str(
                    update.get("case_relative_path", "")
                )
                rows.append(row)
            if row is None:
                unmatched.append(update)
                continue
            for field_name, value in update.items():
                if field_name not in fieldnames:
                    fieldnames.append(field_name)
                if field_name in (
                        "case_id", "batch_name", "case_relative_path"):
                    if not str(row.get(field_name, "")).strip():
                        row[field_name] = value
                else:
                    row[field_name] = value
            changed = True
        if changed:
            for row in rows:
                for field_name in fieldnames:
                    row.setdefault(field_name, "")
            _write_csv_atomic(csv_path, fieldnames, rows)
        return unmatched


def merge_pt_manifest_updates(manifest_path, fallback_path, updates):
    """Merge PT fields without adding unmatched rows to an upstream ledger."""

    updates = list(updates)
    manifest_path = Path(manifest_path) if manifest_path else None
    fallback_path = Path(fallback_path)
    if (
            manifest_path is not None
            and manifest_path.resolve() == fallback_path.resolve()):
        unmatched = _merge_updates_into_csv(
            fallback_path, updates, allow_append=True
        )
        return {
            "upstream_count": len(updates) - len(unmatched),
            "fallback_count": 0,
        }

    unmatched = updates
    if manifest_path is not None and manifest_path.is_file():
        unmatched = _merge_updates_into_csv(
            manifest_path, updates, allow_append=False
        )
    _merge_updates_into_csv(fallback_path, unmatched, allow_append=True)
    return {
        "upstream_count": len(updates) - len(unmatched),
        "fallback_count": len(unmatched),
    }


def _manifest_candidates(path):
    if not path:
        return []
    _fields, rows = _read_csv_rows(path)
    return rows


def _previous_manifest_rows(config, records):
    paths = []
    if config.manifest_path:
        paths.append(Path(config.manifest_path))
    fallback_path = Path(config.fallback_manifest_path)
    if not paths or all(
            path.resolve() != fallback_path.resolve() for path in paths):
        paths.append(fallback_path)
    row_sets = [_manifest_candidates(path) for path in paths]
    previous = {}
    for record in records:
        update = {
            "case_id": record.case_id,
            "batch_name": record.batch_name,
        }
        fallback_match = None
        for rows in row_sets:
            row = _find_manifest_row(rows, update)
            if row is None:
                continue
            prefix = "pt_{0}_".format(record.artery.casefold())
            if any(str(key).startswith(prefix) for key in row):
                fallback_match = row
                break
            if fallback_match is None:
                fallback_match = row
        if fallback_match is not None:
            previous[record.key] = fallback_match
    return previous


def _integer_cell(row, field_name, default=None):
    if row is None:
        return default
    value = str(row.get(field_name, "")).strip()
    if not value:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sampling_from_row(row, artery):
    prefix = "pt_{0}_".format(artery.casefold())
    values = [
        _integer_cell(row, prefix + "sampled_count"),
        _integer_cell(row, prefix + "used_count"),
        _integer_cell(row, prefix + "padded_count"),
        _integer_cell(row, prefix + "truncated_count"),
    ]
    if any(value is None for value in values):
        return None
    return TensorSamplingInfo(*values)


def _signature_matches_row(row, record, signature):
    if row is None:
        return False
    prefix = "pt_{0}_".format(record.artery.casefold())
    size = _integer_cell(row, prefix + "source_size_bytes")
    mtime = _integer_cell(row, prefix + "source_mtime_ns")
    return size == signature[0] and mtime == signature[1]


def _can_skip_output(
        destination, format_name, record, signature, previous_row):
    destination = Path(destination)
    if not destination.is_file():
        return False
    row_has_signature = False
    if previous_row is not None:
        prefix = "pt_{0}_".format(record.artery.casefold())
        row_has_signature = (
            _integer_cell(
                previous_row, prefix + "source_size_bytes"
            ) is not None
            or _integer_cell(
                previous_row, prefix + "source_mtime_ns"
            ) is not None
        )
    if row_has_signature:
        unchanged = _signature_matches_row(
            previous_row, record, signature
        )
    else:
        unchanged = destination.stat().st_mtime_ns >= signature[1]
    return unchanged and validate_pt_output(destination, format_name)


def _scale_preview(array):
    array = np.asarray(array, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(finite, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    scaled[~np.isfinite(scaled)] = 0.0
    return np.asarray(np.rint(scaled * 255.0), dtype=np.uint8)


def _write_png(array, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(_scale_preview(array))
    sitk.WriteImage(image, str(output_path))


def _qc_paths(config, record):
    stem = "{0}_{1}".format(record.case_id, record.artery)
    return {
        "vit": (
            Path(config.vit_output_folder)
            / "_qc"
            / (stem + "_mosaic.png")
        ),
        "patch": (
            Path(config.patch_output_folder)
            / "_qc"
            / (stem + "_patch_contact_sheet.png")
        ),
    }


def _save_qc_previews(config, record, vit_tensor, patch_tensor):
    paths = _qc_paths(config, record)
    if config.write_vit:
        _write_png(vit_tensor[0].detach().cpu().numpy(), paths["vit"])
    if config.write_patch:
        if patch_tensor is None:
            patch_tensor = vit_tensor_to_patch_volume(vit_tensor)
        indices = np.linspace(0, 195, 6).astype(int)
        planes = [
            patch_tensor[:, :, int(index)].detach().cpu().numpy()
            for index in indices
        ]
        gap = np.zeros((48, 2), dtype=np.float32)
        top = np.concatenate(
            (planes[0], gap, planes[1], gap, planes[2]), axis=1
        )
        bottom = np.concatenate(
            (planes[3], gap, planes[4], gap, planes[5]), axis=1
        )
        row_gap = np.zeros((2, top.shape[1]), dtype=np.float32)
        contact_sheet = np.concatenate((top, row_gap, bottom), axis=0)
        _write_png(contact_sheet, paths["patch"])


def _qc_is_needed(config, record):
    if not config.save_qc:
        return False
    paths = _qc_paths(config, record)
    for format_name in config.selected_formats:
        path = paths[format_name]
        if not path.is_file() or path.stat().st_size <= 0:
            return True
    return False


def _deferred_outcomes(config, record, message):
    return tuple(
        PTFormatOutcome(
            format_name=format_name,
            output_path=str(_output_path(config, record, format_name)),
            state="deferred",
            error=message,
        )
        for format_name in config.selected_formats
    )


def _failed_outcomes(config, record, message):
    return tuple(
        PTFormatOutcome(
            format_name=format_name,
            output_path=str(_output_path(config, record, format_name)),
            state="failed",
            error=message,
        )
        for format_name in config.selected_formats
    )


def _process_record(
        record, config, previous_row=None, save_qc_for_record=False):
    source_path = Path(record.source_path)
    try:
        signature_before = _source_signature(source_path)
    except Exception as error:
        message = _clean_error(error)
        return PTSourceOutcome(
            record=record,
            source_size_bytes=0,
            source_mtime_ns=0,
            source_state="failed",
            sampling=None,
            format_outcomes=_failed_outcomes(config, record, message),
            error=message,
        )

    skip_by_format = {}
    for format_name in config.selected_formats:
        destination = _output_path(config, record, format_name)
        try:
            skip_by_format[format_name] = _can_skip_output(
                destination,
                format_name,
                record,
                signature_before,
                previous_row,
            )
        except Exception:
            skip_by_format[format_name] = False

    sampling = _sampling_from_row(previous_row, record.artery)
    qc_needed = save_qc_for_record and _qc_is_needed(config, record)
    needs_read = (
        not all(skip_by_format.values())
        or sampling is None
        or qc_needed
    )
    if not needs_read:
        signature_after = _source_signature(source_path)
        if signature_after != signature_before:
            message = "Source changed while outputs were validated; retry later."
            return PTSourceOutcome(
                record=record,
                source_size_bytes=signature_after[0],
                source_mtime_ns=signature_after[1],
                source_state="deferred",
                sampling=sampling,
                format_outcomes=_deferred_outcomes(
                    config, record, message
                ),
                error=message,
            )
        outcomes = tuple(
            PTFormatOutcome(
                format_name=format_name,
                output_path=str(
                    _output_path(config, record, format_name)
                ),
                state="skipped",
            )
            for format_name in config.selected_formats
        )
        return PTSourceOutcome(
            record=record,
            source_size_bytes=signature_before[0],
            source_mtime_ns=signature_before[1],
            source_state="skipped",
            sampling=sampling,
            format_outcomes=outcomes,
        )

    try:
        vit_tensor, sampling = _read_nifti_to_vit(source_path)
        signature_after = _source_signature(source_path)
    except Exception as error:
        message = _clean_error(error)
        return PTSourceOutcome(
            record=record,
            source_size_bytes=signature_before[0],
            source_mtime_ns=signature_before[1],
            source_state="failed",
            sampling=None,
            format_outcomes=_failed_outcomes(config, record, message),
            error=message,
        )
    if signature_after != signature_before:
        message = "Source changed during NIfTI read; no output was published."
        return PTSourceOutcome(
            record=record,
            source_size_bytes=signature_after[0],
            source_mtime_ns=signature_after[1],
            source_state="deferred",
            sampling=sampling,
            format_outcomes=_deferred_outcomes(config, record, message),
            error=message,
        )

    patch_tensor = None
    outcomes = []
    for format_name in config.selected_formats:
        destination = _output_path(config, record, format_name)
        if skip_by_format[format_name]:
            outcomes.append(
                PTFormatOutcome(
                    format_name=format_name,
                    output_path=str(destination),
                    state="skipped",
                )
            )
            continue
        try:
            if format_name == "vit":
                output_tensor = vit_tensor
            else:
                if patch_tensor is None:
                    patch_tensor = vit_tensor_to_patch_volume(vit_tensor)
                output_tensor = patch_tensor
            _atomic_save_tensor(
                output_tensor, destination, format_name
            )
            outcomes.append(
                PTFormatOutcome(
                    format_name=format_name,
                    output_path=str(destination),
                    state="saved",
                )
            )
        except Exception as error:
            outcomes.append(
                PTFormatOutcome(
                    format_name=format_name,
                    output_path=str(destination),
                    state="failed",
                    error=_clean_error(error),
                )
            )

    qc_error = ""
    if save_qc_for_record:
        try:
            _save_qc_previews(
                config, record, vit_tensor, patch_tensor
            )
        except Exception as error:
            qc_error = _clean_error(error)

    states = [outcome.state for outcome in outcomes]
    if "failed" in states:
        source_state = "failed"
    elif "saved" in states:
        source_state = "saved"
    else:
        source_state = "skipped"
    errors = [
        outcome.error for outcome in outcomes if outcome.error
    ]
    return PTSourceOutcome(
        record=record,
        source_size_bytes=signature_before[0],
        source_mtime_ns=signature_before[1],
        source_state=source_state,
        sampling=sampling,
        format_outcomes=tuple(outcomes),
        error=_clean_error("; ".join(errors)),
        qc_error=qc_error,
    )


class _RunLog(object):
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open(
            "a", encoding="utf-8", newline="\n"
        )
        self._lock = threading.Lock()

    def write(self, message):
        line = "[{0}] {1}".format(_utc_now(), str(message).rstrip())
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self):
        with self._lock:
            if not self._handle.closed:
                self._handle.close()


def _software_versions():
    torch = _require_torch()
    try:
        sitk_version = sitk.Version_VersionString()
    except AttributeError:
        sitk_version = str(sitk.Version())
    return json.dumps(
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "SimpleITK": sitk_version,
            "torch": torch.__version__,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _case_inventory(records):
    inventory = {}
    for record in records:
        case = inventory.setdefault(
            record.case_key,
            {
                "case_id": record.case_id,
                "batch_name": record.batch_name,
                "case_relative_path": record.case_relative_path,
                "records": {},
            },
        )
        case["records"][record.artery] = record
    return inventory


def _format_outcome_fields(update, artery, outcome):
    prefix = "pt_{0}_{1}_".format(
        outcome.format_name, artery.casefold()
    )
    update[prefix + "output_path"] = outcome.output_path
    update[prefix + "complete"] = int(outcome.complete)
    update[prefix + "failed"] = int(outcome.state == "failed")
    update[prefix + "outcome"] = outcome.state
    update[prefix + "error"] = outcome.error


def _case_manifest_update(
        case_data,
        outcomes_by_key,
        config,
        layout,
        run_id,
        started_utc,
        worker_count,
        software_versions):
    update = {
        "case_id": case_data["case_id"],
        "batch_name": case_data["batch_name"],
        "case_relative_path": case_data["case_relative_path"],
        "pt_run_id": run_id,
        "pt_run_started_utc": started_utc,
        "pt_updated_utc": _utc_now(),
        "pt_layout": layout,
        "pt_selected_arteries": ",".join(config.selected_arteries),
        "pt_selected_formats": ",".join(config.selected_formats),
        "pt_worker_count": worker_count,
        "pt_software_versions": software_versions,
    }
    case_states = []
    case_errors = []
    for artery in config.selected_arteries:
        artery_prefix = "pt_{0}_".format(artery.casefold())
        record = case_data["records"].get(artery)
        if record is None:
            update[artery_prefix + "source_path"] = ""
            update[artery_prefix + "source_size_bytes"] = ""
            update[artery_prefix + "source_mtime_ns"] = ""
            update[artery_prefix + "source_state"] = "missing"
            for count_name in (
                    "sampled_count",
                    "used_count",
                    "padded_count",
                    "truncated_count"):
                update[artery_prefix + count_name] = ""
            for format_name in config.selected_formats:
                missing_outcome = PTFormatOutcome(
                    format_name=format_name,
                    output_path=str(
                        _output_path(
                            config,
                            PTInputRecord(
                                source_path="",
                                case_id=case_data["case_id"],
                                artery=artery,
                                batch_name=case_data["batch_name"],
                                layout=layout,
                            ),
                            format_name,
                        )
                    ),
                    state="missing",
                )
                _format_outcome_fields(
                    update, artery, missing_outcome
                )
            case_states.append("missing")
            continue

        outcome = outcomes_by_key.get(record.key)
        if outcome is None:
            continue
        update[artery_prefix + "source_path"] = record.source_path
        update[artery_prefix + "source_size_bytes"] = (
            outcome.source_size_bytes
        )
        update[artery_prefix + "source_mtime_ns"] = (
            outcome.source_mtime_ns
        )
        update[artery_prefix + "source_state"] = outcome.source_state
        if outcome.sampling is not None:
            update[artery_prefix + "sampled_count"] = (
                outcome.sampling.sampled_count
            )
            update[artery_prefix + "used_count"] = (
                outcome.sampling.used_count
            )
            update[artery_prefix + "padded_count"] = (
                outcome.sampling.padded_count
            )
            update[artery_prefix + "truncated_count"] = (
                outcome.sampling.truncated_count
            )
        for format_outcome in outcome.format_outcomes:
            _format_outcome_fields(
                update, artery, format_outcome
            )
        case_states.append(outcome.source_state)
        if outcome.error:
            case_errors.append(
                "{0}: {1}".format(artery, outcome.error)
            )
        if outcome.qc_error:
            case_errors.append(
                "{0} QC: {1}".format(artery, outcome.qc_error)
            )

    if "failed" in case_states:
        state = "failed"
    elif "deferred" in case_states:
        state = "deferred"
    elif "missing" in case_states:
        state = "complete_with_missing"
    elif "saved" in case_states:
        state = "complete"
    else:
        state = "skipped"
    update["pt_state"] = state
    update["pt_complete"] = int(state in (
        "complete", "complete_with_missing", "skipped"
    ))
    update["pt_error"] = _clean_error("; ".join(case_errors))
    return update


def _source_result_line(outcome):
    format_text = ", ".join(
        "{0}={1}".format(item.format_name, item.state)
        for item in outcome.format_outcomes
    )
    sampling_text = ""
    if outcome.sampling is not None:
        sampling_text = (
            " sampled={0} used={1} padded={2} truncated={3}"
        ).format(
            outcome.sampling.sampled_count,
            outcome.sampling.used_count,
            outcome.sampling.padded_count,
            outcome.sampling.truncated_count,
        )
    text = "{0}_{1}: {2}{3}".format(
        outcome.record.case_id,
        outcome.record.artery,
        format_text,
        sampling_text,
    )
    if outcome.error:
        text += " error=" + outcome.error
    if outcome.qc_error:
        text += " qc_error=" + outcome.qc_error
    return text


def _safe_callback(callback, value):
    if callback is None:
        return
    try:
        callback(value)
    except Exception:
        pass


def run_pt_preparation(
        config, progress_callback=None, log_callback=None):
    """Discover, convert, validate, checkpoint, and return one run result."""

    config = _coerce_config(config)
    _require_torch()
    output_parent = Path(config.output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)
    Path(config.vit_output_folder).mkdir(parents=True, exist_ok=True)
    if config.write_patch:
        Path(config.patch_output_folder).mkdir(
            parents=True, exist_ok=True
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = "BatchPTDataPrepare_" + timestamp
    log_path = (
        Path(config.vit_output_folder)
        / "logs"
        / (run_id + ".log")
    )
    run_log = _RunLog(log_path)
    started_monotonic = time.monotonic()
    started_utc = _utc_now()

    def emit_log(message):
        run_log.write(message)
        _safe_callback(log_callback, str(message))

    try:
        layout, all_records = _discover_all_inputs(config)
        selected_set = set(config.selected_arteries)
        records = tuple(
            record for record in all_records
            if record.artery in selected_set
        )
        cases = _case_inventory(all_records)
        worker_count = min(
            config.max_workers,
            max(1, len(records)),
            max(1, os.cpu_count() or 1),
        )
        software_versions = _software_versions()
        previous_rows = _previous_manifest_rows(config, records)
        qc_keys = set(
            record.key
            for record in records[:config.qc_count]
        ) if config.save_qc else set()

        missing_outputs = 0
        pending_by_case = {}
        for case_key, case_data in cases.items():
            present = case_data["records"]
            pending_by_case[case_key] = sum(
                1 for artery in config.selected_arteries
                if artery in present
            )
            missing_outputs += sum(
                len(config.selected_formats)
                for artery in config.selected_arteries
                if artery not in present
            )
        completed_case_keys = set(
            key for key, count in pending_by_case.items()
            if count == 0
        )
        outcomes_by_key = {}
        source_outcomes = []
        total_output_slots = (
            len(cases)
            * len(config.selected_arteries)
            * len(config.selected_formats)
        )
        saved_count = 0
        skipped_count = 0
        failed_count = 0
        deferred_count = 0
        completed_sources = 0
        previous_percent = 0
        last_checkpoint_count = 0

        emit_log(
            "Run {0}: layout={1}, cases={2}, sources={3}, workers={4}, "
            "arteries={5}, formats={6}".format(
                run_id,
                layout,
                len(cases),
                len(records),
                worker_count,
                ",".join(config.selected_arteries),
                ",".join(config.selected_formats),
            )
        )

        def progress_snapshot(active_workers, force_complete=False):
            nonlocal previous_percent
            done_outputs = (
                saved_count
                + skipped_count
                + failed_count
                + deferred_count
                + missing_outputs
            )
            elapsed = time.monotonic() - started_monotonic
            if force_complete:
                percent = 100
            elif total_output_slots:
                percent = int(
                    min(100, 100.0 * done_outputs / total_output_slots)
                )
            else:
                percent = 100
            percent = max(previous_percent, percent)
            previous_percent = percent
            if done_outputs > 0 and total_output_slots > done_outputs:
                eta = elapsed * (
                    total_output_slots - done_outputs
                ) / done_outputs
            else:
                eta = 0.0
            snapshot = PTProgressSnapshot(
                total_source_files=len(records),
                completed_source_files=completed_sources,
                total_cases=len(cases),
                completed_cases=len(completed_case_keys),
                saved_outputs=saved_count,
                skipped_outputs=skipped_count,
                failed_outputs=failed_count,
                missing_outputs=missing_outputs,
                deferred_outputs=deferred_count,
                active_workers=max(0, int(active_workers)),
                elapsed_seconds=round(elapsed, 3),
                eta_seconds=round(max(0.0, eta), 3),
                percent=percent,
            )
            _safe_callback(progress_callback, snapshot)

        def checkpoint(force=False):
            nonlocal last_checkpoint_count
            completed_count = len(completed_case_keys)
            if not force and (
                    completed_count - last_checkpoint_count
                    < config.checkpoint_cases):
                return
            updates = [
                _case_manifest_update(
                    cases[case_key],
                    outcomes_by_key,
                    config,
                    layout,
                    run_id,
                    started_utc,
                    worker_count,
                    software_versions,
                )
                for case_key in sorted(completed_case_keys)
            ]
            merge_counts = merge_pt_manifest_updates(
                config.manifest_path,
                config.fallback_manifest_path,
                updates,
            )
            last_checkpoint_count = completed_count
            emit_log(
                "Manifest checkpoint: cases={0}, upstream={1}, fallback={2}"
                .format(
                    completed_count,
                    merge_counts["upstream_count"],
                    merge_counts["fallback_count"],
                )
            )

        progress_snapshot(
            min(worker_count, len(records))
        )
        if completed_case_keys:
            checkpoint(force=(
                len(completed_case_keys) >= config.checkpoint_cases
            ))

        if records:
            with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="BatchPT") as executor:
                futures = {
                    executor.submit(
                        _process_record,
                        record,
                        config,
                        previous_rows.get(record.key),
                        record.key in qc_keys,
                    ): record
                    for record in records
                }
                for future in as_completed(futures):
                    record = futures[future]
                    try:
                        outcome = future.result()
                    except BaseException as error:
                        message = _clean_error(
                            "{0}: {1}".format(
                                type(error).__name__, error
                            )
                        )
                        outcome = PTSourceOutcome(
                            record=record,
                            source_size_bytes=0,
                            source_mtime_ns=0,
                            source_state="failed",
                            sampling=None,
                            format_outcomes=_failed_outcomes(
                                config, record, message
                            ),
                            error=message,
                        )
                    source_outcomes.append(outcome)
                    outcomes_by_key[record.key] = outcome
                    completed_sources += 1
                    for format_outcome in outcome.format_outcomes:
                        if format_outcome.state == "saved":
                            saved_count += 1
                        elif format_outcome.state == "skipped":
                            skipped_count += 1
                        elif format_outcome.state == "failed":
                            failed_count += 1
                        elif format_outcome.state == "deferred":
                            deferred_count += 1
                    pending_by_case[record.case_key] -= 1
                    if pending_by_case[record.case_key] == 0:
                        completed_case_keys.add(record.case_key)
                    line = _source_result_line(outcome)
                    emit_log(line)
                    active = min(
                        worker_count,
                        len(records) - completed_sources,
                    )
                    progress_snapshot(active)
                    checkpoint(force=False)

        checkpoint(force=True)
        progress_snapshot(0, force_complete=True)
        source_outcomes.sort(
            key=lambda item: (
                _natural_key(item.record.batch_name),
                _natural_key(item.record.case_id),
                ARTERY_ORDER[item.record.artery],
            )
        )
        duration = round(
            time.monotonic() - started_monotonic, 3
        )
        result = PTRunResult(
            input_layout=layout,
            source_count=len(records),
            case_count=len(cases),
            saved_count=saved_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
            missing_count=missing_outputs,
            deferred_count=deferred_count,
            source_outcomes=tuple(source_outcomes),
            manifest_path=config.manifest_path,
            fallback_manifest_path=config.fallback_manifest_path,
            log_path=str(log_path),
            duration_seconds=duration,
        )
        emit_log(result.summary())
        return result
    except BaseException as error:
        emit_log(
            "RUN FAILED: {0}: {1}".format(
                type(error).__name__, _clean_error(error)
            )
        )
        raise
    finally:
        run_log.close()


class PTPreparationCoordinator(QObject):
    """Qt worker object that owns no widgets."""

    progress = Signal(object)
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    @Slot()
    def run(self):
        try:
            result = run_pt_preparation(
                self.config,
                progress_callback=self.progress.emit,
                log_callback=self.log.emit,
            )
        except Exception as error:
            self.log.emit(
                "Coordinator error: {0}\n{1}".format(
                    error, traceback.format_exc()
                )
            )
            self.failed.emit(str(error))
            return
        self.finished.emit(result)


def _widget_text(widget):
    if widget is None:
        return ""
    for method_name in ("toPlainText", "text", "currentText"):
        method = getattr(widget, method_name, None)
        if callable(method):
            return str(method()).strip()
    return ""


def _widget_checked(widget, default=False):
    if widget is None:
        return bool(default)
    method = getattr(widget, "isChecked", None)
    if not callable(method):
        return bool(default)
    return bool(method())


def _widget_integer(widget, default):
    if widget is None:
        return int(default)
    method = getattr(widget, "value", None)
    if not callable(method):
        return int(default)
    try:
        return int(method())
    except (TypeError, ValueError):
        return int(default)


def _set_widget_text(widget, value):
    if widget is None:
        return
    for method_name in ("setPlainText", "setText"):
        method = getattr(widget, method_name, None)
        if callable(method):
            method(str(value))
            return


class BatchPTDataPrepare(QObject):
    """Thin Hymo page controller; conversion remains in the worker thread."""

    def __init__(self, UI=None, Hedys=None):
        super().__init__()
        self.ui = UI
        self.model = Hedys
        self.modelui = Hedys.ui if Hedys is not None else None
        self._run_thread = None
        self._run_coordinator = None
        self._last_run_result = None
        if self.ui is not None:
            self._connect_button(
                "ChooseInputFolder_Btn_BPTDP",
                self.ChooseInputFolder,
            )
            self._connect_button(
                "ChooseOutputFolder_Btn_BPTDP",
                self.ChooseOutputFolder,
            )
            self._connect_button(
                "ChooseManifest_Btn_BPTDP",
                self.ChooseManifest,
            )
            self._connect_button(
                "Run_Btn_BPTDP",
                self.startRun,
            )
            log_widget = getattr(self.ui, "Log_Txt_BPTDP", None)
            document_method = getattr(log_widget, "document", None)
            if callable(document_method):
                document_method().setMaximumBlockCount(500)
            if self.modelui is not None:
                self.modelui.QStackedWidget_Module.setCurrentWidget(
                    self.ui.centralwidget
                )

    def _connect_button(self, name, callback):
        button = getattr(self.ui, name, None)
        if button is not None:
            button.clicked.connect(callback)

    def _choose_directory(self, title, current):
        if QFileDialog is None:
            return ""
        return QFileDialog.getExistingDirectory(
            self.ui, title, current or ""
        )

    @Slot()
    def ChooseInputFolder(self):
        widget = getattr(
            self.ui, "InputFolderPath_Txt_BPTDP", None
        )
        selected = self._choose_directory(
            "Choose cross-sectional NIfTI folder",
            _widget_text(widget),
        )
        if selected:
            _set_widget_text(widget, selected)

    @Slot()
    def ChooseOutputFolder(self):
        widget = getattr(
            self.ui, "OutputFolderPath_Txt_BPTDP", None
        )
        selected = self._choose_directory(
            "Choose output parent folder",
            _widget_text(widget),
        )
        if selected:
            _set_widget_text(widget, selected)

    @Slot()
    def ChooseManifest(self):
        if QFileDialog is None:
            return
        widget = getattr(
            self.ui, "ManifestPath_Txt_BPTDP", None
        )
        selected, _filter = QFileDialog.getOpenFileName(
            self.ui,
            "Choose compatible manifest (optional)",
            _widget_text(widget),
            "CSV files (*.csv);;All files (*)",
        )
        if selected:
            _set_widget_text(widget, selected)

    def _config_from_ui(self):
        selected_arteries = tuple(
            artery
            for artery, widget_name in (
                ("RCA", "RCA_CB_BPTDP"),
                ("LAD", "LAD_CB_BPTDP"),
                ("LCX", "LCX_CB_BPTDP"),
            )
            if _widget_checked(
                getattr(self.ui, widget_name, None), True
            )
        )
        layout_widget = getattr(
            self.ui, "Layout_CB_BPTDP", None
        )
        layout_text = _widget_text(layout_widget)
        if not layout_text:
            layout_text = _widget_text(
                getattr(self.ui, "InputLayout_CB_BPTDP", None)
            )
        return _coerce_config(
            PTPrepareConfig(
                input_folder=_widget_text(
                    getattr(
                        self.ui,
                        "InputFolderPath_Txt_BPTDP",
                        None,
                    )
                ),
                output_parent=_widget_text(
                    getattr(
                        self.ui,
                        "OutputFolderPath_Txt_BPTDP",
                        None,
                    )
                ),
                manifest_path=_widget_text(
                    getattr(
                        self.ui,
                        "ManifestPath_Txt_BPTDP",
                        None,
                    )
                ),
                layout=layout_text or "auto",
                selected_arteries=selected_arteries,
                write_vit=_widget_checked(
                    getattr(
                        self.ui, "VITFormat_CB_BPTDP", None
                    ),
                    True,
                ),
                write_patch=_widget_checked(
                    getattr(
                        self.ui, "PatchFormat_CB_BPTDP", None
                    ),
                    False,
                ),
                max_workers=_widget_integer(
                    getattr(
                        self.ui, "Workers_SB_BPTDP", None
                    ),
                    DEFAULT_WORKERS,
                ),
                save_qc=_widget_checked(
                    getattr(
                        self.ui, "SaveQC_CB_BPTDP", None
                    ),
                    False,
                ),
                qc_count=_widget_integer(
                    getattr(
                        self.ui, "QCCount_SB_BPTDP", None
                    ),
                    DEFAULT_QC_COUNT,
                ),
            )
        )

    def mainFunction(self, config=None):
        if config is None:
            config = self._config_from_ui()
        return run_pt_preparation(config)

    @Slot()
    def startRun(self):
        if not QT_AVAILABLE:
            raise RuntimeError("PySide2 is required for the GUI controller.")
        if (
                self._run_thread is not None
                and self._run_thread.isRunning()):
            return
        try:
            config = self._config_from_ui()
        except Exception as error:
            self._on_failed(str(error))
            return
        self._set_run_enabled(False)
        self._set_progress("Discovering inputs...", 0)
        thread = QThread()
        coordinator = PTPreparationCoordinator(config)
        coordinator.moveToThread(thread)
        thread.started.connect(coordinator.run)
        coordinator.progress.connect(self._on_progress)
        coordinator.log.connect(self._append_log)
        coordinator.finished.connect(self._on_finished)
        coordinator.failed.connect(self._on_failed)
        coordinator.finished.connect(coordinator.deleteLater)
        coordinator.failed.connect(coordinator.deleteLater)
        coordinator.finished.connect(thread.quit)
        coordinator.failed.connect(thread.quit)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._run_thread = thread
        self._run_coordinator = coordinator
        thread.start()

    startPreparation = startRun

    def _set_run_enabled(self, enabled):
        if self.ui is None:
            return
        button = getattr(self.ui, "Run_Btn_BPTDP", None)
        if button is not None:
            button.setEnabled(bool(enabled))

    def _set_progress(self, text, percent=None):
        if self.ui is None:
            return
        label = getattr(
            self.ui, "Progress_Label_BPTDP", None
        )
        if label is None:
            label = getattr(
                self.ui, "ProgressLabel_BPTDP", None
            )
        progress_bar = getattr(
            self.ui, "Progress_PB_BPTDP", None
        )
        if progress_bar is None:
            progress_bar = getattr(
                self.ui, "ProgressBar_BPTDP", None
            )
        if label is not None:
            label.setText(str(text))
        if progress_bar is not None and percent is not None:
            progress_bar.setValue(
                max(0, min(100, int(percent)))
            )

    @staticmethod
    def _duration(seconds):
        seconds = max(0, int(round(float(seconds or 0))))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return "{0}h {1:02d}m".format(hours, minutes)
        if minutes:
            return "{0}m {1:02d}s".format(minutes, seconds)
        return "{0}s".format(seconds)

    @Slot(object)
    def _on_progress(self, snapshot):
        if not isinstance(snapshot, PTProgressSnapshot):
            return
        status = (
            "files {0}/{1} | cases {2}/{3} | saved {4}, skipped {5}, "
            "missing {6}, deferred {7}, failed {8} | active {9} | "
            "elapsed {10} | ETA {11}"
        ).format(
            snapshot.completed_source_files,
            snapshot.total_source_files,
            snapshot.completed_cases,
            snapshot.total_cases,
            snapshot.saved_outputs,
            snapshot.skipped_outputs,
            snapshot.missing_outputs,
            snapshot.deferred_outputs,
            snapshot.failed_outputs,
            snapshot.active_workers,
            self._duration(snapshot.elapsed_seconds),
            self._duration(snapshot.eta_seconds),
        )
        self._set_progress(status, snapshot.percent)

    @Slot(str)
    def _append_log(self, message):
        if self.ui is not None:
            widget = getattr(self.ui, "Log_Txt_BPTDP", None)
            if widget is not None:
                append_method = getattr(
                    widget, "appendPlainText", None
                )
                if not callable(append_method):
                    append_method = getattr(widget, "append", None)
                if callable(append_method):
                    append_method(str(message))
        if self.modelui is not None:
            for name in (
                    "plainTextEdit_Message",
                    "plainTextEdit_Log"):
                widget = getattr(self.modelui, name, None)
                method = getattr(widget, "appendPlainText", None)
                if callable(method):
                    method(str(message))

    @Slot(object)
    def _on_finished(self, result):
        self._last_run_result = result
        self._append_log(result.summary())
        self._set_progress(result.summary(), 100)

    @Slot(str)
    def _on_failed(self, message):
        message = str(message)
        self._append_log("PT preparation failed: " + message)
        self._set_progress("Failed: " + message, 0)
        self._set_run_enabled(True)
        if (
                QMessageBox is not None
                and self.ui is not None
                and self._run_thread is None):
            QMessageBox.critical(
                self.ui, "Batch PT Data Prepare", message
            )

    @Slot()
    def _on_thread_finished(self):
        self._run_thread = None
        self._run_coordinator = None
        self._set_run_enabled(True)


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert coronary CTA cross-section NIfTIs directly to "
            "training PT tensors."
        )
    )
    parser.add_argument(
        "--input-folder", default=DEFAULT_INPUT_FOLDER
    )
    parser.add_argument(
        "--output-parent", default=DEFAULT_OUTPUT_PARENT
    )
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST_PATH
    )
    parser.add_argument(
        "--layout",
        choices=LAYOUTS,
        default="auto",
    )
    parser.add_argument(
        "--arteries",
        nargs="+",
        choices=ARTERIES,
        default=list(ARTERIES),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=FORMATS,
        default=["vit"],
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS
    )
    parser.add_argument(
        "--save-qc", action="store_true"
    )
    parser.add_argument(
        "--qc-count", type=int, default=DEFAULT_QC_COUNT
    )
    return parser


def main(argv=None):
    arguments = _build_cli_parser().parse_args(argv)
    config = PTPrepareConfig(
        input_folder=arguments.input_folder,
        output_parent=arguments.output_parent,
        manifest_path=arguments.manifest,
        layout=arguments.layout,
        selected_arteries=tuple(arguments.arteries),
        write_vit="vit" in arguments.formats,
        write_patch="patch" in arguments.formats,
        max_workers=arguments.workers,
        save_qc=arguments.save_qc,
        qc_count=arguments.qc_count,
    )
    try:
        result = run_pt_preparation(
            config,
            progress_callback=lambda snapshot: print(
                "{0:3d}% files {1}/{2}, cases {3}/{4}, "
                "saved {5}, skipped {6}, failed {7}".format(
                    snapshot.percent,
                    snapshot.completed_source_files,
                    snapshot.total_source_files,
                    snapshot.completed_cases,
                    snapshot.total_cases,
                    snapshot.saved_outputs,
                    snapshot.skipped_outputs,
                    snapshot.failed_outputs,
                )
            ),
            log_callback=print,
        )
    except Exception as error:
        print("Batch PT Data Prepare failed: {0}".format(error), file=sys.stderr)
        return 1
    print(result.summary())
    return 0 if result.failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
