import csv
import errno
import hashlib
import locale
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import json
import numpy
import SimpleITK as sitk

try:
    import psutil
except ImportError:  # The pinned hymo environment includes psutil.
    psutil = None

os.chdir(os.getcwd())
sys.path.insert(0, '../Functions_JZ')
import Save_Load_File
import Preprocess_Mask
# Import functions self-written
from Modules_YL.BatchCoronary import BranchDictionary


XSECTION_ARTERIES = (
    ('LCx', 'LCxCTALI.nii.gz', 'LCxValNN.nii.gz'),
    ('mLAD', 'mLADCTALI.nii.gz', 'mLADValNN.nii.gz'),
    ('mRCA', 'mRCACTALI.nii.gz', 'mRCAValNN.nii.gz'),
)
XSECTION_ARTERY_NAMES = tuple(item[0] for item in XSECTION_ARTERIES)
XSECTION_OUTPUT_LOOKUP = {}
for _artery_name, _cta_name, _seg_name in XSECTION_ARTERIES:
    XSECTION_OUTPUT_LOOKUP[_cta_name.casefold()] = (_artery_name, 'CTA')
    XSECTION_OUTPUT_LOOKUP[_seg_name.casefold()] = (_artery_name, 'seg')

XSECTION_RECORD_FOLDER = 'xsection_record'
XSECTION_RECORD_FILE = 'batch_xsection.csv'
DEFAULT_MANIFEST_FILE_NAME = 'BatchNIIDataPrepare_manifest.csv'
FAILED_XSECTION_FOLDER_NAME = '0_Failed_Xsection'
DEFAULT_XSECTION_TIMEOUT_MINUTES = 20
MAX_XSECTION_ERROR_LENGTH = 4000
FILTER_MASK_INTERRUPTED_MESSAGE = 'Filter Mask started but did not finish.'
XSECTION_INTERRUPTED_MESSAGE = 'Xsection process started but did not finish.'
LUMEN_INTERRUPTED_MESSAGE = 'Lumen Correction started but did not finish.'
FILTER_MASK_REGENERATION_MESSAGE = 'Filter Mask requires downstream regeneration'
CTA_REGENERATION_MESSAGE = 'CTA xsection requires Lumen Correction regeneration'
SEG_REGENERATION_MESSAGE = 'seg xsection requires Lumen Correction regeneration'
LOCK_TIMEOUT_SECONDS = 120
LOCK_POLL_SECONDS = 0.1
ATOMIC_REPLACE_ATTEMPTS = 10
ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.05
ATOMIC_REPLACE_MAX_DELAY_SECONDS = 0.5
POST_WRITE_VALIDATION_ATTEMPTS = 4
POST_WRITE_VALIDATION_DELAY_SECONDS = 0.05
HASH_BLOCK_SIZE = 1024 * 1024


def _artery_status_fields(artery):
    return (
        '{0}_xsection_CTA_exe_failed'.format(artery),
        '{0}_xsection_CTA_exe_er_msg'.format(artery),
        '{0}_xsection_CTA_complete'.format(artery),
        '{0}_xsection_seg_exe_failed'.format(artery),
        '{0}_xsection_seg_exe_er_msg'.format(artery),
        '{0}_xsection_seg_complete'.format(artery),
        '{0}_lumen_complete'.format(artery),
        '{0}_lumen_er_msg'.format(artery),
    )


def _filter_mask_status_fields(artery):
    prefix = '{0}_filter_mask'.format(artery)
    return (
        prefix + '_complete',
        prefix + '_er_msg',
        prefix + '_fingerprint',
    )


XSECTION_STATUS_FIELDS = ['run_xsection', 'complete_segmentation']
for _artery_name in XSECTION_ARTERY_NAMES:
    XSECTION_STATUS_FIELDS.extend(_artery_status_fields(_artery_name)[:3])
XSECTION_STATUS_FIELDS.append('complete_xsection_CTA')
for _artery_name in XSECTION_ARTERY_NAMES:
    XSECTION_STATUS_FIELDS.extend(_artery_status_fields(_artery_name)[3:6])
XSECTION_STATUS_FIELDS.append('complete_xsection_seg')
for _artery_name in XSECTION_ARTERY_NAMES:
    XSECTION_STATUS_FIELDS.extend(_artery_status_fields(_artery_name)[6:])
XSECTION_STATUS_FIELDS.append('complete_lumen_correction')
XSECTION_STATUS_FIELDS = tuple(XSECTION_STATUS_FIELDS)

XSECTION_STATUS_DEFAULTS = {
    field_name: ('' if field_name.endswith('_er_msg') else 0)
    for field_name in XSECTION_STATUS_FIELDS
}
XSECTION_FINGERPRINT_FIELDS = tuple(
    '{0}_{1}_fingerprint'.format(artery, stage)
    for artery in XSECTION_ARTERY_NAMES
    for stage in ('xsection_CTA', 'xsection_seg', 'lumen')
)
XSECTION_LOCAL_FIELDS = [
    'filter_mask_complete',
    'filter_mask_er_msg',
    'filter_mask_fingerprint',
]
for _artery_name in XSECTION_ARTERY_NAMES:
    XSECTION_LOCAL_FIELDS.extend((
        '{0}_filter_mask_complete'.format(_artery_name),
        '{0}_filter_mask_er_msg'.format(_artery_name),
        '{0}_filter_mask_fingerprint'.format(_artery_name),
    ))
XSECTION_LOCAL_FIELDS = tuple(XSECTION_LOCAL_FIELDS)
XSECTION_LOCAL_DEFAULTS = {
    field_name: (0 if field_name.endswith('_complete') else '')
    for field_name in XSECTION_LOCAL_FIELDS
}
XSECTION_RECORD_FIELDS = (
    'case_id',
    'batch_name',
    'case_relative_path',
) + XSECTION_STATUS_FIELDS + XSECTION_LOCAL_FIELDS + XSECTION_FINGERPRINT_FIELDS


def _widget_text(widget):
    if widget is None:
        return ''
    for method_name in ('toPlainText', 'text'):
        method = getattr(widget, method_name, None)
        if callable(method):
            return str(method()).strip()
    return ''


def _widget_integer(widget, default):
    if widget is None:
        return int(default)
    value_method = getattr(widget, 'value', None)
    if callable(value_method):
        try:
            return int(value_method())
        except (TypeError, ValueError):
            return int(default)
    text_value = _widget_text(widget)
    try:
        return int(text_value)
    except (TypeError, ValueError):
        return int(default)


def _widget_checked(widget, default=False):
    if widget is None:
        return bool(default)
    method = getattr(widget, 'isChecked', None)
    return bool(method()) if callable(method) else bool(default)


def _clean_error_message(message, limit=MAX_XSECTION_ERROR_LENGTH):
    message = ' '.join(str(message or '').replace('\x00', '').split())
    if len(message) <= limit:
        return message
    marker = ' ...[truncated]... '
    if limit <= len(marker) + 2:
        return message[:limit]
    prefix_length = (limit - len(marker)) // 2
    suffix_length = limit - len(marker) - prefix_length
    return message[:prefix_length] + marker + message[-suffix_length:]


def _append_error(existing, message):
    parts = [part for part in (existing, message) if str(part or '').strip()]
    return _clean_error_message('; '.join(str(part) for part in parts))


def _is_nonempty_regular_file(file_path):
    try:
        path = Path(file_path)
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def validate_nifti(file_path, require_nonzero=False):
    """Return ``(valid, message)`` for a readable, usable NIfTI artifact."""

    path = Path(file_path)
    if not _is_nonempty_regular_file(path):
        return False, 'Missing or empty NIfTI: {0}'.format(path)
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        number_of_pixels = 1
        for dimension_size in reader.GetSize():
            number_of_pixels *= int(dimension_size)
        if number_of_pixels <= 0:
            return False, 'NIfTI has no pixels: {0}'.format(path)
        if require_nonzero:
            image = reader.Execute()
            array = sitk.GetArrayViewFromImage(image)
            if not numpy.any(array != 0):
                return False, 'NIfTI contains no nonzero segmentation voxels: {0}'.format(path)
    except Exception as error:
        return False, 'Unreadable NIfTI {0}: {1}'.format(path, error)
    return True, ''


def validate_nifti_after_write(
        file_path,
        require_nonzero=False,
        attempts=POST_WRITE_VALIDATION_ATTEMPTS,
        delay=POST_WRITE_VALIDATION_DELAY_SECONDS):
    """Retry transient missing/unreadable results immediately after a writer."""

    attempts = max(1, int(attempts))
    delay = max(0.0, float(delay))
    result = (False, '')
    for attempt_number in range(1, attempts + 1):
        result = validate_nifti(file_path, require_nonzero=require_nonzero)
        if result[0]:
            return result
        message = result[1]
        semantic_failure = (
            'contains no nonzero segmentation voxels' in message
            or 'has no pixels' in message
        )
        if semantic_failure or attempt_number >= attempts:
            return result
        if delay:
            time.sleep(delay)
    return result


def _sha256_file(file_path):
    digest = hashlib.sha256()
    with Path(file_path).open('rb') as file_object:
        while True:
            block = file_object.read(HASH_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _file_signature(file_path, hash_contents=False):
    path = Path(file_path)
    if not path.is_file():
        return 'missing:{0}'.format(path)
    stat_result = path.stat()
    values = [str(path.resolve()), str(stat_result.st_size)]
    if hash_contents:
        values.append(_sha256_file(path))
    else:
        values.append(str(stat_result.st_mtime_ns))
    return '|'.join(values)


def _fingerprint(parts):
    digest = hashlib.sha256()
    for part in parts:
        value = str(part).encode('utf-8')
        digest.update(str(len(value)).encode('ascii'))
        digest.update(b':')
        digest.update(value)
        digest.update(b'\n')
    return digest.hexdigest()


def _path_is_within(path, parent):
    path = os.path.normcase(os.path.abspath(str(path)))
    parent = os.path.normcase(os.path.abspath(str(parent)))
    try:
        return os.path.commonpath([path, parent]) == parent
    except ValueError:
        return False


def _flush_and_sync(file_object):
    """Flush a file, tolerating Windows CRT ``fsync`` descriptor failures.

    A successful Python flush is sufficient for publishing a complete file by
    same-directory atomic replacement.  Some Windows filesystems intermittently
    reject ``os.fsync`` with ``EBADF`` even though the still-open file handle is
    writable.  Do not let that optional durability step abort the batch, while
    continuing to propagate every other I/O failure.
    """

    file_object.flush()
    try:
        os.fsync(file_object.fileno())
    except OSError as error:
        if os.name != 'nt' or error.errno != errno.EBADF or file_object.closed:
            raise
        try:
            os.fstat(file_object.fileno())
        except OSError:
            raise error


def _replace_with_retry(
        source_path,
        destination_path,
        attempts=ATOMIC_REPLACE_ATTEMPTS,
        initial_delay=ATOMIC_REPLACE_INITIAL_DELAY_SECONDS,
        maximum_delay=ATOMIC_REPLACE_MAX_DELAY_SECONDS):
    """Atomically publish a file, retrying Windows sharing violations."""

    source_path = Path(source_path)
    destination_path = Path(destination_path)
    attempts = max(1, int(attempts))
    delay = max(0.0, float(initial_delay))
    maximum_delay = max(delay, float(maximum_delay))
    retryable_errnos = {errno.EACCES, errno.EBUSY, errno.EPERM}
    retryable_winerrors = {5, 32, 33}
    last_error = None

    for attempt_number in range(1, attempts + 1):
        try:
            os.replace(str(source_path), str(destination_path))
            return
        except OSError as error:
            last_error = error
            retryable = (
                isinstance(error, PermissionError)
                or error.errno in retryable_errnos
                or getattr(error, 'winerror', None) in retryable_winerrors
            )
            if not retryable or attempt_number >= attempts:
                break
            if delay:
                time.sleep(delay)
                delay = min(maximum_delay, delay * 2)

    if last_error is None:  # Defensive; the loop always returns or records one.
        raise RuntimeError(
            'Atomic replacement did not run for {0}'.format(destination_path)
        )
    if (
            isinstance(last_error, PermissionError)
            or last_error.errno in retryable_errnos
            or getattr(last_error, 'winerror', None) in retryable_winerrors):
        raise PermissionError(
            errno.EACCES,
            'Could not atomically publish {0} after {1} attempts because the '
            'destination remained inaccessible or locked. Close Excel and any '
            'other program using the file, then resume the batch. Original '
            'error: {2}'.format(destination_path, attempts, last_error),
            str(destination_path),
        ) from last_error
    raise last_error


@contextmanager
def interprocess_file_lock(lock_path, timeout_seconds=LOCK_TIMEOUT_SECONDS):
    """Hold a one-byte OS lock; the OS releases it if the process exits."""

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open('a+b')
    acquired = False
    try:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b'0')
            lock_file.flush()
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                lock_file.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (IOError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Timed out waiting for file lock: {0}'.format(lock_path))
                time.sleep(LOCK_POLL_SECONDS)
        yield
    finally:
        if acquired:
            try:
                lock_file.seek(0)
                if os.name == 'nt':
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
    with csv_path.open('r', encoding='utf-8-sig', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader.fieldnames or []), list(reader)


def _write_csv_atomic(csv_path, fieldnames, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix='.{0}.'.format(csv_path.name),
        suffix='.tmp',
        dir=str(csv_path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, 'w', encoding='utf-8', newline='') as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
                extrasaction='ignore',
                lineterminator='\n',
            )
            writer.writeheader()
            writer.writerows(rows)
            _flush_and_sync(csv_file)
        _replace_with_retry(temporary_path, csv_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_xsection_records(record_path, records, lock_timeout=LOCK_TIMEOUT_SECONDS):
    record_path = Path(record_path)
    ordered_records = sorted(
        records,
        key=lambda row: (
            str(row.get('batch_name', '')).casefold(),
            str(row.get('case_id', '')).casefold(),
        ),
    )
    with interprocess_file_lock(str(record_path) + '.lock', lock_timeout):
        _write_csv_atomic(record_path, XSECTION_RECORD_FIELDS, ordered_records)
    return record_path


def merge_xsection_records(manifest_path, record_path, lock_timeout=LOCK_TIMEOUT_SECONDS):
    """Merge one batch ledger into a shared manifest without losing columns."""

    manifest_path = Path(manifest_path)
    record_path = Path(record_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with interprocess_file_lock(str(manifest_path) + '.lock', lock_timeout):
        manifest_fields, manifest_rows = _read_csv_rows(manifest_path)
        if not manifest_fields:
            manifest_fields = ['case_id', 'batch_name', 'case_relative_path']
        missing_identity = [
            field_name for field_name in ('case_id', 'batch_name')
            if field_name not in manifest_fields
        ]
        if missing_identity:
            raise ValueError(
                'Manifest is missing required identity columns {0}: {1}'.format(
                    ', '.join(missing_identity), manifest_path
                )
            )
        if 'case_relative_path' not in manifest_fields:
            manifest_fields.append('case_relative_path')
        for field_name in XSECTION_STATUS_FIELDS:
            if field_name not in manifest_fields:
                manifest_fields.append(field_name)

        exact_rows = {}
        case_rows = {}
        for row in manifest_rows:
            case_id = str(row.get('case_id', '')).strip()
            batch_name = str(row.get('batch_name', '')).strip()
            if not case_id:
                continue
            exact_rows.setdefault((batch_name.casefold(), case_id.casefold()), row)
            case_rows.setdefault(case_id.casefold(), []).append(row)

        with interprocess_file_lock(str(record_path) + '.lock', lock_timeout):
            record_fields, batch_records = _read_csv_rows(record_path)
        merge_fields = [
            field_name for field_name in XSECTION_STATUS_FIELDS
            if field_name in record_fields
        ]
        merged_count = 0
        for record in batch_records:
            case_id = str(record.get('case_id', '')).strip()
            batch_name = str(record.get('batch_name', '')).strip()
            if not case_id or not batch_name:
                continue
            key = (batch_name.casefold(), case_id.casefold())
            row = exact_rows.get(key)
            if row is None:
                candidates = [
                    candidate for candidate in case_rows.get(case_id.casefold(), [])
                    if not str(candidate.get('batch_name', '')).strip()
                ]
                if len(candidates) == 1:
                    row = candidates[0]
                    row['batch_name'] = batch_name
                    exact_rows[key] = row
                else:
                    row = {field_name: '' for field_name in manifest_fields}
                    row['case_id'] = case_id
                    row['batch_name'] = batch_name
                    row['case_relative_path'] = record.get('case_relative_path', '')
                    manifest_rows.append(row)
                    exact_rows[key] = row
                    case_rows.setdefault(case_id.casefold(), []).append(row)
            for field_name in merge_fields:
                row[field_name] = record.get(field_name, '')
            merged_count += 1
        _write_csv_atomic(manifest_path, manifest_fields, manifest_rows)
    return {
        'manifest_path': manifest_path,
        'record_count': merged_count,
        'row_count': len(manifest_rows),
    }


def _terminate_process_tree(process, wait_seconds=5):
    errors = []
    if process is None or process.poll() is not None:
        return errors
    if psutil is None:
        try:
            process.kill()
        except Exception as error:
            errors.append(str(error))
        return errors
    try:
        parent = psutil.Process(process.pid)
        processes = list(reversed(parent.children(recursive=True))) + [parent]
        for child_process in processes:
            try:
                child_process.terminate()
            except psutil.NoSuchProcess:
                pass
            except Exception as error:
                errors.append(str(error))
        _, alive = psutil.wait_procs(processes, timeout=wait_seconds)
        for child_process in alive:
            try:
                child_process.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception as error:
                errors.append(str(error))
        if alive:
            psutil.wait_procs(alive, timeout=wait_seconds)
    except psutil.NoSuchProcess:
        pass
    except Exception as error:
        errors.append(str(error))
        try:
            process.kill()
        except Exception as kill_error:
            errors.append(str(kill_error))
    return errors


def run_xsection_command(arguments, timeout_seconds, output_limit=MAX_XSECTION_ERROR_LENGTH):
    """Run one CPR command and return bounded diagnostics instead of raising."""

    result = {
        'arguments': [str(value) for value in arguments],
        'returncode': None,
        'timed_out': False,
        'exe_failed': True,
        'error_message': '',
        'duration_seconds': 0.0,
    }
    process = None
    started_at = time.monotonic()
    with tempfile.TemporaryFile(mode='w+b') as output_file:
        try:
            creation_flags = 0
            if os.name == 'nt':
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
            process = subprocess.Popen(
                result['arguments'],
                stdin=subprocess.DEVNULL,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                shell=False,
                creationflags=creation_flags,
            )
            try:
                process.wait(timeout=float(timeout_seconds))
            except subprocess.TimeoutExpired:
                result['timed_out'] = True
                cleanup_errors = _terminate_process_tree(process)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                        process.wait(timeout=10)
                    except Exception as error:
                        cleanup_errors.append(str(error))
                result['error_message'] = 'Timed out after {0} seconds'.format(timeout_seconds)
                if cleanup_errors:
                    result['error_message'] += '; cleanup: {0}'.format('; '.join(cleanup_errors))
            result['returncode'] = process.poll()
            if not result['timed_out'] and result['returncode'] != 0:
                result['error_message'] = 'Exited with return code {0}'.format(result['returncode'])
        except (OSError, ValueError) as error:
            result['error_message'] = 'Could not start process: {0}'.format(error)
        except Exception as error:
            if process is not None and process.poll() is None:
                _terminate_process_tree(process)
            result['error_message'] = 'Process error: {0}'.format(error)

        try:
            output_file.flush()
            end_position = output_file.seek(0, os.SEEK_END)
            output_file.seek(max(0, end_position - int(output_limit)))
            output_tail = output_file.read().decode(
                locale.getpreferredencoding(False) or 'utf-8', errors='replace'
            ).strip()
        except Exception as error:
            output_tail = ''
            result['error_message'] = _append_error(
                result['error_message'], 'Could not read process output: {0}'.format(error)
            )
    if result['error_message'] and output_tail:
        result['error_message'] = _append_error(
            result['error_message'], 'output: {0}'.format(output_tail)
        )
    result['exe_failed'] = bool(
        result['timed_out']
        or result['returncode'] != 0
        or result['error_message']
    )
    result['duration_seconds'] = round(time.monotonic() - started_at, 3)
    return result


class BatchSliceRadiomics:
    """Batch xsection orchestration with resumable, per-artifact outcomes."""

    FILTER_MASK_HEADERS = (
        'MaskPath', 'MaskVal', 'NewMaskVal', 'OutputFolder', 'FileNameRef'
    )
    XSECTION_HEADERS = (
        'inputDir', 'inputCSVDir', 'outputDir', 'outputCSVDir',
        'resampleDepth', 'CPRAngle', 'cpus', 'casePath', 'EXEPath'
    )
    CASE_HEADERS = (
        'Interpolation', 'OutCSV', 'inputImagePath', 'inputCSVPath',
        'outputImagePath', 'outputCSVPath'
    )
    XSECTION_FIX_HEADERS = (
        'Image Path', 'Mask Path', 'Process Path', 'Fill Original Value',
        'Output Directory', 'Output Name'
    )

    def __init__(
            self,
            UI=None,
            Hedys=None,
            xsection_runner=None,
            lumen_runner=None,
            filter_mask_runner=None):
        self.ui = UI
        self.model = Hedys
        self.modelui = Hedys.ui if Hedys is not None else None
        self._xsection_runner = xsection_runner or run_xsection_command
        self._lumen_runner = lumen_runner
        self._filter_mask_runner = filter_mask_runner
        self.initProcess()

        if self.ui is not None:
            self._connect_button('LoadFolderPath_Btn_BSRJZ', self.ChooseWorkingDir)
            self._connect_button('ChooseCopyOriginPath_Btn_BSRJZ', self.ChooseCopyOriginDir)
            self._connect_button('ChooseCPRExePathBtn_2_BPP', self.ChooseCPRExePath)
            self._connect_button(
                'ChooseLumenCorrectionPathBtn_BPP', self.ChooseLumenCorretionPath
            )
            self._connect_button('ChooseManifestPathBtn_BSRJZ', self.ChooseManifestPath)
            self._connect_button('ChooseFailedRootPathBtn_BSRJZ', self.ChooseFailedRootPath)
            self._connect_button('RunBtn_CPP', self.mainFunction)
            if self.modelui is not None:
                self.modelui.QStackedWidget_Module.setCurrentWidget(self.ui.centralwidget)

    def _connect_button(self, name, callback):
        button = getattr(self.ui, name, None)
        if button is not None:
            button.clicked.connect(callback)

    def initProcess(self):
        self.WorkingDir = None
        self.CopyOriginDir = None
        self.CPREXEPath = None
        self.LumenProcessSourcePath = None
        self.ManifestPath = None
        self.FailedRootPath = None
        self.BatchName = None
        self.TableDir = None

        self.DCMDirList = []
        self.CTADirList = []
        self.CSVPathList = []
        self.OriDataPathList = []
        self.BranchPathList = []
        self.XsectionPathList = []
        self.XsectionFixPathList = []

        self.DictionaryPath = None
        self.FilterMaskTable = []
        self.FilterMaskTablePath = None
        self.XsectionTable = []
        self.XsectionTablePath = None
        self.XsectionFixTable = []
        self.XsectionFixTablePath = None
        self.LCProcessPath = None
        self.XsectionRecordPath = None
        self.RecordMap = {}
        self._branch_validation_attempted = False
        self._validated_use_paths = {}

        self.CopyFileFlag = False
        self.ScanFileFlag = True
        self.CreateFolderFlag = True
        self.BranchDictFlag = True
        self.GenerateTFlag = True
        self.FMFlag = True
        self.XsecFlag = True
        self.LCFlag = True
        self.MaskXsectionFlag = True
        self.MaskXsectionOnlyFlag = False
        self.RerunFailedFlag = False
        self.SelectedArteries = list(XSECTION_ARTERY_NAMES)
        self.XsectionTimeoutSeconds = DEFAULT_XSECTION_TIMEOUT_MINUTES * 60

    def CheckCheckBoxStatus(self):
        if self.ui is None:
            return
        self.CopyFileFlag = _widget_checked(getattr(self.ui, 'CopyFile_CB_BSRJZ', None))
        self.ScanFileFlag = _widget_checked(getattr(self.ui, 'ScanFile_CB_BSRJZ', None), True)
        self.CreateFolderFlag = _widget_checked(
            getattr(self.ui, 'CreateFolders_CB_BSRJZ', None), True
        )
        self.BranchDictFlag = _widget_checked(
            getattr(self.ui, 'BranchDict_CB_BSRJZ', None), True
        )
        self.GenerateTFlag = _widget_checked(
            getattr(self.ui, 'GenerateTable_CB_BSRJZ', None), True
        )
        self.FMFlag = _widget_checked(getattr(self.ui, 'FM_CB_BSRJZ', None), True)
        self.XsecFlag = _widget_checked(
            getattr(self.ui, 'Xsection_CB_BSRJZ', None), True
        )
        self.LCFlag = _widget_checked(
            getattr(self.ui, 'LumenCorrection_CB_BSRJZ', None), True
        )
        self.MaskXsectionFlag = _widget_checked(
            getattr(self.ui, 'maskXsection_CB_BSRJZ', None), True
        )
        self.MaskXsectionOnlyFlag = _widget_checked(
            getattr(self.ui, 'maskXsectionOnly_CB_BSRJZ', None), False
        )
        self.RerunFailedFlag = _widget_checked(
            getattr(self.ui, 'RerunFailedCases_CB_BSRJZ', None), False
        )
        artery_widgets = (
            ('LCx', 'pmdCx_CB_BSRJZ'),
            ('mLAD', 'pmdLAD_CB_BSRJZ'),
            ('mRCA', 'pmdRCA_CB_BSRJZ'),
        )
        self.SelectedArteries = [
            artery for artery, widget_name in artery_widgets
            if _widget_checked(getattr(self.ui, widget_name, None), True)
        ]
        timeout_minutes = _widget_integer(
            getattr(self.ui, 'XsectionTimeout_SB_BSRJZ', None),
            DEFAULT_XSECTION_TIMEOUT_MINUTES,
        )
        self.XsectionTimeoutSeconds = max(1, timeout_minutes) * 60

    '''
    ##############################################################################
    # Choose files and folders
    ##############################################################################
    '''

    def ChooseWorkingDir(self):
        dirname = Save_Load_File.OpenDirPathQt(dispMsg='Working Directory',
                                               fileObj=self.ui,
                                               qtObj=True)
        if dirname:
            self.ui.LoadFolderPath_Txt_BSRJZ.setPlainText(dirname)

    def ChooseCopyOriginDir(self):
        dirname = Save_Load_File.OpenDirPathQt(dispMsg='Copying from Which Directory?',
                                               fileObj=self.ui,
                                               qtObj=True)
        if dirname:
            self.ui.CopyOriginFolderPath_Txt_BSRJZ.setPlainText(dirname)

    # used to choose CPR exe path
    def ChooseCPRExePath(self):
        filename = Save_Load_File.OpenFilePathQt(
            dispMsg='Load CPR Exe file',
            fileTypes='All files (*.*);; exe files(*.exe)',
            fileObj=self.ui,
            qtObj=True
        )
        if filename:
            self.ui.CPRExePath_Txt_BSRJZ.setPlainText(filename)

    # used to choose lumen correction path
    def ChooseLumenCorretionPath(self):
        filename = Save_Load_File.OpenFilePathQt(
            dispMsg='Load Lumen Correction process table',
            fileTypes='CSV files (*.csv);; All files (*.*)',
            fileObj=self.ui,
            qtObj=True
        )
        if filename:
            self.ui.LumenCorrectionPath_Txt_BSRJZ.setPlainText(filename)

    def ChooseManifestPath(self):
        filename = Save_Load_File.SaveFilePathQT(
            dispMsg='Select or create the shared manifest CSV',
            fileTypes='CSV files (*.csv);; All files (*.*)',
            fileObj=self.ui,
            qtObj=True,
        )
        if filename:
            self.ui.ManifestPath_Txt_BSRJZ.setPlainText(filename)

    def ChooseFailedRootPath(self):
        dirname = Save_Load_File.OpenDirPathQt(
            dispMsg='Failed artifact parent directory',
            fileObj=self.ui,
            qtObj=True,
        )
        if dirname:
            self.ui.FailedRootPath_Txt_BSRJZ.setPlainText(dirname)

    '''
    ##############################################################################
    # Other functions
    ##############################################################################
    '''

    def ScanDCMDir(self):
        working_root = Path(self.WorkingDir)
        case_paths = set()
        for root, _, files in os.walk(str(working_root)):
            if not any(str(file_name).lower().endswith('.dcm') for file_name in files):
                continue
            try:
                relative_parts = Path(root).relative_to(working_root).parts
            except ValueError:
                continue
            if relative_parts:
                case_paths.add(working_root / relative_parts[0])
        self.DCMDirList = [
            str(path) for path in sorted(case_paths, key=lambda item: str(item).casefold())
        ]

    def ScanCTADir(self):
        working_root = Path(self.WorkingDir)
        case_paths = []
        for child_path in working_root.iterdir():
            if not child_path.is_dir() or child_path.name == 'Table':
                continue
            if any(path.is_file() for path in self._cta_candidates(child_path)):
                case_paths.append(child_path)
        self.DCMDirList = [
            str(path) for path in sorted(case_paths, key=lambda item: str(item).casefold())
        ]

    @staticmethod
    def _cta_candidates(case_path):
        """Return the preferred single-CTA path followed by the legacy path."""

        case_path = Path(case_path)
        return (
            case_path / 'Preproc' / 'OriData' / 'CTA.nii.gz',
            case_path / 'CTA' / 'CTA.nii.gz',
        )

    @classmethod
    def _preferred_cta_path(cls, case_path):
        candidates = cls._cta_candidates(case_path)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    def CreateFoldersDCM(self):
        for case_path in self.DCMDirList:
            self.CreateFolders(CasePath=case_path)

    # input path then create structured folder under this folder
    def CreateFolders(self, CasePath=None):
        case_path = Path(CasePath)
        folders = (
            Path(self.TableDir),
            case_path / 'CTA',
            case_path / 'Preproc',
            case_path / 'Preproc' / 'OriData',
            case_path / 'Preproc' / 'OriData' / 'Branches',
            case_path / 'Preproc' / 'CSV',
            case_path / 'Preproc' / 'Xsection',
            case_path / 'Preproc' / 'XsectionFix',
        )
        for folder_path in folders:
            folder_path.mkdir(parents=True, exist_ok=True)

    # store the required path information as list
    def StoreListInformation(self):
        self.OriDataPathList = []
        self.CSVPathList = []
        self.BranchPathList = []
        self.XsectionPathList = []
        self.XsectionFixPathList = []
        for case_path_text in self.DCMDirList:
            case_path = Path(case_path_text)
            self.OriDataPathList.append(str(case_path / 'Preproc' / 'OriData'))
            self.CSVPathList.append(str(case_path / 'Preproc' / 'CSV'))
            self.BranchPathList.append(
                str(case_path / 'Preproc' / 'OriData' / 'Branches')
            )
            self.XsectionPathList.append(str(case_path / 'Preproc' / 'Xsection'))
            self.XsectionFixPathList.append(
                str(case_path / 'Preproc' / 'XsectionFix')
            )

        self.CTADirList = [
            str(self._preferred_cta_path(case_path))
            for case_path in self.DCMDirList
        ]
        table_dir = Path(self.TableDir)
        self.DictionaryPath = str(table_dir / 'coronarySegments_BSRJZ.json')
        self.FilterMaskTablePath = str(table_dir / 'FilterMask_BSRJZ.csv')
        self.XsectionTablePath = str(table_dir / 'Xsection_BSRJZ.csv')
        self.XsectionFixTablePath = str(table_dir / 'XsectionFix_BSRJZ.csv')
        self.LCProcessPath = str(table_dir / 'LCProcess_BSRJZ.csv')
        self.XsectionRecordPath = str(
            table_dir / XSECTION_RECORD_FOLDER / XSECTION_RECORD_FILE
        )

    # input path, join path with json file, return a list include l and r json file path
    @staticmethod
    def ReturnJsonList(CSVPath=None):
        tempJsonList = []
        tempJsonPath1 = os.path.join(CSVPath, 'L_coronary_centerline_network.json')
        tempJsonPath2 = os.path.join(CSVPath, 'R_coronary_centerline_network.json')
        tempJsonList.append(tempJsonPath1)
        tempJsonList.append(tempJsonPath2)

        return tempJsonList

    # write branch processing dictionary
    def CheckWriteDict(self):
        branch_list = list(self.SelectedArteries)
        ignore_text = _widget_text(
            getattr(self.ui, 'ignoreCheck_Txt_BSRJZ', None)
        ) if self.ui is not None else ''
        ignore_coordinates = Preprocess_Mask.StrToLst(
            strIn=ignore_text
        )["listOut"]
        self._branch_validation_attempted = True
        self._validated_use_paths = {}
        dictionary_path = Path(self.DictionaryPath)
        dictionary_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix='.coronarySegments_BSRJZ.',
            suffix='.json.tmp',
            dir=str(dictionary_path.parent),
            text=True,
        )
        os.close(file_descriptor)
        temporary_dictionary_path = Path(temporary_name)
        try:
            BranchDictionary.WriteDic(
                Outpath=str(temporary_dictionary_path),
                branchLst=branch_list,
            )
            with temporary_dictionary_path.open('r', encoding='utf-8') as json_file:
                dictionary_content = json.load(json_file)
            if not isinstance(dictionary_content, list):
                raise ValueError('artery dictionary must contain a list')
            _replace_with_retry(temporary_dictionary_path, dictionary_path)
        finally:
            if temporary_dictionary_path.exists():
                temporary_dictionary_path.unlink()
        outcomes = {}
        for csv_path_text in self.CSVPathList:
            csv_path = Path(csv_path_text)
            output_path = csv_path / 'coronarySegmentsUse_BSRJZ.json'
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix='.coronarySegmentsUse_BSRJZ.',
                suffix='.json.tmp',
                dir=str(csv_path),
                text=True,
            )
            os.close(file_descriptor)
            temporary_output_path = Path(temporary_name)
            try:
                BranchDictionary.ValidateUse(
                    dicPath=self.DictionaryPath,
                    jsonList=self.ReturnJsonList(str(csv_path)),
                    outputPath=str(temporary_output_path),
                    ignoreCoordsLst=ignore_coordinates,
                    checkParent=False,
                    checkSecondary=False,
                )
                with temporary_output_path.open('r', encoding='utf-8') as json_file:
                    validated_content = json.load(json_file)
                if not isinstance(validated_content, list):
                    raise ValueError(
                        'validated artery dictionary must contain a list'
                    )
                _replace_with_retry(temporary_output_path, output_path)
                self._validated_use_paths[
                    os.path.normcase(os.path.abspath(str(csv_path)))
                ] = output_path
                outcomes[str(csv_path)] = True
            except Exception as error:
                outcomes[str(csv_path)] = False
                self.UpdateMsgLog(
                    'Artery dictionary failed; continuing with later cases: '
                    '{0}: {1}'.format(csv_path.parent.parent.name, error)
                )
            finally:
                if temporary_output_path.exists():
                    temporary_output_path.unlink()
        return outcomes

    def _case_use_json(self, csv_path):
        csv_path = Path(csv_path)
        csv_key = os.path.normcase(os.path.abspath(str(csv_path)))
        if self._branch_validation_attempted:
            current_path = self._validated_use_paths.get(csv_key)
            if current_path is None:
                raise ValueError(
                    'Current-run artery validation did not complete for {0}; '
                    'a stale generated JSON will not be reused'.format(csv_path)
                )
            return current_path
        suffixed_path = csv_path / 'coronarySegmentsUse_BSRJZ.json'
        if suffixed_path.is_file():
            return suffixed_path
        return csv_path / 'coronarySegmentsUse.json'

    @staticmethod
    def _write_table(path, headers, rows):
        dictionaries = [
            dict(zip(headers, row)) if not isinstance(row, dict) else row
            for row in rows
        ]
        _write_csv_atomic(path, list(headers), dictionaries)

    def _prepare_lumen_process(self):
        source_path = Path(self.LumenProcessSourcePath or '')
        destination_path = Path(self.LCProcessPath)
        if not _is_nonempty_regular_file(source_path):
            raise FileNotFoundError(
                'Lumen correction process table is missing or empty: {0}'.format(
                    source_path
                )
            )
        if source_path.resolve() == destination_path.resolve():
            return destination_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix='.{0}.'.format(destination_path.name),
            suffix='.tmp',
            dir=str(destination_path.parent),
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            shutil.copy2(str(source_path), str(temporary_path))
            _replace_with_retry(temporary_path, destination_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return destination_path

    # create Filter Mask Object and batch process
    def GenerateTable(
            self,
            resampleDepth='0.1',
            cpus='2',
            CPRAngle='0',
            maskXsection=None,
            maskXsectionOnly=None):
        if not self.SelectedArteries:
            raise ValueError('Select at least one artery before generating tables')
        if maskXsection is None:
            maskXsection = self.MaskXsectionFlag
        if maskXsectionOnly is None:
            maskXsectionOnly = self.MaskXsectionOnlyFlag
        generate_cta = not bool(maskXsectionOnly)
        generate_segmentation = bool(maskXsection or maskXsectionOnly)

        self.FilterMaskTable = []
        self.XsectionTable = []
        self.XsectionFixTable = []
        successful_cases = []

        for case_index, csv_path_text in enumerate(self.CSVPathList):
            csv_path = Path(csv_path_text)
            case_path = Path(self.DCMDirList[case_index])
            case_rows = []
            filter_checkpoint = len(self.FilterMaskTable)
            fix_checkpoint = len(self.XsectionFixTable)
            try:
                use_json_path = self._case_use_json(csv_path)
                with use_json_path.open('r', encoding='utf-8') as json_file:
                    branches = json.load(json_file)
                if not isinstance(branches, list):
                    raise ValueError('artery-use JSON must contain a list')

                for branch in branches:
                    artery = str(branch.get('branch_name', ''))
                    if artery not in self.SelectedArteries or not branch.get('use'):
                        continue
                    label = branch.get('label')
                    if isinstance(label, (list, tuple)):
                        label = label[0] if label else None
                    required_keys = (
                        'original_branch', 'original_csv', '3D_csv',
                        'cross_section_image_LI', 'cross_section_mask_NN',
                    )
                    missing_keys = [key for key in required_keys if not branch.get(key)]
                    if label is None or missing_keys:
                        raise ValueError(
                            'Invalid {0} branch metadata; missing {1}'.format(
                                artery,
                                ', '.join((['label'] if label is None else []) + missing_keys),
                            )
                        )

                    if generate_segmentation:
                        self.FilterMaskTable.append([
                            str(Path(self.OriDataPathList[case_index]) /
                                'sc_labelled_coronary_segment.nii.gz'),
                            str(label),
                            '1',
                            self.BranchPathList[case_index],
                            branch['original_branch'],
                        ])
                    if generate_cta:
                        case_rows.append([
                            '0', '0', 'CTA.nii.gz', branch['original_csv'],
                            branch['cross_section_image_LI'], branch['3D_csv'],
                        ])
                    if generate_segmentation:
                        # OutCSV stays disabled because per-artery 3D point CSVs
                        # are very large and are not used by this workflow.
                        case_rows.append([
                            '1', '0',
                            str(Path('Branches') /
                                (branch['original_branch'] + 'Val.nii.gz')),
                            branch['original_csv'],
                            branch['cross_section_mask_NN'], branch['3D_csv'],
                        ])
                        correction_name = branch.get(
                            'cross_section_mask_fix',
                            branch['cross_section_mask_NN'].replace(
                                '.nii.gz', 'FixVal.nii.gz'
                            ),
                        )
                        if correction_name.endswith('Val.nii.gz'):
                            correction_name = correction_name[:-len('Val.nii.gz')]
                        else:
                            correction_name = correction_name.replace('.nii.gz', '')
                        self.XsectionFixTable.append([
                            str(Path(self.XsectionPathList[case_index]) /
                                branch['cross_section_image_LI']),
                            str(Path(self.XsectionPathList[case_index]) /
                                branch['cross_section_mask_NN']),
                            self.LCProcessPath,
                            'Yes',
                            self.XsectionFixPathList[case_index],
                            correction_name,
                        ])

                case_table_path = csv_path / 'case_BSRJZ.csv'
                self._write_table(case_table_path, self.CASE_HEADERS, case_rows)
                self.XsectionTable.append([
                    self.OriDataPathList[case_index],
                    str(csv_path),
                    self.XsectionPathList[case_index],
                    self.XsectionPathList[case_index],
                    str(resampleDepth),
                    str(CPRAngle),
                    str(cpus),
                    str(case_table_path),
                    self.CPREXEPath,
                ])
                successful_cases.append(str(case_path))
            except Exception as error:
                del self.FilterMaskTable[filter_checkpoint:]
                del self.XsectionFixTable[fix_checkpoint:]
                self.UpdateMsgLog(
                    'Table generation failed; continuing: {0}: {1}'.format(
                        case_path.name, error
                    )
                )

        self._write_table(
            self.FilterMaskTablePath,
            self.FILTER_MASK_HEADERS,
            self.FilterMaskTable,
        )
        self._write_table(
            self.XsectionTablePath,
            self.XSECTION_HEADERS,
            self.XsectionTable,
        )
        self._write_table(
            self.XsectionFixTablePath,
            self.XSECTION_FIX_HEADERS,
            self.XsectionFixTable,
        )
        return successful_cases

    def BatchFilterMask(self):
        if not self.SelectedArteries:
            raise ValueError('Select at least one artery before running Filter Mask')

        table_path = Path(self.FilterMaskTablePath)
        if not table_path.is_file():
            legacy_path = Path(self.TableDir) / 'FilterMask.csv'
            if legacy_path.is_file():
                table_path = legacy_path
        _, rows = _read_csv_rows(table_path)
        grouped_rows = {}
        for row in rows:
            output_folder = str(row.get('OutputFolder', '')).strip()
            grouped_rows.setdefault(os.path.normcase(os.path.abspath(output_folder)), []).append(row)

        record_map = self._load_xsection_record_map()
        outcomes = {}
        skipped_arteries = 0
        skipped_failed_arteries = 0
        completed_arteries = 0
        incomplete_arteries = 0
        table_error_cases = 0
        for case_path_text, branch_path_text in zip(
                self.DCMDirList, self.BranchPathList):
            case_path = Path(case_path_text)
            record = self._ensure_case_record(record_map, case_path)
            legacy_filter_complete = record.get('filter_mask_complete', 0)
            legacy_filter_error = record.get('filter_mask_er_msg', '')
            legacy_filter_fingerprint = record.get(
                'filter_mask_fingerprint', ''
            )
            case_rows = grouped_rows.get(
                os.path.normcase(os.path.abspath(branch_path_text)), []
            )
            artery_rows = {}
            artery_expected_paths = {}
            artery_fingerprint_parts = {}
            artery_preflight_errors = {
                artery: [] for artery in self.SelectedArteries
            }
            fingerprint_parts = ['filter-mask-v1']
            case_preflight_errors = []
            source_signatures = {}
            for row in case_rows:
                mask_path = Path(str(row.get('MaskPath', '')).strip())
                output_folder = Path(str(row.get('OutputFolder', '')).strip())
                file_name_reference = str(row.get('FileNameRef', '')).strip()
                expected_path = output_folder / (
                    file_name_reference + 'Val.nii.gz'
                )
                expected_non_value_path = output_folder / (
                    file_name_reference + 'NVl.nii.gz'
                )
                artery = next((
                    artery_name for artery_name in self.SelectedArteries
                    if file_name_reference.casefold().startswith(
                        artery_name.casefold()
                    )
                ), None)
                if artery is None:
                    case_preflight_errors.append(
                        'Cannot identify Filter Mask artery from FileNameRef: '
                        '{0}'.format(file_name_reference)
                    )
                elif artery in artery_rows:
                    artery_preflight_errors[artery].append(
                        'Duplicate Filter Mask row for {0}'.format(artery)
                    )
                else:
                    artery_rows[artery] = row
                    artery_expected_paths[artery] = (
                        expected_path, expected_non_value_path
                    )
                if not _path_is_within(mask_path, case_path):
                    input_error = (
                        'Filter Mask input is outside case: {0}'.format(mask_path)
                    )
                    if artery is None:
                        case_preflight_errors.append(input_error)
                    else:
                        artery_preflight_errors[artery].append(input_error)
                else:
                    valid_input, input_error = validate_nifti(
                        mask_path, require_nonzero=True
                    )
                    if not valid_input:
                        if artery is None:
                            case_preflight_errors.append(input_error)
                        else:
                            artery_preflight_errors[artery].append(input_error)
                if not _path_is_within(expected_path, case_path):
                    output_error = (
                        'Filter Mask output is outside case: {0}'.format(
                            expected_path
                        )
                    )
                    if artery is None:
                        case_preflight_errors.append(output_error)
                    else:
                        artery_preflight_errors[artery].append(output_error)
                if not _path_is_within(expected_non_value_path, case_path):
                    output_error = (
                        'Filter Mask output is outside case: {0}'.format(
                            expected_non_value_path
                        )
                    )
                    if artery is None:
                        case_preflight_errors.append(output_error)
                    else:
                        artery_preflight_errors[artery].append(output_error)
                mask_key = os.path.normcase(os.path.abspath(str(mask_path)))
                if mask_key not in source_signatures:
                    source_signatures[mask_key] = _file_signature(
                        mask_path, hash_contents=True
                    )
                fingerprint_parts.extend((
                    source_signatures[mask_key],
                    str(row.get('MaskVal', '')),
                    str(row.get('NewMaskVal', '')),
                    str(output_folder.resolve()),
                    file_name_reference,
                ))
                if artery is not None:
                    artery_fingerprint_parts[artery] = (
                        'filter-mask-artery-v1',
                        source_signatures[mask_key],
                        str(row.get('MaskVal', '')),
                        str(row.get('NewMaskVal', '')),
                        str(output_folder.resolve()),
                        file_name_reference,
                    )
            fingerprint = _fingerprint(fingerprint_parts)
            artery_fingerprints = {
                artery: _fingerprint(parts)
                for artery, parts in artery_fingerprint_parts.items()
            }
            missing_arteries = [
                artery for artery in self.SelectedArteries
                if artery not in artery_expected_paths
            ]
            for artery in missing_arteries:
                artery_preflight_errors[artery].append(
                    'No Filter Mask row for {0}'.format(artery)
                )
            record['filter_mask_fingerprint'] = fingerprint
            artery_outcomes = {}
            for artery in self.SelectedArteries:
                complete_field, message_field, fingerprint_field = (
                    _filter_mask_status_fields(artery)
                )
                artery_fingerprint = artery_fingerprints.get(artery)
                if not artery_fingerprint:
                    artery_fingerprint = _fingerprint((
                        'filter-mask-artery-v1',
                        'missing-row',
                        str(case_path.resolve()),
                        artery,
                    ))
                expected_paths = artery_expected_paths.get(artery, ())
                preflight_errors = artery_preflight_errors[artery]
                artery_status_known = bool(
                    self._status_is_complete(record, complete_field)
                    or str(record.get(message_field, '')).strip()
                    or str(record.get(fingerprint_field, '')).strip()
                )
                if artery_status_known:
                    previous_complete = record.get(complete_field, 0)
                    previous_error = record.get(message_field, '')
                    previous_fingerprint = record.get(fingerprint_field, '')
                else:
                    # Older ledgers stored only one case-level Filter Mask
                    # result.  Its fingerprint is not comparable with the
                    # newer per-artery fingerprint, so keep it unknown.
                    previous_complete = legacy_filter_complete
                    previous_error = legacy_filter_error
                    previous_fingerprint = ''
                    if str(previous_error).strip().startswith(
                            FILTER_MASK_INTERRUPTED_MESSAGE.rstrip('.')):
                        previous_error = FILTER_MASK_INTERRUPTED_MESSAGE
                resume_state = self._resume_state(
                    previous_complete,
                    False,
                    previous_error,
                    previous_fingerprint,
                    artery_fingerprint,
                    FILTER_MASK_INTERRUPTED_MESSAGE,
                )
                outputs_valid = bool(expected_paths) and all(
                    validate_nifti(path, require_nonzero=True)[0]
                    for path in expected_paths
                )
                matching_resume = (
                    resume_state == 'complete'
                    and not preflight_errors
                    and outputs_valid
                    and self._status_is_complete(record, complete_field)
                    and str(record.get(fingerprint_field, '')) == artery_fingerprint
                )
                legacy_adoption = (
                    resume_state in ('complete', 'pending')
                    and not preflight_errors
                    and outputs_valid
                    and not str(record.get(fingerprint_field, '')).strip()
                )
                if matching_resume or legacy_adoption:
                    record[complete_field] = 1
                    record[message_field] = ''
                    record[fingerprint_field] = artery_fingerprint
                    artery_outcomes[artery] = {
                        'complete': True,
                        'skipped': True,
                        'error_message': '',
                    }
                    skipped_arteries += 1
                    self._save_xsection_record_map(record_map)
                    continue

                if resume_state == 'failed' and not self.RerunFailedFlag:
                    artery_outcomes[artery] = {
                        'complete': False,
                        'skipped': True,
                        'failed_skipped': True,
                        'legacy_failed_skip': not artery_status_known,
                        'error_message': str(previous_error),
                    }
                    skipped_failed_arteries += 1
                    incomplete_arteries += 1
                    continue

                if preflight_errors:
                    message = _clean_error_message('; '.join(preflight_errors))
                    record['filter_mask_complete'] = 0
                    record['filter_mask_er_msg'] = message
                    record[complete_field] = 0
                    record[message_field] = message
                    record[fingerprint_field] = artery_fingerprint
                    self._invalidate_filter_dependents(
                        record,
                        'Filter Mask preflight failed: {0}'.format(message),
                        arteries=(artery,),
                    )
                    artery_outcomes[artery] = {
                        'complete': False,
                        'skipped': False,
                        'error_message': message,
                    }
                    incomplete_arteries += 1
                    self._save_xsection_record_map(record_map)
                    continue

                record['filter_mask_complete'] = 0
                record['filter_mask_er_msg'] = (
                    'Filter Mask started but did not finish: {0}.'.format(
                        artery
                    )
                )
                record[complete_field] = 0
                record[message_field] = (
                    FILTER_MASK_INTERRUPTED_MESSAGE
                )
                record[fingerprint_field] = artery_fingerprint
                self._invalidate_filter_dependents(
                    record,
                    FILTER_MASK_REGENERATION_MESSAGE,
                    arteries=(artery,),
                )
                self._save_xsection_record_map(record_map)

                pre_output_signatures = {
                    path: _file_signature(path) for path in expected_paths
                }
                file_descriptor, temporary_name = tempfile.mkstemp(
                    prefix='.filter_mask_{0}_{1}.'.format(
                        case_path.name, artery
                    ),
                    suffix='.csv',
                    dir=str(self.TableDir),
                    text=True,
                )
                os.close(file_descriptor)
                temporary_path = Path(temporary_name)
                message = ''
                try:
                    self._write_table(
                        temporary_path,
                        self.FILTER_MASK_HEADERS,
                        [artery_rows[artery]],
                    )
                    if self._filter_mask_runner is not None:
                        self._filter_mask_runner(str(temporary_path))
                    elif (self.model is not None
                          and hasattr(self.model, 'filterMask')):
                        self.model.filterMask.BatchProcessing(
                            str(temporary_path)
                        )
                    else:
                        raise RuntimeError('Filter Mask processor is unavailable')
                    output_errors = []
                    for path in expected_paths:
                        output_valid, validation_error = validate_nifti_after_write(
                            path, require_nonzero=True
                        )
                        pre_output_signature = pre_output_signatures[path]
                        if (
                                pre_output_signature.split(':', 1)[0] != 'missing'
                                and _file_signature(path) == pre_output_signature):
                            output_valid = False
                            validation_error = (
                                'Filter Mask output was not updated by this run: '
                                '{0}'.format(path)
                            )
                        if not output_valid:
                            output_errors.append(validation_error)
                    message = _clean_error_message('; '.join(output_errors))
                except Exception as error:
                    message = _clean_error_message(error)
                finally:
                    if temporary_path.exists():
                        temporary_path.unlink()

                artery_complete = not bool(message)
                record[complete_field] = int(artery_complete)
                record[message_field] = message
                record[fingerprint_field] = artery_fingerprint
                artery_outcomes[artery] = {
                    'complete': artery_complete,
                    'skipped': False,
                    'error_message': message,
                }
                if artery_complete:
                    completed_arteries += 1
                else:
                    incomplete_arteries += 1
                    self._invalidate_filter_dependents(
                        record,
                        'Filter Mask output validation failed: {0}'.format(
                            message
                        ),
                        arteries=(artery,),
                    )
                    self.UpdateMsgLog(
                        'Filter Mask failed; continuing: {0}/{1}: {2}'.format(
                            case_path.name, artery, message
                        )
                    )
                self._save_xsection_record_map(record_map)

            artery_messages = [
                '{0}: {1}'.format(
                    artery, artery_outcomes[artery]['error_message']
                )
                for artery in self.SelectedArteries
                if artery_outcomes[artery]['error_message']
            ]
            artery_messages.extend(case_preflight_errors)
            if case_preflight_errors:
                table_error_cases += 1
            preserve_legacy_failure = bool(self.SelectedArteries) and all(
                artery_outcomes[artery].get('legacy_failed_skip', False)
                for artery in self.SelectedArteries
            )
            if preserve_legacy_failure:
                record['filter_mask_complete'] = legacy_filter_complete
                record['filter_mask_er_msg'] = legacy_filter_error
                record['filter_mask_fingerprint'] = legacy_filter_fingerprint
            else:
                record['filter_mask_complete'] = int(
                    not case_preflight_errors and all(
                        artery_outcomes[artery]['complete']
                        for artery in self.SelectedArteries
                    )
                )
                record['filter_mask_er_msg'] = _clean_error_message(
                    '; '.join(artery_messages)
                )
                record['filter_mask_fingerprint'] = fingerprint
            outcomes[str(case_path)] = {
                'complete': self._status_is_complete(
                    record, 'filter_mask_complete'
                ),
                'skipped': bool(
                    self.SelectedArteries and all(
                        artery_outcomes[artery]['skipped']
                        for artery in self.SelectedArteries
                    )
                ),
                'failed_skipped': any(
                    artery_outcomes[artery].get('failed_skipped', False)
                    for artery in self.SelectedArteries
                ),
                'error_message': record['filter_mask_er_msg'],
                'arteries': artery_outcomes,
            }
            self._save_xsection_record_map(record_map)

        self.UpdateMsgLog(
            'Filter Mask resume summary: skipped complete {0}, skipped failed '
            '{1}, completed {2}, incomplete {3} artery outputs; table-error '
            'cases {4}.'.format(
                skipped_arteries,
                skipped_failed_arteries,
                completed_arteries,
                incomplete_arteries,
                table_error_cases,
            )
        )
        return outcomes

    @staticmethod
    def _record_key(batch_name, case_id):
        return str(batch_name).casefold(), str(case_id).casefold()

    def _load_xsection_record_map(self):
        record_path = Path(self.XsectionRecordPath)
        with interprocess_file_lock(str(record_path) + '.lock'):
            _, rows = _read_csv_rows(record_path)
        record_map = {}
        for row in rows:
            case_id = str(row.get('case_id', '')).strip()
            batch_name = str(row.get('batch_name', '')).strip()
            if not case_id or not batch_name:
                continue
            normalized = dict(XSECTION_STATUS_DEFAULTS)
            normalized.update(XSECTION_LOCAL_DEFAULTS)
            normalized.update({field: '' for field in XSECTION_FINGERPRINT_FIELDS})
            normalized.update(row)
            normalized['case_id'] = case_id
            normalized['batch_name'] = batch_name
            normalized.setdefault('case_relative_path', '')
            record_map[self._record_key(batch_name, case_id)] = normalized
        self.RecordMap = record_map
        return record_map

    def _ensure_case_record(self, record_map, case_path):
        case_path = Path(case_path)
        key = self._record_key(self.BatchName, case_path.name)
        record = record_map.get(key)
        if record is None:
            record = dict(XSECTION_STATUS_DEFAULTS)
            record.update(XSECTION_LOCAL_DEFAULTS)
            record.update({field: '' for field in XSECTION_FINGERPRINT_FIELDS})
            record_map[key] = record
        try:
            relative_path = case_path.relative_to(Path(self.WorkingDir).parent)
        except ValueError:
            relative_path = Path(self.BatchName) / case_path.name
        record['case_id'] = case_path.name
        record['batch_name'] = self.BatchName
        record['case_relative_path'] = str(relative_path).replace('\\', '/')
        for field_name, default_value in XSECTION_STATUS_DEFAULTS.items():
            record.setdefault(field_name, default_value)
        for field_name, default_value in XSECTION_LOCAL_DEFAULTS.items():
            record.setdefault(field_name, default_value)
        for field_name in XSECTION_FINGERPRINT_FIELDS:
            record.setdefault(field_name, '')
        return record

    def _save_xsection_record_map(self, record_map=None):
        if record_map is None:
            record_map = self.RecordMap
        self.RecordMap = record_map
        write_xsection_records(
            self.XsectionRecordPath,
            list(record_map.values()),
        )

    def _merge_xsection_records(self):
        try:
            return merge_xsection_records(
                self.ManifestPath,
                self.XsectionRecordPath,
            )
        except Exception as error:
            self.UpdateMsgLog(
                'Manifest merge warning; the batch ledger is preserved at '
                '{0}: {1}'.format(self.XsectionRecordPath, error)
            )
            return None

    def _initialize_case_records(self):
        record_map = self._load_xsection_record_map()
        for case_path_text in self.DCMDirList:
            case_path = Path(case_path_text)
            record = self._ensure_case_record(record_map, case_path)
            source_segmentation = (
                case_path / 'Preproc' / 'OriData' /
                'sc_labelled_coronary_segment.nii.gz'
            )
            valid_segmentation, _ = validate_nifti(
                source_segmentation, require_nonzero=True
            )
            record['complete_segmentation'] = int(valid_segmentation)
        self._save_xsection_record_map(record_map)
        self._merge_xsection_records()
        return record_map

    @staticmethod
    def _status_is_complete(record, field_name):
        return str(record.get(field_name, '0')).strip() == '1'

    @staticmethod
    def _resume_state(
            complete,
            failed,
            error_message,
            stored_fingerprint,
            current_fingerprint,
            interrupted_message,
            pending_messages=()):
        """Classify one persisted task without changing its record.

        A non-empty changed fingerprint represents new work, while a blank
        legacy fingerprint cannot prove that the failed task has changed.
        Interruption sentinels take precedence over pessimistic failure bits
        written immediately before an external process starts.
        """

        error_message = str(error_message or '').strip()
        stored_fingerprint = str(stored_fingerprint or '').strip()
        current_fingerprint = str(current_fingerprint or '').strip()
        failed = failed is True or str(failed or '0').strip() == '1'
        complete = complete is True or str(complete or '0').strip() == '1'

        if (
                stored_fingerprint
                and current_fingerprint
                and stored_fingerprint != current_fingerprint):
            return 'changed'
        pending_messages = set(pending_messages or ())
        error_parts = [
            part.strip() for part in error_message.split(';') if part.strip()
        ]
        if (
                error_message == interrupted_message
                or (
                    interrupted_message in error_parts
                    and all(
                        part == interrupted_message or part in pending_messages
                        for part in error_parts
                    )
                )):
            return 'interrupted'
        if error_parts and all(part in pending_messages for part in error_parts):
            return 'pending'
        if failed or error_message:
            return 'failed'
        if complete:
            return 'complete'
        return 'pending'

    def _update_case_aggregates(self, record):
        selected = list(self.SelectedArteries)
        record['complete_xsection_CTA'] = int(
            bool(selected) and all(
                self._status_is_complete(
                    record, '{0}_xsection_CTA_complete'.format(artery)
                ) for artery in selected
            )
        )
        record['complete_xsection_seg'] = int(
            bool(selected) and all(
                self._status_is_complete(
                    record, '{0}_xsection_seg_complete'.format(artery)
                ) for artery in selected
            )
        )
        record['complete_lumen_correction'] = int(
            bool(selected) and all(
                self._status_is_complete(
                    record, '{0}_lumen_complete'.format(artery)
                ) for artery in selected
            )
        )

    def _invalidate_filter_dependents(self, record, message, arteries=None):
        for artery in (self.SelectedArteries if arteries is None else arteries):
            segmentation_message = '{0}_xsection_seg_exe_er_msg'.format(artery)
            record['{0}_xsection_seg_complete'.format(artery)] = 0
            record[segmentation_message] = _append_error(
                record.get(segmentation_message, ''), message
            )
            lumen_message = '{0}_lumen_er_msg'.format(artery)
            record['{0}_lumen_complete'.format(artery)] = 0
            record[lumen_message] = _append_error(
                record.get(lumen_message, ''), message
            )
        self._update_case_aggregates(record)

    def _invalidate_lumen_dependency(self, record, artery, message):
        message_field = '{0}_lumen_er_msg'.format(artery)
        record['{0}_lumen_complete'.format(artery)] = 0
        record[message_field] = _append_error(
            record.get(message_field, ''), message
        )
        self._update_case_aggregates(record)

    @staticmethod
    def _xsection_fields(artery, artifact_type):
        prefix = '{0}_xsection_{1}'.format(artery, artifact_type)
        return (
            prefix + '_exe_failed',
            prefix + '_exe_er_msg',
            prefix + '_complete',
            prefix + '_fingerprint',
        )

    @staticmethod
    def _artifact_file_name(artery, artifact_type):
        for artery_name, cta_name, segmentation_name in XSECTION_ARTERIES:
            if artery_name == artery:
                return cta_name if artifact_type == 'CTA' else segmentation_name
        raise KeyError(artery)

    def _move_failed_artifact(self, artifact_path, case_path):
        artifact_path = Path(artifact_path)
        case_path = Path(case_path)
        if not artifact_path.is_file():
            return None
        if not _path_is_within(artifact_path, case_path):
            raise ValueError(
                'Refusing to move an artifact outside its case: {0}'.format(
                    artifact_path
                )
            )
        failed_root = (
            Path(self.FailedRootPath) / FAILED_XSECTION_FOLDER_NAME /
            self.BatchName / case_path.name
        )
        failed_parent = Path(self.FailedRootPath) / FAILED_XSECTION_FOLDER_NAME
        if not _path_is_within(failed_root, failed_parent):
            raise ValueError('Unsafe failed-artifact destination: {0}'.format(failed_root))
        failed_root.mkdir(parents=True, exist_ok=True)
        destination = failed_root / artifact_path.name
        if destination.exists():
            if artifact_path.name.lower().endswith('.nii.gz'):
                stem = artifact_path.name[:-7]
                suffix = '.nii.gz'
            else:
                stem = artifact_path.stem
                suffix = artifact_path.suffix
            counter = 1
            while destination.exists():
                destination = failed_root / '{0}_{1}{2}'.format(
                    stem, counter, suffix
                )
                counter += 1
        shutil.move(str(artifact_path), str(destination))
        return destination

    def _case_path_from_output(self, output_path):
        output_path = Path(output_path)
        for case_path_text in self.DCMDirList:
            case_path = Path(case_path_text)
            if _path_is_within(output_path, case_path):
                return case_path
        raise ValueError('Output is outside the current cases: {0}'.format(output_path))

    def _load_xsection_jobs(self):
        table_path = Path(self.XsectionTablePath)
        if not table_path.is_file():
            legacy_path = Path(self.TableDir) / 'Xsection.csv'
            if legacy_path.is_file():
                table_path = legacy_path
        _, batch_rows = _read_csv_rows(table_path)
        jobs = {}
        for batch_row in batch_rows:
            case_table_path = Path(str(batch_row.get('casePath', '')).strip())
            _, case_rows = _read_csv_rows(case_table_path)
            for case_row in case_rows:
                output_name = Path(
                    str(case_row.get('outputImagePath', '')).strip()
                ).name
                output_metadata = XSECTION_OUTPUT_LOOKUP.get(output_name.casefold())
                if output_metadata is None:
                    continue
                artery, artifact_type = output_metadata
                if artery not in self.SelectedArteries:
                    continue
                input_image_path = Path(str(batch_row['inputDir'])) / str(
                    case_row['inputImagePath']
                )
                centerline_path = Path(str(batch_row['inputCSVDir'])) / str(
                    case_row['inputCSVPath']
                )
                output_path = Path(str(batch_row['outputDir'])) / output_name
                output_points_path = Path(str(batch_row['outputCSVDir'])) / str(
                    case_row['outputCSVPath']
                )
                case_path = self._case_path_from_output(output_path)
                arguments = [
                    str(batch_row['EXEPath']),
                    str(input_image_path),
                    str(centerline_path),
                    str(output_path),
                    str(output_points_path),
                    str(batch_row['resampleDepth']),
                    str(batch_row['CPRAngle']),
                    '0',
                    str(case_row['Interpolation']),
                    str(case_row['OutCSV']),
                ]
                fingerprint = _fingerprint((
                    'xsection-v1', artery, artifact_type,
                    _file_signature(
                        input_image_path,
                        hash_contents=(artifact_type == 'seg'),
                    ),
                    _file_signature(centerline_path, hash_contents=True),
                    'exe-path:{0}'.format(
                        os.path.normcase(
                            os.path.abspath(str(batch_row['EXEPath']))
                        )
                    ),
                    *arguments[5:],
                ))
                key = (
                    self._record_key(self.BatchName, case_path.name),
                    artery,
                    artifact_type,
                )
                if key in jobs:
                    raise ValueError(
                        'Duplicate xsection row for {0} {1} {2}'.format(
                            case_path.name, artery, artifact_type
                        )
                    )
                try:
                    workers = max(1, int(float(batch_row.get('cpus', 1))))
                except (TypeError, ValueError):
                    workers = 1
                jobs[key] = {
                    'case_path': case_path,
                    'artery': artery,
                    'artifact_type': artifact_type,
                    'input_image_path': input_image_path,
                    'centerline_path': centerline_path,
                    'output_path': output_path,
                    'output_points_path': output_points_path,
                    'arguments': arguments,
                    'fingerprint': fingerprint,
                    'workers': workers,
                }
        return jobs

    def _xsection_preflight_error(self, job):
        errors = []
        input_valid, input_error = validate_nifti(
            job['input_image_path'],
            require_nonzero=(job['artifact_type'] == 'seg'),
        )
        if not input_valid:
            errors.append(input_error)
        if not _is_nonempty_regular_file(job['centerline_path']):
            errors.append(
                'Missing or empty centerline CSV: {0}'.format(
                    job['centerline_path']
                )
            )
        job['output_path'].parent.mkdir(parents=True, exist_ok=True)
        job['output_points_path'].parent.mkdir(parents=True, exist_ok=True)
        return _clean_error_message('; '.join(errors))

    # create Xsection Object and batch process
    def BatchXsection(self):
        record_map = self._load_xsection_record_map()
        jobs = self._load_xsection_jobs()
        requested_types = {'seg'} if self.MaskXsectionOnlyFlag else {'CTA'}
        if self.MaskXsectionFlag:
            requested_types.add('seg')
        pending_jobs = []
        outcomes = {}

        for case_path_text in self.DCMDirList:
            case_path = Path(case_path_text)
            case_key = self._record_key(self.BatchName, case_path.name)
            record = self._ensure_case_record(record_map, case_path)
            record['run_xsection'] = 1
            source_segmentation = (
                case_path / 'Preproc' / 'OriData' /
                'sc_labelled_coronary_segment.nii.gz'
            )
            valid_source, source_validation_error = validate_nifti(
                source_segmentation, require_nonzero=True
            )
            record['complete_segmentation'] = int(valid_source)
            filter_status_known = bool(
                str(record.get('filter_mask_fingerprint', '')).strip()
                or str(record.get('filter_mask_er_msg', '')).strip()
            )
            filter_incomplete = (
                filter_status_known
                and not self._status_is_complete(record, 'filter_mask_complete')
            )

            for artery in self.SelectedArteries:
                for artifact_type in ('CTA', 'seg'):
                    failed_field, message_field, complete_field, fingerprint_field = (
                        self._xsection_fields(artery, artifact_type)
                    )
                    output_path = (
                        case_path / 'Preproc' / 'Xsection' /
                        self._artifact_file_name(artery, artifact_type)
                    )
                    valid_output, output_validation_error = validate_nifti(
                        output_path,
                        require_nonzero=(artifact_type == 'seg'),
                    )
                    if artifact_type not in requested_types:
                        record[complete_field] = int(valid_output)
                        if valid_output:
                            record[failed_field] = 0
                            record[message_field] = ''
                        continue

                    job = jobs.get((case_key, artery, artifact_type))
                    if job is None:
                        missing_job_fingerprint = _fingerprint((
                            'xsection-v1',
                            'missing-row',
                            str(case_path.resolve()),
                            artery,
                            artifact_type,
                        ))
                        missing_job_state = self._resume_state(
                            record.get(complete_field, 0),
                            record.get(failed_field, 0),
                            record.get(message_field, ''),
                            record.get(fingerprint_field, ''),
                            missing_job_fingerprint,
                            XSECTION_INTERRUPTED_MESSAGE,
                            (
                                (FILTER_MASK_REGENERATION_MESSAGE,)
                                if artifact_type == 'seg' else ()
                            ),
                        )
                        if (
                                missing_job_state == 'failed'
                                and not self.RerunFailedFlag):
                            outcomes[(case_path.name, artery, artifact_type)] = {
                                'complete': False,
                                'skipped': True,
                                'failed_skipped': True,
                                'error_message': record.get(message_field, ''),
                            }
                            continue
                        record[failed_field] = 0
                        record[complete_field] = 0
                        record[fingerprint_field] = missing_job_fingerprint
                        error_message = (
                            'No {0} Xsection table row for {1}'.format(
                                artifact_type, artery
                            )
                        )
                        if output_path.is_file() and not valid_output:
                            try:
                                moved_path = self._move_failed_artifact(
                                    output_path, case_path
                                )
                                error_message = _append_error(
                                    error_message,
                                    '{0}; moved failed artifact to {1}'.format(
                                        output_validation_error, moved_path
                                    ),
                                )
                            except Exception as move_error:
                                error_message = _append_error(
                                    error_message,
                                    'Could not move failed artifact: {0}'.format(
                                        move_error
                                    ),
                                )
                        record[message_field] = error_message
                        self._invalidate_lumen_dependency(
                            record, artery, error_message
                        )
                        outcomes[(case_path.name, artery, artifact_type)] = {
                            'complete': False,
                            'skipped': False,
                            'error_message': error_message,
                        }
                        continue

                    resume_state = self._resume_state(
                        record.get(complete_field, 0),
                        record.get(failed_field, 0),
                        record.get(message_field, ''),
                        record.get(fingerprint_field, ''),
                        job['fingerprint'],
                        XSECTION_INTERRUPTED_MESSAGE,
                        (
                            (FILTER_MASK_REGENERATION_MESSAGE,)
                            if artifact_type == 'seg' else ()
                        ),
                    )
                    if resume_state == 'failed' and not self.RerunFailedFlag:
                        outcomes[(case_path.name, artery, artifact_type)] = {
                            'complete': False,
                            'skipped': True,
                            'failed_skipped': True,
                            'error_message': record.get(message_field, ''),
                        }
                        continue

                    dependency_error = ''
                    if artifact_type == 'seg' and not valid_source:
                        dependency_error = source_validation_error
                    artery_filter_complete, artery_filter_message, artery_filter_fingerprint = (
                        _filter_mask_status_fields(artery)
                    )
                    artery_filter_status_known = bool(
                        str(record.get(artery_filter_fingerprint, '')).strip()
                        or str(record.get(artery_filter_message, '')).strip()
                    )
                    artery_filter_incomplete = (
                        not self._status_is_complete(
                            record, artery_filter_complete
                        )
                        if artery_filter_status_known
                        else filter_incomplete
                    )
                    if artifact_type == 'seg' and artery_filter_incomplete:
                        filter_error = (
                            record.get(artery_filter_message, '')
                            if artery_filter_status_known
                            else record.get('filter_mask_er_msg', '')
                        )
                        dependency_error = _append_error(
                            dependency_error,
                            'Filter Mask is incomplete for {0}: {1}'.format(
                                artery,
                                filter_error
                                or 'no successful Filter Mask output'
                            ),
                        )
                    if dependency_error:
                        record[failed_field] = 0
                        record[message_field] = dependency_error
                        record[complete_field] = 0
                        record[fingerprint_field] = job['fingerprint']
                        self._invalidate_lumen_dependency(
                            record, artery, dependency_error
                        )
                        outcomes[(case_path.name, artery, artifact_type)] = {
                            'complete': False,
                            'skipped': False,
                            'error_message': dependency_error,
                        }
                        continue

                    preflight_error = self._xsection_preflight_error(job)

                    matching_resume = (
                        resume_state == 'complete'
                        and not preflight_error
                        and valid_output
                        and self._status_is_complete(record, complete_field)
                        and str(record.get(fingerprint_field, '')) == job['fingerprint']
                    )
                    legacy_adoption = (
                        resume_state in ('complete', 'pending')
                        and not preflight_error
                        and valid_output
                        and not str(record.get(fingerprint_field, '')).strip()
                    )
                    if matching_resume or legacy_adoption:
                        record[failed_field] = 0
                        record[message_field] = ''
                        record[complete_field] = 1
                        record[fingerprint_field] = job['fingerprint']
                        outcomes[(case_path.name, artery, artifact_type)] = {
                            'complete': True,
                            'skipped': True,
                            'error_message': '',
                        }
                        continue

                    if preflight_error:
                        if output_path.is_file() and not valid_output:
                            try:
                                moved_path = self._move_failed_artifact(
                                    output_path, case_path
                                )
                                preflight_error = _append_error(
                                    preflight_error,
                                    '{0}; moved failed artifact to {1}'.format(
                                        output_validation_error, moved_path
                                    ),
                                )
                            except Exception as move_error:
                                preflight_error = _append_error(
                                    preflight_error,
                                    'Could not move failed artifact: {0}'.format(
                                        move_error
                                    ),
                                )
                        record[failed_field] = 0
                        record[message_field] = preflight_error
                        record[complete_field] = 0
                        record[fingerprint_field] = job['fingerprint']
                        self._invalidate_lumen_dependency(
                            record, artery, preflight_error
                        )
                        outcomes[(case_path.name, artery, artifact_type)] = {
                            'complete': False,
                            'skipped': False,
                            'error_message': preflight_error,
                        }
                        continue

                    record[failed_field] = 1
                    record[message_field] = XSECTION_INTERRUPTED_MESSAGE
                    record[complete_field] = 0
                    record[fingerprint_field] = job['fingerprint']
                    self._invalidate_lumen_dependency(
                        record,
                        artery,
                        (
                            CTA_REGENERATION_MESSAGE
                            if artifact_type == 'CTA'
                            else SEG_REGENERATION_MESSAGE
                        ),
                    )
                    job['pre_output_signature'] = _file_signature(output_path)
                    job['fields'] = (
                        failed_field, message_field, complete_field,
                        fingerprint_field,
                    )
                    pending_jobs.append(job)
            self._update_case_aggregates(record)

        # Publish pessimistic state before the first external process starts.
        self._save_xsection_record_map(record_map)

        max_workers = max(
            [job['workers'] for job in pending_jobs] or [1]
        )
        max_workers = min(max_workers, max(1, len(pending_jobs)), 64)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_jobs = {
                executor.submit(
                    self._xsection_runner,
                    job['arguments'],
                    self.XsectionTimeoutSeconds,
                    MAX_XSECTION_ERROR_LENGTH,
                ): job
                for job in pending_jobs
            }
            for future in as_completed(future_jobs):
                job = future_jobs[future]
                record = record_map[
                    self._record_key(self.BatchName, job['case_path'].name)
                ]
                failed_field, message_field, complete_field, fingerprint_field = (
                    job['fields']
                )
                try:
                    result = future.result()
                    if not isinstance(result, dict):
                        raise TypeError('Xsection runner did not return a result dictionary')
                except Exception as error:
                    result = {
                        'exe_failed': True,
                        'error_message': 'Xsection runner error: {0}'.format(error),
                    }

                exe_failed = bool(result.get('exe_failed', True))
                error_message = _clean_error_message(
                    result.get('error_message', '')
                )
                valid_output, validation_error = validate_nifti_after_write(
                    job['output_path'],
                    require_nonzero=(job['artifact_type'] == 'seg'),
                )
                current_signature = _file_signature(job['output_path'])
                if (
                        job['pre_output_signature'].split(':', 1)[0] != 'missing'
                        and current_signature == job['pre_output_signature']):
                    valid_output = False
                    validation_error = (
                        'Xsection output was not updated by this run: {0}'.format(
                            job['output_path']
                        )
                    )
                complete = bool(not exe_failed and valid_output)
                if not complete and validation_error:
                    error_message = _append_error(error_message, validation_error)
                if not complete:
                    try:
                        moved_path = self._move_failed_artifact(
                            job['output_path'], job['case_path']
                        )
                        if moved_path is not None:
                            error_message = _append_error(
                                error_message,
                                'Moved failed artifact to {0}'.format(moved_path),
                            )
                    except Exception as move_error:
                        error_message = _append_error(
                            error_message,
                            'Could not move failed artifact: {0}'.format(move_error),
                        )
                record[failed_field] = int(exe_failed)
                record[message_field] = error_message
                record[complete_field] = int(complete)
                record[fingerprint_field] = job['fingerprint']
                self._update_case_aggregates(record)
                outcomes[(
                    job['case_path'].name,
                    job['artery'],
                    job['artifact_type'],
                )] = {
                    'complete': complete,
                    'skipped': False,
                    'error_message': error_message,
                }
                self._save_xsection_record_map(record_map)

        for record in record_map.values():
            self._update_case_aggregates(record)
        self._save_xsection_record_map(record_map)
        self._merge_xsection_records()
        skipped_complete = sum(
            1 for outcome in outcomes.values()
            if outcome.get('complete') and outcome.get('skipped')
        )
        skipped_failed = sum(
            1 for outcome in outcomes.values()
            if outcome.get('failed_skipped')
        )
        incomplete = sum(
            1 for outcome in outcomes.values()
            if not outcome.get('complete')
        )
        self.UpdateMsgLog(
            'Xsection resume summary: skipped complete {0}, skipped failed '
            '{1}, submitted {2}, incomplete {3}.'.format(
                skipped_complete,
                skipped_failed,
                len(pending_jobs),
                incomplete,
            )
        )
        return outcomes

    def _run_lumen_table(self, table_path):
        if self._lumen_runner is not None:
            return self._lumen_runner(str(table_path))
        if self.model is None or not hasattr(self.model, 'lumenCorrect'):
            raise RuntimeError('Lumen Correction processor is unavailable')
        return self.model.lumenCorrect.BatchCorrect(CSVPath=str(table_path))

    # create Lumen Correction Object and batch process
    def BatchLumenCorrection(self, process_snapshot_error=''):
        table_path = Path(self.XsectionFixTablePath)
        if not table_path.is_file():
            legacy_path = Path(self.TableDir) / 'XsectionFix.csv'
            if legacy_path.is_file():
                table_path = legacy_path
        _, correction_rows = _read_csv_rows(table_path)
        row_map = {}
        for row in correction_rows:
            row = dict(row)
            if _is_nonempty_regular_file(self.LCProcessPath):
                row['Process Path'] = self.LCProcessPath
            mask_path = Path(str(row.get('Mask Path', '')).strip())
            metadata = XSECTION_OUTPUT_LOOKUP.get(mask_path.name.casefold())
            if metadata is None or metadata[1] != 'seg':
                continue
            artery = metadata[0]
            output_directory = Path(
                str(row.get('Output Directory', '')).strip()
            )
            case_path = self._case_path_from_output(output_directory)
            row_map[(
                self._record_key(self.BatchName, case_path.name), artery
            )] = (case_path, row)

        record_map = self._load_xsection_record_map()
        outcomes = {}
        submitted_corrections = 0
        for case_path_text in self.DCMDirList:
            case_path = Path(case_path_text)
            case_key = self._record_key(self.BatchName, case_path.name)
            record = self._ensure_case_record(record_map, case_path)
            source_segmentation = (
                case_path / 'Preproc' / 'OriData' /
                'sc_labelled_coronary_segment.nii.gz'
            )
            source_valid, source_validation_error = validate_nifti(
                source_segmentation, require_nonzero=True
            )
            record['complete_segmentation'] = int(source_valid)
            filter_status_known = bool(
                str(record.get('filter_mask_fingerprint', '')).strip()
                or str(record.get('filter_mask_er_msg', '')).strip()
            )
            filter_incomplete = (
                filter_status_known
                and not self._status_is_complete(record, 'filter_mask_complete')
            )
            for artery in self.SelectedArteries:
                complete_field = '{0}_lumen_complete'.format(artery)
                message_field = '{0}_lumen_er_msg'.format(artery)
                fingerprint_field = '{0}_lumen_fingerprint'.format(artery)
                row_entry = row_map.get((case_key, artery))
                if row_entry is None:
                    missing_row_fingerprint = _fingerprint((
                        'lumen-v1',
                        'missing-row',
                        str(case_path.resolve()),
                        artery,
                    ))
                    missing_row_state = self._resume_state(
                        record.get(complete_field, 0),
                        False,
                        record.get(message_field, ''),
                        record.get(fingerprint_field, ''),
                        missing_row_fingerprint,
                        LUMEN_INTERRUPTED_MESSAGE,
                        (
                            FILTER_MASK_REGENERATION_MESSAGE,
                            CTA_REGENERATION_MESSAGE,
                            SEG_REGENERATION_MESSAGE,
                        ),
                    )
                    if (
                            missing_row_state == 'failed'
                            and not self.RerunFailedFlag):
                        outcomes[(case_path.name, artery)] = {
                            'complete': False,
                            'skipped': True,
                            'failed_skipped': True,
                            'error_message': record.get(message_field, ''),
                        }
                        continue
                    record[complete_field] = 0
                    record[fingerprint_field] = missing_row_fingerprint
                    record[message_field] = (
                        'No Lumen Correction table row for {0}'.format(artery)
                    )
                    outcomes[(case_path.name, artery)] = {
                        'complete': False,
                        'skipped': False,
                        'error_message': record[message_field],
                    }
                    continue
                _, row = row_entry
                image_path = Path(str(row['Image Path']))
                mask_path = Path(str(row['Mask Path']))
                process_path = Path(str(row['Process Path']))
                output_path = (
                    Path(str(row['Output Directory'])) /
                    ('{0}Val.nii.gz'.format(str(row['Output Name'])))
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fingerprint = _fingerprint((
                    'lumen-v1', artery,
                    _file_signature(image_path),
                    _file_signature(mask_path, hash_contents=True),
                    _file_signature(process_path, hash_contents=True),
                    str(row.get('Fill Original Value', '')),
                    str(row.get('Output Name', '')),
                ))
                valid_output, output_validation_error = validate_nifti(
                    output_path, require_nonzero=True
                )
                resume_state = self._resume_state(
                    record.get(complete_field, 0),
                    False,
                    record.get(message_field, ''),
                    record.get(fingerprint_field, ''),
                    fingerprint,
                    LUMEN_INTERRUPTED_MESSAGE,
                    (
                        FILTER_MASK_REGENERATION_MESSAGE,
                        CTA_REGENERATION_MESSAGE,
                        SEG_REGENERATION_MESSAGE,
                    ),
                )
                if resume_state == 'failed' and not self.RerunFailedFlag:
                    outcomes[(case_path.name, artery)] = {
                        'complete': False,
                        'skipped': True,
                        'failed_skipped': True,
                        'error_message': record.get(message_field, ''),
                    }
                    continue
                dependency_error = ''
                if not source_valid:
                    dependency_error = source_validation_error
                artery_filter_complete, artery_filter_message, artery_filter_fingerprint = (
                    _filter_mask_status_fields(artery)
                )
                artery_filter_status_known = bool(
                    str(record.get(artery_filter_fingerprint, '')).strip()
                    or str(record.get(artery_filter_message, '')).strip()
                )
                artery_filter_incomplete = (
                    not self._status_is_complete(record, artery_filter_complete)
                    if artery_filter_status_known
                    else filter_incomplete
                )
                if artery_filter_incomplete:
                    filter_error = (
                        record.get(artery_filter_message, '')
                        if artery_filter_status_known
                        else record.get('filter_mask_er_msg', '')
                    )
                    dependency_error = _append_error(
                        dependency_error,
                        'Filter Mask is incomplete for {0}: {1}'.format(
                            artery,
                            filter_error
                            or 'no successful Filter Mask output'
                        ),
                    )
                segmentation_complete_field = (
                    '{0}_xsection_seg_complete'.format(artery)
                )
                if (
                        self._status_is_complete(record, 'run_xsection')
                        and not self._status_is_complete(
                            record, segmentation_complete_field
                        )):
                    dependency_error = _append_error(
                        dependency_error,
                        'The {0} segmentation xsection is incomplete'.format(
                            artery
                        ),
                    )
                if dependency_error:
                    if output_path.is_file() and not valid_output:
                        try:
                            moved_path = self._move_failed_artifact(
                                output_path, case_path
                            )
                            dependency_error = _append_error(
                                dependency_error,
                                '{0}; moved failed artifact to {1}'.format(
                                    output_validation_error, moved_path
                                ),
                            )
                        except Exception as move_error:
                            dependency_error = _append_error(
                                dependency_error,
                                'Could not move failed corrected artifact: '
                                '{0}'.format(move_error),
                            )
                    record[complete_field] = 0
                    record[message_field] = dependency_error
                    record[fingerprint_field] = fingerprint
                    outcomes[(case_path.name, artery)] = {
                        'complete': False,
                        'skipped': False,
                        'error_message': dependency_error,
                    }
                    continue

                image_valid, image_error = validate_nifti(image_path)
                mask_valid, mask_error = validate_nifti(
                    mask_path, require_nonzero=True
                )
                process_valid = _is_nonempty_regular_file(process_path)
                resume_inputs_valid = bool(
                    not process_snapshot_error
                    and image_valid
                    and mask_valid
                    and process_valid
                )
                matching_resume = (
                    resume_state == 'complete'
                    and resume_inputs_valid
                    and valid_output
                    and self._status_is_complete(record, complete_field)
                    and str(record.get(fingerprint_field, '')) == fingerprint
                )
                legacy_adoption = (
                    resume_state in ('complete', 'pending')
                    and resume_inputs_valid
                    and valid_output
                    and not str(record.get(fingerprint_field, '')).strip()
                )
                if matching_resume or legacy_adoption:
                    record[complete_field] = 1
                    record[message_field] = ''
                    record[fingerprint_field] = fingerprint
                    outcomes[(case_path.name, artery)] = {
                        'complete': True,
                        'skipped': True,
                        'error_message': '',
                    }
                    continue

                preflight_errors = []
                if process_snapshot_error:
                    preflight_errors.append(str(process_snapshot_error))
                if not image_valid:
                    preflight_errors.append(image_error)
                    _, cta_message, cta_complete, _ = self._xsection_fields(
                        artery, 'CTA'
                    )
                    record[cta_complete] = 0
                    record[cta_message] = _append_error(
                        record.get(cta_message, ''), image_error
                    )
                    if image_path.is_file():
                        try:
                            moved_path = self._move_failed_artifact(
                                image_path, case_path
                            )
                            record[cta_message] = _append_error(
                                record[cta_message],
                                'Moved failed artifact to {0}'.format(moved_path),
                            )
                        except Exception as move_error:
                            record[cta_message] = _append_error(
                                record[cta_message], move_error
                            )
                if not mask_valid:
                    preflight_errors.append(mask_error)
                    _, seg_message, seg_complete, _ = (
                        self._xsection_fields(artery, 'seg')
                    )
                    record[seg_complete] = 0
                    record[seg_message] = _append_error(
                        record.get(seg_message, ''), mask_error
                    )
                    if mask_path.is_file():
                        try:
                            moved_path = self._move_failed_artifact(
                                mask_path, case_path
                            )
                            record[seg_message] = _append_error(
                                record[seg_message],
                                'Moved failed artifact to {0}'.format(moved_path),
                            )
                        except Exception as move_error:
                            record[seg_message] = _append_error(
                                record[seg_message], move_error
                            )
                if not process_valid:
                    preflight_errors.append(
                        'Missing or empty correction process table: {0}'.format(
                            process_path
                        )
                    )
                if output_path.is_file() and not valid_output:
                    try:
                        moved_path = self._move_failed_artifact(
                            output_path, case_path
                        )
                        preflight_errors.append(
                            '{0}; moved failed artifact to {1}'.format(
                                output_validation_error, moved_path
                            )
                        )
                    except Exception as move_error:
                        preflight_errors.append(
                            'Could not move failed corrected artifact: {0}'.format(
                                move_error
                            )
                        )
                if preflight_errors:
                    error_message = _clean_error_message(
                        '; '.join(preflight_errors)
                    )
                    record[complete_field] = 0
                    record[message_field] = error_message
                    record[fingerprint_field] = fingerprint
                    outcomes[(case_path.name, artery)] = {
                        'complete': False,
                        'skipped': False,
                        'error_message': error_message,
                    }
                    self._update_case_aggregates(record)
                    self._save_xsection_record_map(record_map)
                    continue

                record[complete_field] = 0
                record[message_field] = LUMEN_INTERRUPTED_MESSAGE
                record[fingerprint_field] = fingerprint
                self._update_case_aggregates(record)
                self._save_xsection_record_map(record_map)
                submitted_corrections += 1
                pre_output_signature = _file_signature(output_path)

                file_descriptor, temporary_name = tempfile.mkstemp(
                    prefix='.lumen_{0}_{1}.'.format(case_path.name, artery),
                    suffix='.csv',
                    dir=str(self.TableDir),
                    text=True,
                )
                os.close(file_descriptor)
                temporary_path = Path(temporary_name)
                error_message = ''
                try:
                    self._write_table(
                        temporary_path,
                        self.XSECTION_FIX_HEADERS,
                        [row],
                    )
                    self._run_lumen_table(temporary_path)
                except Exception as error:
                    error_message = _clean_error_message(error)
                finally:
                    if temporary_path.exists():
                        temporary_path.unlink()

                output_valid, validation_error = validate_nifti_after_write(
                    output_path, require_nonzero=True
                )
                if (
                        pre_output_signature.split(':', 1)[0] != 'missing'
                        and _file_signature(output_path) == pre_output_signature):
                    output_valid = False
                    validation_error = (
                        'Corrected segmentation was not updated by this run: '
                        '{0}'.format(output_path)
                    )
                complete = bool(not error_message and output_valid)
                if not complete and validation_error:
                    error_message = _append_error(
                        error_message, validation_error
                    )
                if not complete:
                    try:
                        moved_path = self._move_failed_artifact(
                            output_path, case_path
                        )
                        if moved_path is not None:
                            error_message = _append_error(
                                error_message,
                                'Moved failed artifact to {0}'.format(moved_path),
                            )
                    except Exception as move_error:
                        error_message = _append_error(
                            error_message,
                            'Could not move failed artifact: {0}'.format(move_error),
                        )
                record[complete_field] = int(complete)
                record[message_field] = error_message
                record[fingerprint_field] = fingerprint
                outcomes[(case_path.name, artery)] = {
                    'complete': complete,
                    'skipped': False,
                    'error_message': error_message,
                }
                self._update_case_aggregates(record)
                self._save_xsection_record_map(record_map)

            self._update_case_aggregates(record)
            self._save_xsection_record_map(record_map)

        self._merge_xsection_records()
        skipped_complete = sum(
            1 for outcome in outcomes.values()
            if outcome.get('complete') and outcome.get('skipped')
        )
        skipped_failed = sum(
            1 for outcome in outcomes.values()
            if outcome.get('failed_skipped')
        )
        incomplete = sum(
            1 for outcome in outcomes.values()
            if not outcome.get('complete')
        )
        self.UpdateMsgLog(
            'Lumen Correction resume summary: skipped complete {0}, skipped '
            'failed {1}, submitted {2}, incomplete {3}.'.format(
                skipped_complete,
                skipped_failed,
                submitted_corrections,
                incomplete,
            )
        )
        return outcomes


    '''
    ##############################################################################
    # Main Function
    ##############################################################################
    '''

    @staticmethod
    def _default_lumen_process_path():
        return str(
            Path(__file__).resolve().parents[3] /
            'Base' / 'Tables' / 'LC' / 'LCProcess.csv'
        )

    def _configure_from_ui(self):
        if self.ui is None:
            raise RuntimeError('BatchSliceRadiomics requires UI paths for mainFunction')
        self.WorkingDir = _widget_text(
            getattr(self.ui, 'LoadFolderPath_Txt_BSRJZ', None)
        )
        self.CopyOriginDir = _widget_text(
            getattr(self.ui, 'CopyOriginFolderPath_Txt_BSRJZ', None)
        )
        self.CPREXEPath = _widget_text(
            getattr(self.ui, 'CPRExePath_Txt_BSRJZ', None)
        )
        self.LumenProcessSourcePath = _widget_text(
            getattr(self.ui, 'LumenCorrectionPath_Txt_BSRJZ', None)
        ) or self._default_lumen_process_path()
        if not self.WorkingDir:
            raise ValueError('Working directory is required')
        working_path = Path(self.WorkingDir)
        if not working_path.is_dir():
            raise NotADirectoryError(
                'Working directory does not exist: {0}'.format(working_path)
            )
        self.WorkingDir = str(working_path)
        self.BatchName = working_path.name
        manifest_text = _widget_text(
            getattr(self.ui, 'ManifestPath_Txt_BSRJZ', None)
        )
        failed_root_text = _widget_text(
            getattr(self.ui, 'FailedRootPath_Txt_BSRJZ', None)
        )
        self.ManifestPath = str(
            Path(manifest_text) if manifest_text else
            working_path.parent / DEFAULT_MANIFEST_FILE_NAME
        )
        self.FailedRootPath = str(
            Path(failed_root_text) if failed_root_text else working_path.parent
        )
        self.TableDir = str(working_path / 'Table')
        Path(self.TableDir).mkdir(parents=True, exist_ok=True)
        self.StoreListInformation()

    def mainFunction(self):
        self.initProcess()
        self.CheckCheckBoxStatus()
        if not self.SelectedArteries:
            raise ValueError('Select at least one artery before running the batch')
        self._configure_from_ui()
        self.UpdateMsgLog(
            'Resume policy: completed work is validated and skipped; unchanged '
            'failed work will {0}.'.format(
                'be retried' if self.RerunFailedFlag else 'remain skipped'
            )
        )
        run_lock_path = (
            Path(self.TableDir) / XSECTION_RECORD_FOLDER / 'batch_run.lock'
        )
        started_at = time.monotonic()
        with interprocess_file_lock(run_lock_path):
            if self.CopyFileFlag:
                self.UpdateMsgLog('Copy Files Start')
                Save_Load_File.BatchCopyFiles(
                    fromFolder=self.CopyOriginDir,
                    toFolder=self.WorkingDir,
                    fileRefs=Preprocess_Mask.StrToLst(
                        _widget_text(getattr(self.ui, 'fileRefs_Txt_BSRJZ', None))
                    )["listOut"],
                    numberLimit=_widget_integer(
                        getattr(self.ui, 'numberLimit_Txt_BSRJZ', None), 20
                    ),
                )
                self.UpdateMsgLog('Copy Files Done')

            if self.ScanFileFlag:
                self.UpdateMsgLog('Scan DCM folders start')
                self.ScanDCMDir()
            else:
                self.UpdateMsgLog('Scan CTA folders start')
                self.ScanCTADir()
            self.UpdateMsgLog(
                'Found {0} case folders'.format(len(self.DCMDirList))
            )

            if self.CreateFolderFlag:
                self.CreateFoldersDCM()
            self.StoreListInformation()
            self._initialize_case_records()

            if self.MaskXsectionOnlyFlag and self.LCFlag:
                self.UpdateMsgLog(
                    'Mask xsections only: CTA xsections will not be generated. '
                    'Lumen Correction will use existing valid CTA xsections; '
                    'arteries without one will be recorded incomplete.'
                )

            if self.BranchDictFlag:
                self.UpdateMsgLog('Write artery dictionary start')
                try:
                    self.CheckWriteDict()
                except Exception as error:
                    self._branch_validation_attempted = True
                    self._validated_use_paths = {}
                    self.UpdateMsgLog(
                        'Artery dictionary generation failed; stale generated '
                        'JSON files will not be reused: {0}'.format(error)
                    )
                self.UpdateMsgLog('Write artery dictionary done')

            if self.GenerateTFlag:
                self.UpdateMsgLog('Generate tables start')
                self.GenerateTable(
                    resampleDepth=_widget_text(
                        getattr(self.ui, 'resampleDepthLineTxt_BSRJZ', None)
                    ) or '0.1',
                    cpus=_widget_text(
                        getattr(self.ui, 'cpusLineTxt_BSRJZ', None)
                    ) or '2',
                    maskXsection=self.MaskXsectionFlag,
                    maskXsectionOnly=self.MaskXsectionOnlyFlag,
                )
                self.UpdateMsgLog('Generate tables done')

            if self.FMFlag:
                self.UpdateMsgLog('Filter Mask start')
                self.BatchFilterMask()
                self.UpdateMsgLog('Filter Mask done')

            if self.XsecFlag:
                self.UpdateMsgLog('Xsection start')
                self.BatchXsection()
                self.UpdateMsgLog('Xsection done')

            if self.LCFlag:
                self.UpdateMsgLog('Lumen Correction start')
                process_snapshot_error = ''
                try:
                    self._prepare_lumen_process()
                except Exception as error:
                    process_snapshot_error = _clean_error_message(error)
                    self.UpdateMsgLog(
                        'Could not snapshot the correction process table; '
                        'individual corrections will be recorded incomplete: '
                        '{0}'.format(error)
                    )
                self.BatchLumenCorrection(
                    process_snapshot_error=process_snapshot_error
                )
                self.UpdateMsgLog('Lumen Correction done')

            self._merge_xsection_records()
        summary = {
            'case_count': len(self.DCMDirList),
            'record_path': self.XsectionRecordPath,
            'manifest_path': self.ManifestPath,
            'failed_root': str(
                Path(self.FailedRootPath) / FAILED_XSECTION_FOLDER_NAME
            ),
            'duration_seconds': round(time.monotonic() - started_at, 3),
        }
        self.UpdateMsgLog(
            'Batch Slice Radiomics finished: {0} cases in {1} seconds'.format(
                summary['case_count'], summary['duration_seconds']
            )
        )
        return summary

    def UpdateMsgLog(self, msg=""):
        # Date and time
        nowStr = datetime.now().strftime("%d/%b/%y - %H:%M:%S")
        disp = "##############" \
               + nowStr \
               + "############## \n" \
               + msg \
               + "\n############################\n"

        if self.modelui:
            # update log and display message
            self.modelui.plainTextEdit_Message.setPlainText(disp)
            self.modelui.plainTextEdit_Log.appendPlainText(disp)
        print(disp)


if __name__ == '__main__':
    raise SystemExit('Launch BatchSliceRadiomics from HemodynamicsPlatform.py')
