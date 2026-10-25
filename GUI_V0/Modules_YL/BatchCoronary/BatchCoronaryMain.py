import os
import sys
import csv
import hashlib
import pandas as pd
import re
import json
import string
import ntpath
import pyparsing
import time
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import numpy

os.chdir(os.getcwd())
sys.path.insert(0, '../Functions_JZ')
import Save_Load_File
import Preprocess_Mask
import Pd_Funs
# Import functions self-written
from Modules_YL.BatchCoronary import BranchDictionary


SEGMENTATION_CORE_STATUS_FIELDS = (
    'run_segmentation',
    'seg_exe_failed',
    'seg_exe_er_msg',
    'coronary_segment_failed',
    'complete_segmentation',
)
CENTERLINE_CSV_STATUS_FIELDS = (
    'run_centerline_csv_check',
    'centerline_csv_failed',
    'centerline_csv_er_msg',
    'complete_centerline_csv',
)
FILTER_MASK_STATUS_FIELDS = (
    'run_filter_mask',
    'filter_mask_failed',
    'filter_mask_er_msg',
    'complete_filter_mask',
)
SEGMENTATION_STATUS_DEFAULTS = {
    'run_segmentation': 0,
    'seg_exe_failed': 0,
    'seg_exe_er_msg': '',
    'coronary_segment_failed': 0,
    'complete_segmentation': 0,
    'run_centerline_csv_check': 0,
    'centerline_csv_failed': 0,
    'centerline_csv_er_msg': '',
    'complete_centerline_csv': 0,
    'run_filter_mask': 0,
    'filter_mask_failed': 0,
    'filter_mask_er_msg': '',
    'complete_filter_mask': 0,
}
SEGMENTATION_STATUS_FIELDS = (
    SEGMENTATION_CORE_STATUS_FIELDS
    + CENTERLINE_CSV_STATUS_FIELDS
    + FILTER_MASK_STATUS_FIELDS
)
SEGMENTATION_RECORD_FIELDS = (
    'case_id',
    'batch_name',
    'case_relative_path',
    'source_sha256',
    'centerline_json_source_sha256',
    'filter_mask_input_sha256',
) + SEGMENTATION_STATUS_FIELDS
SEGMENTATION_RECORD_FOLDER = 'seg_record'
SEGMENTATION_RECORD_FILE = 'batch_seg.csv'
SEGMENTATION_MASK_RELATIVE_PATH = Path('Preproc') / 'OriData' / 'sc_labelled_coronary_segment.nii.gz'
DEFAULT_SEGMENTATION_TIMEOUT_MINUTES = 20
DEFAULT_MANIFEST_FILE_NAME = 'BatchNIIDataPrepare_manifest.csv'
FAILED_SEGMENTATION_FOLDER_NAME = '0_Failed_Segmentation'
LOCK_POLL_SECONDS = 0.1
LOCK_TIMEOUT_SECONDS = 120
MAX_SEGMENTATION_ERROR_LENGTH = 4000
HASH_BLOCK_SIZE = 1024 * 1024
CENTERLINE_SIDES = ('L', 'R')
CENTERLINE_JSON_NAME = '{0}_coronary_centerline_network.json'
CENTERLINE_AGGREGATE_CSV_NAME = '{0}_coronary_centerline_network.csv'
FILTER_MASK_REQUIRED_COLUMNS = (
    'MaskPath', 'MaskVal', 'NewMaskVal', 'OutputFolder', 'FileNameRef'
)
GENERATED_TABLE_ATTRIBUTES = (
    'FilterMaskTable',
    'XsectionTable',
    'XsectionFixTable',
    'XsectionFixExtTable',
    'OriVolumeTable',
    'SSTable',
    'SGTable',
    'AVCTable',
    'OriVolumeFixTable',
    'CFD_MeshTable',
    'CFD_STLcutTable',
    'CFD_STLmeshTable',
    'CFD_FindFaceTable',
    'CFD_GenerateInTable',
    'CFD_PostProgressTable',
    'DataOutputTable',
    'PostNpyTable',
    'Map_ResultTable',
    'FFRCalculationTable',
    'ParaviewTable',
    'SaveVtuTable',
)


def _clean_error_message(message, limit=MAX_SEGMENTATION_ERROR_LENGTH):
    """Return a compact, single-line error suitable for a CSV cell."""

    message = ' '.join(str(message or '').replace('\x00', '').split())
    if len(message) <= limit:
        return message
    marker = ' ...[truncated]... '
    if limit <= len(marker) + 2:
        return message[:limit]
    prefix_length = (limit - len(marker)) // 2
    suffix_length = limit - len(marker) - prefix_length
    return message[:prefix_length] + marker + message[-suffix_length:]


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
        return default
    value_method = getattr(widget, 'value', None)
    if not callable(value_method):
        return default
    try:
        return int(value_method())
    except (TypeError, ValueError):
        return default


def is_complete_segmentation(case_path):
    """Return whether the requested coronary segmentation exists and is nonempty."""

    mask_path = Path(case_path) / SEGMENTATION_MASK_RELATIVE_PATH
    try:
        return mask_path.is_file() and mask_path.stat().st_size > 0
    except OSError:
        return False


def _sha256_file(file_path):
    digest = hashlib.sha256()
    with Path(file_path).open('rb') as file_object:
        while True:
            block = file_object.read(HASH_BLOCK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _hash_text_parts(parts):
    digest = hashlib.sha256()
    for part in parts:
        encoded_part = str(part).encode('utf-8')
        digest.update(str(len(encoded_part)).encode('ascii'))
        digest.update(b':')
        digest.update(encoded_part)
        digest.update(b'\n')
    return digest.hexdigest()


def _is_nonempty_regular_file(file_path):
    file_path = Path(file_path)
    try:
        return file_path.is_file() and file_path.stat().st_size > 0
    except OSError:
        return False


def validate_centerline_json(json_path):
    """Load and validate one vendor centerline JSON document."""

    json_path = Path(json_path)
    if not _is_nonempty_regular_file(json_path):
        raise ValueError('missing or empty centerline JSON: {0}'.format(json_path))
    try:
        with json_path.open('r', encoding='utf-8-sig') as json_file:
            centerlines = json.load(json_file)
    except Exception as error:
        raise ValueError(
            'could not read centerline JSON {0}: {1}'.format(json_path, error)
        )
    if not isinstance(centerlines, list) or not centerlines:
        raise ValueError(
            'centerline JSON must contain a nonempty list: {0}'.format(json_path)
        )

    branch_names = set()
    for branch_index, branch in enumerate(centerlines):
        if not isinstance(branch, dict):
            raise ValueError(
                'centerline branch {0} is not an object: {1}'.format(
                    branch_index, json_path
                )
            )
        branch_name = branch.get('branch_name')
        if not isinstance(branch_name, str) or not branch_name.strip():
            raise ValueError(
                'centerline branch {0} has no branch_name: {1}'.format(
                    branch_index, json_path
                )
            )
        # Vendor JSON intentionally distinguishes names by case.  For example,
        # ``LCx`` is an anatomical branch while ``LCX`` is an aggregate branch.
        # Reject only an exact duplicate after trimming surrounding whitespace.
        normalized_name = branch_name.strip()
        if normalized_name in branch_names:
            raise ValueError(
                'duplicate centerline branch {0}: {1}'.format(branch_name, json_path)
            )
        branch_names.add(normalized_name)
        coordinates = branch.get('coordinates')
        if not isinstance(coordinates, list):
            raise ValueError(
                'coordinates for branch {0} are not a list: {1}'.format(
                    branch_name, json_path
                )
            )
        for coordinate_index, coordinate in enumerate(coordinates):
            if not isinstance(coordinate, dict):
                raise ValueError(
                    'coordinate {0} for branch {1} is not an object: {2}'.format(
                        coordinate_index, branch_name, json_path
                    )
                )
            for axis in ('X', 'Y', 'Z'):
                value = coordinate.get(axis)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        'coordinate {0} for branch {1} has invalid {2}: {3}'.format(
                            coordinate_index, branch_name, axis, json_path
                        )
                    )
        if branch_name in ('.', '..') or '/' in branch_name or '\\' in branch_name:
            raise ValueError(
                'centerline branch name is not a safe filename: {0}'.format(
                    branch_name
                )
            )
    return centerlines


def resolve_centerline_json_source(case_path, side):
    """Resolve a valid ``.json`` or ``.json.txt`` source without guessing."""

    if side not in CENTERLINE_SIDES:
        raise ValueError('Unsupported centerline side: {0}'.format(side))
    ori_data_path = Path(case_path) / 'Preproc' / 'OriData'
    canonical_name = CENTERLINE_JSON_NAME.format(side)
    candidates = (
        ori_data_path / canonical_name,
        ori_data_path / (canonical_name + '.txt'),
    )
    valid_candidates = []
    candidate_errors = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = validate_centerline_json(candidate)
            normalized_data = json.dumps(
                data, sort_keys=True, separators=(',', ':'), ensure_ascii=False
            )
            valid_candidates.append((candidate, data, normalized_data))
        except Exception as error:
            candidate_errors.append(str(error))

    if not valid_candidates:
        detail = '; '.join(candidate_errors) or 'no source file found'
        raise ValueError(
            '{0} centerline JSON source is unavailable: {1}'.format(side, detail)
        )
    normalized_documents = {item[2] for item in valid_candidates}
    if len(normalized_documents) > 1:
        raise ValueError(
            'conflicting {0} centerline .json and .json.txt sources'.format(side)
        )
    return valid_candidates[0]


def _copy_file_atomic(source_path, destination_path):
    source_path = Path(source_path)
    destination_path = Path(destination_path)
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
        if _sha256_file(temporary_path) != _sha256_file(source_path):
            raise IOError(
                'centerline JSON copy verification failed: {0}'.format(source_path)
            )
        os.replace(str(temporary_path), str(destination_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def centerline_json_source_fingerprint(case_path):
    resolved = []
    for side in CENTERLINE_SIDES:
        source_path, data, normalized_data = resolve_centerline_json_source(
            case_path, side
        )
        resolved.append((side, source_path, data, normalized_data))
    fingerprint = _hash_text_parts(
        [item[0] + ':' + item[3] for item in resolved]
    )
    return fingerprint, resolved


def stage_centerline_json_pair(case_path):
    """Publish both validated source JSONs under their canonical CSV names."""

    fingerprint, resolved = centerline_json_source_fingerprint(case_path)
    csv_path = Path(case_path) / 'Preproc' / 'CSV'
    csv_path.mkdir(parents=True, exist_ok=True)
    temporary_paths = {}
    backup_paths = {}
    published_paths = []
    publication_complete = False
    try:
        # Prepare and validate both copies before either canonical file changes.
        for side, source_path, _data, _normalized_data in resolved:
            destination_path = csv_path / CENTERLINE_JSON_NAME.format(side)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix='.{0}.'.format(destination_path.name),
                suffix='.tmp',
                dir=str(csv_path),
            )
            os.close(file_descriptor)
            temporary_path = Path(temporary_name)
            shutil.copy2(str(source_path), str(temporary_path))
            if _sha256_file(temporary_path) != _sha256_file(source_path):
                raise IOError(
                    'centerline JSON copy verification failed: {0}'.format(
                        source_path
                    )
                )
            validate_centerline_json(temporary_path)
            temporary_paths[destination_path] = temporary_path

        # Preserve both old files so a failure publishing the second side can
        # roll the pair back to its exact prior state.
        for destination_path in temporary_paths:
            if not destination_path.exists():
                continue
            if not destination_path.is_file():
                raise ValueError(
                    'Existing centerline JSON destination is not a file: {0}'
                    .format(destination_path)
                )
            file_descriptor, backup_name = tempfile.mkstemp(
                prefix='.{0}.'.format(destination_path.name),
                suffix='.bak',
                dir=str(csv_path),
            )
            os.close(file_descriptor)
            backup_path = Path(backup_name)
            backup_path.unlink()
            os.replace(str(destination_path), str(backup_path))
            backup_paths[destination_path] = backup_path

        for destination_path, temporary_path in temporary_paths.items():
            os.replace(str(temporary_path), str(destination_path))
            published_paths.append(destination_path)
        publication_complete = True
    except Exception as publication_error:
        restoration_errors = []
        for destination_path in published_paths:
            try:
                if destination_path.exists():
                    destination_path.unlink()
            except OSError as error:
                restoration_errors.append(str(error))
        for destination_path, backup_path in backup_paths.items():
            if backup_path.exists():
                try:
                    os.replace(str(backup_path), str(destination_path))
                except OSError as error:
                    restoration_errors.append(
                        '{0} remains at {1}: {2}'.format(
                            destination_path, backup_path, error
                        )
                    )
        if restoration_errors:
            raise RuntimeError(
                'Centerline JSON publication failed ({0}); rollback also '
                'reported: {1}'.format(
                    publication_error, '; '.join(restoration_errors)
                )
            )
        raise
    finally:
        for temporary_path in temporary_paths.values():
            if temporary_path.exists():
                temporary_path.unlink()
    if publication_complete:
        for backup_path in backup_paths.values():
            try:
                if backup_path.exists():
                    backup_path.unlink()
            except OSError:
                # The new pair is already verified and published. Retaining an
                # old backup is safer than turning cleanup into data loss.
                pass
    return fingerprint


def _validate_aggregate_centerline_csv(csv_path):
    csv_path = Path(csv_path)
    if not _is_nonempty_regular_file(csv_path):
        raise ValueError('missing or empty aggregate centerline CSV: {0}'.format(csv_path))
    try:
        with csv_path.open('r', encoding='utf-8-sig', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = set(reader.fieldnames or ())
            if not {'branch_name', 'coordinates'}.issubset(fieldnames):
                raise ValueError(
                    'aggregate CSV is missing branch_name/coordinates columns'
                )
            first_row = next(reader, None)
            if first_row is None:
                raise ValueError('aggregate CSV has no data rows')
    except ValueError:
        raise
    except Exception as error:
        raise ValueError('could not read aggregate CSV: {0}'.format(error))


def _validate_branch_coordinate_csv(csv_path):
    csv_path = Path(csv_path)
    if not _is_nonempty_regular_file(csv_path):
        raise ValueError('missing or empty branch coordinate CSV: {0}'.format(csv_path))
    try:
        with csv_path.open('r', encoding='utf-8-sig', newline='') as csv_file:
            reader = csv.reader(csv_file)
            row_count = 0
            for row in reader:
                if len(row) < 3:
                    raise ValueError('branch coordinate CSV has an invalid XYZ row')
                for value in row[:3]:
                    float(value)
                row_count += 1
        if row_count == 0:
            raise ValueError('branch coordinate CSV has no XYZ data')
    except ValueError:
        raise
    except Exception as error:
        raise ValueError('could not read branch coordinate CSV: {0}'.format(error))


def inspect_centerline_csv_case(case_path):
    """Return semantic readiness of one case's complete centerline CSV output."""

    case_path = Path(case_path)
    csv_path = case_path / 'Preproc' / 'CSV'
    try:
        if not csv_path.is_dir():
            raise ValueError('Preproc/CSV folder is missing')
        for side in CENTERLINE_SIDES:
            json_path = csv_path / CENTERLINE_JSON_NAME.format(side)
            centerlines = validate_centerline_json(json_path)
            _validate_aggregate_centerline_csv(
                csv_path / CENTERLINE_AGGREGATE_CSV_NAME.format(side)
            )
            for branch in centerlines:
                if not branch['coordinates']:
                    continue
                branch_csv_path = csv_path / ('{0}.csv'.format(branch['branch_name']))
                if not _path_is_within(branch_csv_path, csv_path):
                    raise ValueError(
                        'branch CSV path escapes Preproc/CSV: {0}'.format(
                            branch_csv_path
                        )
                    )
                _validate_branch_coordinate_csv(branch_csv_path)
        return {'complete': True, 'error_message': ''}
    except Exception as error:
        return {
            'complete': False,
            'error_message': _clean_error_message(error),
        }


def is_complete_centerline_csv(case_path):
    return bool(inspect_centerline_csv_case(case_path)['complete'])


def _filter_output_file_paths(output_folder, file_name_reference):
    safe_characters = set(string.ascii_letters + string.digits + '~ -_.')
    head, tail = ntpath.split(str(file_name_reference))
    leaf_name = tail or ntpath.basename(head)
    paths = []
    for suffix in ('NVl', 'Val'):
        safe_name = ''.join(
            character for character in leaf_name + suffix
            if character in safe_characters
        )
        paths.append(Path(output_folder) / (safe_name + '.nii.gz'))
    return tuple(paths)


def _normalized_path_key(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def _canonical_filter_value(value):
    if pd.isna(value):
        raise ValueError('Filter Mask table contains a blank value')
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _load_filter_mask_case_plans(
        filter_table_path,
        case_paths,
        warning_messages=None):
    """Validate and group one generated FilterMask table by exact case output."""

    filter_table_path = Path(filter_table_path)
    if not _is_nonempty_regular_file(filter_table_path):
        raise ValueError(
            'Filter Mask table is missing or empty: {0}'.format(filter_table_path)
        )
    try:
        dataframe = pd.read_csv(str(filter_table_path))
    except Exception as error:
        raise ValueError('Could not read Filter Mask table: {0}'.format(error))
    missing_columns = [
        column for column in FILTER_MASK_REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            'Filter Mask table is missing columns: {0}'.format(
                ', '.join(missing_columns)
            )
        )

    case_by_output = {}
    case_path_by_key = {}
    plans = {}
    for case_path_text in case_paths:
        case_path = Path(case_path_text)
        case_key = _normalized_path_key(case_path)
        output_key = _normalized_path_key(
            case_path / 'Preproc' / 'OriData' / 'Branches'
        )
        if output_key in case_by_output:
            raise ValueError('Duplicate case output folder: {0}'.format(output_key))
        case_by_output[output_key] = case_key
        case_path_by_key[case_key] = case_path
        plans[case_key] = []

    unknown_output_folders = []
    for row_index, pandas_row in dataframe.iterrows():
        row = {
            column: pandas_row[column]
            for column in FILTER_MASK_REQUIRED_COLUMNS
        }
        if pd.isna(row['OutputFolder']):
            unknown_output_folders.append(
                'row {0}: <blank>'.format(row_index + 2)
            )
            continue
        output_folder_text = str(row['OutputFolder']).strip()
        if not output_folder_text:
            unknown_output_folders.append(
                'row {0}: <blank>'.format(row_index + 2)
            )
            continue

        output_key = _normalized_path_key(output_folder_text)
        case_key = case_by_output.get(output_key)
        if case_key is None:
            unknown_output_folders.append(output_folder_text)
            continue
        row['_row_number'] = row_index + 2
        plans[case_key].append(row)

    if unknown_output_folders:
        warning = (
            'Filter Mask ignored rows whose output folders are outside the '
            'current cases: {0}'.format(
                ', '.join(sorted(set(unknown_output_folders)))
            )
        )
        if warning_messages is not None:
            warning_messages.append(warning)

    return plans


def _filter_mask_case_fingerprint(case_path, rows):
    if not rows:
        raise ValueError('Filter Mask table has no rows for this case')
    case_path = Path(case_path)
    canonical_rows = []
    expected_paths = []
    expected_path_keys = set()
    input_hashes = {}
    for row in rows:
        row_number = row.get('_row_number', '?')
        for column in FILTER_MASK_REQUIRED_COLUMNS:
            if column not in row or pd.isna(row[column]):
                raise ValueError(
                    'Filter Mask row {0} has a blank {1}'.format(
                        row_number, column
                    )
                )
        if (not isinstance(row['FileNameRef'], str)
                or not row['FileNameRef'].strip()):
            raise ValueError(
                'Filter Mask row {0} has an invalid FileNameRef'.format(
                    row_number
                )
            )
        mask_path = Path(str(row['MaskPath']).strip())
        if not _path_is_within(mask_path, case_path):
            raise ValueError(
                'Filter Mask input is outside case {0}: {1}'.format(
                    case_path.name, mask_path
                )
            )
        if not _is_nonempty_regular_file(mask_path):
            raise ValueError(
                'Filter Mask input is missing or empty: {0}'.format(mask_path)
            )
        mask_key = _normalized_path_key(mask_path)
        if mask_key not in input_hashes:
            input_hashes[mask_key] = _sha256_file(mask_path)
        row_expected_paths = _filter_output_file_paths(
            row['OutputFolder'], row['FileNameRef']
        )
        for expected_path in row_expected_paths:
            if not _path_is_within(expected_path, case_path):
                raise ValueError(
                    'Filter Mask output is outside case {0}: {1}'.format(
                        case_path.name, expected_path
                    )
                )
            expected_key = _normalized_path_key(expected_path)
            if expected_key in expected_path_keys:
                raise ValueError(
                    'Filter Mask rows for {0} produce the same output: {1}'
                    .format(case_path.name, expected_path)
                )
            expected_path_keys.add(expected_key)
            expected_paths.append(expected_path)
        canonical_rows.append({
            'MaskPath': mask_key,
            'MaskVal': _canonical_filter_value(row['MaskVal']),
            'NewMaskVal': _canonical_filter_value(row['NewMaskVal']),
            'OutputFolder': _normalized_path_key(row['OutputFolder']),
            'FileNameRef': str(row['FileNameRef']),
        })

    fingerprint_parts = [
        json.dumps(row, sort_keys=True, separators=(',', ':'))
        for row in sorted(
            canonical_rows,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(',', ':')),
        )
    ]
    fingerprint_parts.extend(
        'input:{0}:{1}'.format(path_key, input_hashes[path_key])
        for path_key in sorted(input_hashes)
    )
    return _hash_text_parts(fingerprint_parts), expected_paths


def _filter_mask_outputs_complete(expected_paths):
    return all(_is_nonempty_regular_file(path) for path in expected_paths)


def _path_is_within(path, parent):
    """Return whether path resolves inside parent (Python 3.7 compatible)."""

    path = os.path.normcase(os.path.abspath(str(path)))
    parent = os.path.normcase(os.path.abspath(str(parent)))
    try:
        return os.path.commonpath([path, parent]) == parent
    except ValueError:
        return False


@contextmanager
def interprocess_file_lock(lock_path, timeout_seconds=LOCK_TIMEOUT_SECONDS):
    """Hold a one-byte OS lock, released automatically if the process exits."""

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open('a+b')
    acquired = False
    try:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b'0')
            lock_file.flush()
            os.fsync(lock_file.fileno())

        deadline = time.time() + timeout_seconds
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
                if time.time() >= deadline:
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


def _write_csv_atomic(csv_path, fieldnames, rows):
    """Publish a complete CSV without exposing a partially written file."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix='.{0}.'.format(csv_path.name),
        suffix='.tmp',
        dir=str(csv_path.parent),
        text=True,
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
            csv_file.flush()
            os.fsync(csv_file.fileno())
        os.replace(str(temporary_path), str(csv_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_dataframe_atomic(dataframe, csv_path):
    """Atomically publish a pandas DataFrame as a CSV file."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix='.{0}.'.format(csv_path.name),
        suffix='.tmp',
        dir=str(csv_path.parent),
        text=True,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        dataframe.to_csv(str(temporary_path), index=False)
        os.replace(str(temporary_path), str(csv_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_csv_rows(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return [], []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader.fieldnames or []), list(reader)


def write_batch_segmentation_records(record_path, records, lock_timeout=LOCK_TIMEOUT_SECONDS):
    """Atomically write the single, cumulative segmentation ledger for a batch."""

    record_path = Path(record_path)
    ordered_records = sorted(
        records,
        key=lambda row: (str(row.get('batch_name', '')).casefold(),
                         str(row.get('case_id', '')).casefold()),
    )
    with interprocess_file_lock(str(record_path) + '.lock', lock_timeout):
        _write_csv_atomic(record_path, SEGMENTATION_RECORD_FIELDS, ordered_records)
    return record_path


def merge_batch_segmentation_records(
        manifest_path,
        initial_root,
        lock_timeout=LOCK_TIMEOUT_SECONDS):
    """Merge every available per-batch ledger into the shared manifest safely."""

    manifest_path = Path(manifest_path)
    initial_root = Path(initial_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with interprocess_file_lock(str(manifest_path) + '.lock', lock_timeout):
        manifest_fields, manifest_rows = _read_csv_rows(manifest_path)
        if not manifest_fields:
            manifest_fields = ['case_id', 'batch_name', 'case_relative_path']
        else:
            missing_keys = [
                field_name
                for field_name in ('case_id', 'batch_name')
                if field_name not in manifest_fields
            ]
            if missing_keys:
                raise ValueError(
                    'Manifest is missing required identity columns {0}: {1}'.format(
                        ', '.join(missing_keys), manifest_path
                    )
                )
            if 'case_relative_path' not in manifest_fields:
                manifest_fields.append('case_relative_path')
        for field_name in SEGMENTATION_STATUS_FIELDS:
            if field_name not in manifest_fields:
                manifest_fields.append(field_name)

        row_by_key = {}
        for row in manifest_rows:
            key = (str(row.get('batch_name', '')).casefold(),
                   str(row.get('case_id', '')).casefold())
            if all(key) and key not in row_by_key:
                row_by_key[key] = row

        record_paths = sorted(
            initial_root.glob('batch_*/Table/{0}/{1}'.format(
                SEGMENTATION_RECORD_FOLDER, SEGMENTATION_RECORD_FILE)),
            key=lambda path: str(path).casefold(),
        )
        merged_record_count = 0
        for record_path in record_paths:
            with interprocess_file_lock(str(record_path) + '.lock', lock_timeout):
                _, batch_records = _read_csv_rows(record_path)
            for record in batch_records:
                case_id = str(record.get('case_id', '')).strip()
                batch_name = str(record.get('batch_name', '')).strip()
                if not case_id or not batch_name:
                    continue
                key = (batch_name.casefold(), case_id.casefold())
                row = row_by_key.get(key)
                if row is None:
                    row = {field_name: '' for field_name in manifest_fields}
                    row['case_id'] = case_id
                    row['batch_name'] = batch_name
                    row['case_relative_path'] = record.get('case_relative_path', '')
                    manifest_rows.append(row)
                    row_by_key[key] = row
                for field_name in SEGMENTATION_STATUS_FIELDS:
                    row[field_name] = record.get(field_name, '')
                merged_record_count += 1

        _write_csv_atomic(manifest_path, manifest_fields, manifest_rows)

    return {
        'manifest_path': manifest_path,
        'record_count': merged_record_count,
        'row_count': len(manifest_rows),
    }


class BatchCoronary:
    def __init__(self, UI=None, Hedys=None):
        self.ui = None
        if UI:
            self.ui = UI
            self.ui.LoadFolderPath_Btn_BPP.clicked.connect(lambda: self.ChooseWorkingDir())
            self.ui.ChooseCGExePathBtn_1_BPP.clicked.connect(lambda: self.ChooseCGExePath())
            self.ui.ChooseCPRExePathBtn_2_BPP.clicked.connect(lambda: self.ChooseCPRExePath())
            self.ui.ChooseSTLExePath_2_BPP.clicked.connect(lambda: self.ChooseSTLExePath())
            self.ui.ChooseLumenCorrectionPathBtn_BPP.clicked.connect(lambda: self.ChooseLumenCorretionPath())
            self.ui.ChooseOriVolumeCorrectionPathBtn_BPP.clicked.connect(lambda: self.ChooseOriVolumeCorretionPath())
            self.ui.ChooseAVCPathBtn_BPP.clicked.connect(lambda: self.ChooseAVCPath())
            self.ui.ChooseBloodPressureAndHeartRatePathBtn_BPP.clicked.connect(
                lambda: self.ChooseBloodPressureAndHeartRatePath())
            self.ui.MeshExePathBtn_BPP.clicked.connect(lambda: self.ChooseMeshExePath())
            self.ui.ChooseAUIPathBtn_BPP.clicked.connect(lambda: self.ChooseAUIExePath())
            self.ui.ChoosesetPathBtn_BPP.clicked.connect(lambda: self.ChooseSetCsvPath())
            manifest_button = getattr(self.ui, 'ChooseManifestPathBtn_BPP', None)
            if manifest_button is not None:
                manifest_button.clicked.connect(lambda: self.ChooseManifestPath())
            failed_root_button = getattr(self.ui, 'ChooseFailedRootPathBtn_BPP', None)
            if failed_root_button is not None:
                failed_root_button.clicked.connect(lambda: self.ChooseFailedRootPath())
            self.ui.RunBtn_CPP.clicked.connect(lambda: self.mainFunction())

        self.modelui = None
        if Hedys:
            self.modelui = Hedys.ui
            self.model = Hedys

        # initial set up
        self.initProcess()

    def initProcess(self):
        if self.modelui: self.modelui.QStackedWidget_Module.setCurrentWidget(self.ui.centralwidget)

        # create & clean all variables
        self.WorkingDir = None
        self.ManifestPath = None
        self.FailedRootPath = None
        self.InitialSegmentationRoot = None
        self.BatchName = None
        self.SegmentationTimeoutSeconds = DEFAULT_SEGMENTATION_TIMEOUT_MINUTES * 60
        self.BatchSegmentationRecordPath = None
        self.BranchDictionaryReadyCSVPaths = None
        self.CenterlineCsvOriginalCasePaths = []
        self.CenterlineCsvSkipPaths = set()
        self.CenterlineCsvSourceHashes = {}
        self.CenterlineJsonStageErrors = {}
        self.DCMDirList = []  # DCM list
        self.CTADirList = []  # store the cta location, include file name

        # copy CTA variables
        self.CTANewFilePathList = None
        self.CTANewPathList = None

        # copy centerline Json
        self.leftCenterlineJsonList = []  # centerline L json file positions
        self.rightCenterlineJsonList = []  # centerline R json file positions

        self.CTAPathList = []  # store the cta location, exclude file name
        self.XsecFixCTAPathList = []  # store the cta located in XsectionFix folder, include the file name
        self.OriVolCTAPathList = []  # store the cta located in OriVolume folder, include the file name
        self.CSVPathList = []  # store the csv directory path lists
        self.OriDataPathList = []
        self.BranchPathList = []  # store the absolute paths of branch folder list
        self.AortaPathList = []  # store the absolute paths of aorta mask file list
        self.AortaFixedPathList = []  # store the list of absolute paths list of aorta mask after lumen correction
        self.NewAortaPathList = []  # store the new absolute paths of aorta mask file list
        self.XsectionPathList = []  # store the absolute paths of Xsection folder list
        self.XsectionFixPathList = []  # store the absolute paths of XsectionFix folder list
        self.XsectionFixExtPathList = []  # store the absolute paths of XsectionFix extension folder list
        self.OriVolumePathList = []  # store the absolute paths of OriVolume folder list
        self.FinalMaskPathList = []  # store the absolute paths of final mask file path list
        self.StlFilePathList = []  # store the absolute paths of stl file path list
        self.ArVolCalPathList = []  # Murray's law results

        self.DictionaryPath = None  # store the absolute path of branch that required
        self.BatchDCMTablePath = None  # store the absolute path of branch that required
        self.JasonUpdateTablePath = None  # store the absolute path of Jason update
        self.FilterMaskTable = []  # store the columns of filter mask single branch
        self.FilterMaskTablePath = None
        self.XsectionTable = []
        self.XsectionTablePath = None
        self.XsectionFixTable = []
        self.XsectionFixTablePath = None
        self.LCProcessPath = None
        self.AortaFixTablePath = None  # store the absolute path of Aorta fix table
        self.XsectionFixExtTable = []
        self.XsectionFixExtTablePath = None
        self.OriVolumeTable = []
        self.OriVolumeTablePath = None
        self.SSTable = []
        self.SSTablePath = None
        self.SGTable = []  #
        self.SGTablePath = None  # store the stl generate table path
        self.AVCTable = []  # Murray's Law batch
        self.AVCTablePath = None
        self.OCProcessPath = None
        self.OriVolumeFixTable = []
        self.OriVolumeFixTablePath = None
        self.CFD_MeshTable = []
        self.CFD_MeshTablePath = None
        self.CFD_STLcutTable = []
        self.CFD_STLcutTablePath = None
        self.CFD_STLmeshTable = []
        self.CFD_STLmeshTablePath = None
        self.CFD_FindFaceTable = []
        self.CFD_FindFaceTablePath = None
        self.CFD_setTablePath = None
        self.CFD_GenerateInTable = []
        self.CFD_GenerateInTablePath = None
        self.CFD_PostProgressTable = []
        self.CFD_PostProgressTablePath = None
        self.DataOutputTable = []
        self.DataOutputTablePath = None
        self.PostNpyTable = []
        self.PostNpyTablePath = None
        self.Map_ResultTable = []
        self.Map_ResultTablePath = None
        self.FFRCalculationTable = []
        self.FFRCalculationTablePath = None
        self.ParaviewTable = []
        self.ParaviewTablePath = None
        self.SaveVtuTable = []
        self.SaveVtuTablePath = None

        self.STLFolderList = []  # store the absolute paths of STL folder list
        self.MeshFolderList = []
        self.MeshFilePathList = []
        self.TableDir = None

        # self.CPREXEPath = 'D:\CPP\CPR_NoCSV_16Jul21\CPRProject.exe' # *****need to change
        self.CPREXEPath = None  # *****need to change
        self.STLEXEPath = None  # ****need to change
        self.CGEXEPath = None

        # status that if certain process executed
        self.GTFlag = True  # generate table flag
        self.D2NFlag = True  # dicom to nifti flag
        self.CGFlag = True  # centerline generation flag
        self.FMFlag = True  # filter mask flag
        self.XsecFlag = True  # Xsection Flag
        self.LCFlag = True  # lumen correction flag
        self.ExtFlag = True  # Extension Flag
        self.XsecOriFlag = True  # Xsection Orivolume flag
        self.SSFlag = True  # stack segmentation flag
        self.M2SFlag = True  # mask 2 stl flag
        self.ArVolCalFlag = True  # area volume calculation
        self.CenterlineEFlag = True  # centerline extraction
        self.GenerateTFlag = True
        self.BranchDictFlag = True
        self.OriVolumeFixFlag = True
        self.getFFRresult = True
        self.MeshCut = True
        self.Mesh = True
        self.FindFace = True
        self.GenerateInput = True
        self.DataOutput = True
        self.CFD_PostProcessing = True
        self.MapResult = True
        self.FFRcalculation = True
        self.Paraview_processing = True
        self.SaveVTU = True
        self.VisualizationPyVista = True
        self.VTUdataExtraction = True
        self.VisualizationParaview = True

    def timmer(func):
        def warpper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            stop_time = time.time()
            print('{} run time is {}'.format(func.__name__, stop_time - start_time))
            return result

        return warpper

    def ChooseWorkingDir(self):
        dirname = Save_Load_File.OpenDirPathQt(dispMsg='Working Directory',
                                               fileObj=self.ui,
                                               qtObj=True
                                               )
        self.ui.LoadFolderPath_Txt_BPP.setPlainText(dirname)

    def ChooseManifestPath(self):
        filename = Save_Load_File.SaveFilePathQT(
            dispMsg='Choose or create segmentation manifest CSV',
            fileTypes='CSV files (*.csv);;All files (*.*)',
            fileObj=self.ui,
            qtObj=True,
        )
        if filename:
            self.ui.ManifestPath_Txt_BPP.setPlainText(filename)

    def ChooseFailedRootPath(self):
        dirname = Save_Load_File.OpenDirPathQt(
            dispMsg='Choose failed segmentation output root',
            fileObj=self.ui,
            qtObj=True,
        )
        if dirname:
            self.ui.FailedRootPath_Txt_BPP.setPlainText(dirname)

    # used to choose CG exe path
    def ChooseCGExePath(self):
        filename = Save_Load_File.OpenFilePathQt(
            dispMsg='Load Centerline Generation Exe file',
            fileTypes='All files (*.*);; exe files(*.exe)',
            fileObj=self.ui,
            qtObj=True
        )
        self.ui.CGExePath_Txt_BPP.setPlainText(filename)

    # used to choose Blood Pressure And Heart Rate path
    def ChooseBloodPressureAndHeartRatePath(self):
        filename = Save_Load_File.OpenFilePathQt(
            dispMsg='Load Centerline Generation Exe file',
            fileTypes='All files (*.*);; exe files(*.exe)',
            fileObj=self.ui,
            qtObj=True
        )
        self.ui.BloodPressureAndHeartRatePath_Txt_BPP.setPlainText(filename)

    # used to choose Mesh exe path
    def ChooseMeshExePath(self):
        filename = Save_Load_File.OpenFilePathQt(
            dispMsg='Load Centerline Generation Exe file',
            fileTypes='All files (*.*);; exe files(*.exe)',
            fileObj=self.ui,
            qtObj=True
        )
        self.ui.MeshExePath_Txt_BPP.setPlainText(filename)

    # used to choose AUI exe path
    def ChooseAUIExePath(self):
        filename = Save_Load_File.OpenFilePathQt(
            dispMsg='Load Centerline Generation Exe file',
            fileTypes='All files (*.*);; exe files(*.exe)',
            fileObj=self.ui,
            qtObj=True
        )
        self.ui.AUIPath_Txt_BPP.setPlainText(filename)

    # used to choose set path
    def ChooseSetCsvPath(self):
        filename = Save_Load_File.OpenFilePathQt(dispMsg="Choose csv file",
                                                 fileTypes="All files (*.*);; CSV files(*.csv)",
                                                 fileObj=self.ui, qtObj=True)
        self.ui.setPath_Txt_BPP.setPlainText(filename)

    # used to choose CPR exe path
    def ChooseCPRExePath(self):
        filename = Save_Load_File.OpenFilePathQt(
            dispMsg='Load CPR Exe file',
            fileTypes='All files (*.*);; exe files(*.exe)',
            fileObj=self.ui,
            qtObj=True
        )
        self.ui.CPRExePath_Txt_BPP.setPlainText(filename)

    # used to choose STL generation exe path
    def ChooseSTLExePath(self):
        filename = Save_Load_File.OpenFilePathQt(
            dispMsg='Load STL Generate Exe file',
            fileTypes='All files (*.*);; exe files(*.exe)',
            fileObj=self.ui,
            qtObj=True
        )
        self.ui.STLGExePath_Txt_BPP.setPlainText(filename)

    # used to choose lumen correction path
    def ChooseLumenCorretionPath(self):
        filename = Save_Load_File.OpenFilePathQt(
            dispMsg='Load Lumen Correction file',
            fileTypes='All files (*.*);; exe files(*.exe)',
            fileObj=self.ui,
            qtObj=True
        )
        self.ui.LumenCorrectionPath_Txt_BPP.setPlainText(filename)

    # used to choose lumen correction path
    def ChooseOriVolumeCorretionPath(self):
        filename = Save_Load_File.OpenFilePathQt(
            dispMsg='Load OriVolume Correction file',
            fileTypes='All files (*.*);; exe files(*.exe)',
            fileObj=self.ui,
            qtObj=True
        )
        self.ui.OriVolumePath_Txt_BPP.setPlainText(filename)

    # used to choose lumen correction path
    def ChooseAVCPath(self):
        # Save data
        dirname = Save_Load_File.OpenDirPathQt(
            dispMsg="Choose Murray's law information directory",
            fileObj=self.ui,
            qtObj=True
        )

        # set filename
        self.ui.AVCPath_Txt_BPP.setPlainText(dirname)

        # update message
        self.UpdateMsgLog(msg="Choose Murray's law information directory:\n{}".format(dirname))

    # if ui inserted, check box status to pend if certain process shall be running
    def CheckCheckBoxStatus(self):
        # preliminary settings
        self.CreateFolderFlag = self.ui.CreateFolders_CB_CPP.isChecked()
        self.CopyCTAFlag = self.ui.CopyCTA_CB_CPP.isChecked()
        self.CopyJSONFlag = self.ui.CopyJSON_CB_CPP.isChecked()

        # main module
        self.GTFlag = self.ui.GenerateTable_CB_CPP.isChecked()
        self.D2NFlag = self.ui.DCM2Nifti_CB_CPP.isChecked()
        self.CGFlag = self.ui.CG_CB_CPP.isChecked()
        self.FMFlag = self.ui.FM_CB_CPP.isChecked()
        self.XsecFlag = self.ui.Xsection_CB_CPP.isChecked()
        self.LCFlag = self.ui.LumenCorrection_CB_CPP.isChecked()
        self.ExtFlag = self.ui.Extension_CB_CPP.isChecked()
        self.XsecOriFlag = self.ui.XsecOriVolume_CB_CPP.isChecked()
        self.SSFlag = self.ui.SS_CB_CPP.isChecked()
        self.M2SFlag = self.ui.Mask2STL_CB_CPP.isChecked()
        self.ArVolCalFlag = self.ui.ArVolCal_CB_CPP.isChecked()
        self.CenterlineEFlag = self.ui.CenterlineE_CB_CPP.isChecked()
        self.UpdateJasonFlag = self.ui.UpdateJason_CB_CPP.isChecked()
        self.GenerateTFlag = self.ui.GenerateTable_CB_CPP.isChecked()
        self.BranchDictFlag = self.ui.BranchDict_CB_CPP.isChecked()
        self.OriVolumeFixFlag = self.ui.OriVolumeFix_CB_CPP.isChecked()
        self.getFFRresult = self.ui.getFFR_BPP.isChecked()
        self.getIFRresult = self.ui.getIFR_BPP.isChecked()
        self.MeshCut = self.ui.MeshCut_CB_CPP.isChecked()
        self.Mesh = self.ui.Mesh_CB_CPP.isChecked()
        self.FindFace = self.ui.FindFace_CB_CPP.isChecked()
        self.GenerateInput = self.ui.GenerateInput_CB_CPP.isChecked()
        self.DataOutput = self.ui.DataOutput_CB_CPP.isChecked()
        self.CFD_PostProcessing = self.ui.CFD_PostProcessing_CB_CPP.isChecked()
        self.MapResult = self.ui.MapResult_CB_CPP.isChecked()
        self.FFRcalculation = self.ui.FFRcalculation_CB_CPP.isChecked()
        self.Paraview_processing = self.ui.Paraviewprocessing_CB_CPP.isChecked()
        self.SaveVTU = self.ui.SaveVTU_CB_CPP.isChecked()
        self.VisualizationPyVista = self.ui.VisualizationPyVista_CB_CPP.isChecked()
        self.VTUdataExtraction = self.ui.VTUdataExtraction_CB_CPP.isChecked()
        self.VisualizationParaview = self.ui.VisualizationParaview_CB_CPP.isChecked()

    def ScanDCMDir(self):
        # get all folders that contain DCM
        for root, dirs, files in os.walk(self.WorkingDir):
            for file in files:
                if file.endswith('.dcm') or file.endswith('.DCM') or (".DCM" in file):
                    self.DCMDirList.append(root)
                    break

        self.DCMDirList.sort(key=lambda path: os.path.normcase(os.path.abspath(path)))
        # CTA path creation
        self.CTADirList = [
            os.path.join(x, 'CTA', 'CTA.nii.gz') for x in self.DCMDirList
        ]

    def CreateFoldersDCM(self):
        # create necessary folders under all folders
        for dir in self.DCMDirList:
            self.CreateFolders(CasePath=dir)

    @staticmethod
    def CreateSingleFolder(path):
        # only create folder when it does not exist
        try:
            os.mkdir(path)
        except FileExistsError:
            print(f"The directory {path} already exists.")
        except PermissionError:
            print(f"Permission denied: cannot create directory {path}.")
        except Exception as e:
            print(f"An error occurred while trying to create the directory {path}: {e}")

    # input path then create structured folder under this folder
    def CreateFolders(self, CasePath=None):
        CTADir = os.path.join(CasePath, 'CTA')
        if not os.path.exists(CTADir):
            os.mkdir(CTADir)

        PreProcPath = os.path.join(CasePath, 'Preproc')
        OriDataPath = os.path.join(PreProcPath, 'OriData')
        BranchPath = os.path.join(OriDataPath, 'Branches')
        CSVPath = os.path.join(PreProcPath, 'CSV')
        XsectionPath = os.path.join(PreProcPath, 'Xsection')
        XsectionFixPath = os.path.join(PreProcPath, 'XsectionFix')
        XsectionFixExtPath = os.path.join(PreProcPath, 'XsectionFixExt')
        OriVolPath = os.path.join(PreProcPath, 'OriVolume')

        CFDPath = os.path.join(CasePath, 'CFD')
        FFRPath = os.path.join(CasePath, 'FFR')
        STLPath = os.path.join(CFDPath, 'STL')
        ArVolCalPath = os.path.join(CFDPath, 'MurraysLaw')

        TablePath = os.path.join(CasePath, 'Table_Preproc')
        CFDTablePath = os.path.join(CasePath, 'Table_CFD')
        CFDResultPath = os.path.join(CFDPath, 'result')
        FFRResultPath = os.path.join(FFRPath, 'result')
        PostProcessingPath = os.path.join(CasePath, 'PostProgress')
        FFRPostProcessingPath = os.path.join(CasePath, 'FFRPostProgress')
        IFRPostProcessingPath = os.path.join(CasePath, 'IFRPostProgress')

        self.CreateSingleFolder(self.TableDir)
        self.CreateSingleFolder(PreProcPath)
        self.CreateSingleFolder(OriDataPath)
        self.CreateSingleFolder(BranchPath)
        self.CreateSingleFolder(OriVolPath)
        self.CreateSingleFolder(CSVPath)
        self.CreateSingleFolder(XsectionPath)
        self.CreateSingleFolder(XsectionFixPath)
        self.CreateSingleFolder(XsectionFixExtPath)
        self.CreateSingleFolder(CFDPath)
        self.CreateSingleFolder(FFRPath)
        self.CreateSingleFolder(CFDResultPath)
        self.CreateSingleFolder(FFRResultPath)
        self.CreateSingleFolder(STLPath)
        self.CreateSingleFolder(ArVolCalPath)
        self.CreateSingleFolder(TablePath)
        self.CreateSingleFolder(CFDTablePath)
        self.CreateSingleFolder(PostProcessingPath)
        self.CreateSingleFolder(FFRPostProcessingPath)
        self.CreateSingleFolder(IFRPostProcessingPath)

    def ResetCasePathLists(self):
        """Clear lists derived from DCMDirList so they can be rebuilt safely."""

        for attribute_name in (
                'CTANewFilePathList', 'CTANewPathList', 'CTAPathList',
                'XsecFixCTAPathList', 'OriVolCTAPathList', 'leftCenterlineJsonList',
                'rightCenterlineJsonList', 'OriDataPathList', 'CSVPathList',
                'BranchPathList', 'AortaPathList', 'AortaFixedPathList',
                'NewAortaPathList', 'XsectionPathList', 'XsectionFixPathList',
                'XsectionFixExtPathList', 'OriVolumePathList', 'FinalMaskPathList',
                'StlFilePathList', 'ArVolCalPathList', 'STLFolderList',
                'MeshFolderList', 'MeshFilePathList'):
            setattr(self, attribute_name, [])

    # store the required path information as list
    def StoreListInformation(self):
        self.ResetCasePathLists()
        for path in self.DCMDirList:
            self.OriDataPathList.append(os.path.join(path, 'Preproc/OriData'))
            self.CSVPathList.append(os.path.join(path, 'Preproc/CSV'))
            self.BranchPathList.append(os.path.join(path, 'Preproc/OriData/Branches'))
            self.XsectionPathList.append(os.path.join(path, 'Preproc/Xsection'))
            self.XsectionFixPathList.append(os.path.join(path, 'Preproc/XsectionFix'))
            self.XsectionFixExtPathList.append(os.path.join(path, 'Preproc/XsectionFixExt'))
            self.OriVolumePathList.append(os.path.join(path, 'Preproc/OriVolume'))
            self.STLFolderList.append(os.path.join(path, 'CFD/STL'))
            self.ArVolCalPathList.append(os.path.join(path, 'CFD/MurraysLaw'))
            self.MeshFolderList.append(os.path.join(path, 'CFD/Mesh'))
            pass

        # copy centerline JSON files update 13Mar24
        self.leftCenterlineJsonList = [
            os.path.join(path, CENTERLINE_JSON_NAME.format('L'))
            for path in self.CSVPathList
        ]
        self.rightCenterlineJsonList = [
            os.path.join(path, CENTERLINE_JSON_NAME.format('R'))
            for path in self.CSVPathList
        ]

        # store the final mask path using orivolume path list
        self.FinalMaskPathList = [os.path.join(x, 'OriVolumeCAVal.nii.gz') for x in self.OriVolumePathList]
        self.StlFilePathList = [os.path.join(x, 'Raw.stl') for x in self.STLFolderList]
        self.MeshFilePathList = [os.path.join(x, 'FluidMeshInit.nas') for x in self.MeshFolderList]
        # store the aorta path list
        self.AortaPathList = [os.path.join(x, 'AortaVal.nii.gz') for x in self.BranchPathList]
        # iteration of store Aorta mask after lumen correction
        self.AortaFixedPathList = [os.path.join(x, 'AortaFixVal.nii.gz') for x in self.XsectionFixPathList]

        # generate table paths
        self.DictionaryPath = os.path.join(self.TableDir, 'coronarySegments.json')
        self.FilterMaskTablePath = os.path.join(self.TableDir, 'FilterMask.csv')
        self.AortaFixTablePath = os.path.join(self.TableDir, 'AortaLCFix.csv')
        self.XsectionTablePath = os.path.join(self.TableDir, 'Xsection.csv')
        self.XsectionFixTablePath = os.path.join(self.TableDir, 'XsectionFix.csv')
        self.OriVolumeFixTablePath = os.path.join(self.TableDir, 'OriVolumeFix.csv')
        self.LCProcessPath = os.path.join(self.TableDir, 'LCProcess.csv')
        self.OCProcessPath = os.path.join(self.TableDir, 'OCProcess.csv')
        self.XsectionFixExtTablePath = os.path.join(self.TableDir, 'Extension.csv')
        self.OriVolumeTablePath = os.path.join(self.TableDir, 'XsectionOriVolume.csv')
        self.SSTablePath = os.path.join(self.TableDir, 'SS.csv')
        self.SGTablePath = os.path.join(self.TableDir, 'SG.csv')
        self.AVCTablePath = os.path.join(self.TableDir, 'MurraysLaw.csv')
        self.BatchDCMTablePath = os.path.join(self.TableDir, 'BatchDCM.csv')
        self.JasonUpdateTablePath = os.path.join(self.TableDir, 'JasonUpdate.csv')
        self.CFD_MeshTablePath = os.path.join(self.TableDir, 'CFD_Mesh.csv')
        self.CFD_STLcutTablePath = os.path.join(self.TableDir, 'CFD_STLcut.csv')
        self.CFD_STLmeshTablePath = os.path.join(self.TableDir, 'CFD_STLmesh.csv')
        self.CFD_FindFaceTablePath = os.path.join(self.TableDir, 'CFD_FindFace.csv')
        self.CFD_setTablePath = os.path.join(self.TableDir, 'set.csv')
        self.CFD_GenerateInTablePath = os.path.join(self.TableDir, 'CFD_GenerateIn.csv')
        self.CFD_PostProgressTablePath = os.path.join(self.TableDir, 'CFD_PostProgress.csv')
        self.DataOutputTablePath = os.path.join(self.TableDir, 'CFD_DataOutput.csv')
        self.PostNpyTablePath = os.path.join(self.TableDir, 'CFD_PostNpy.csv')
        self.Map_ResultTablePath = os.path.join(self.TableDir, 'CFD_Map_Result.csv')
        self.FFRCalculationTablePath = os.path.join(self.TableDir, 'CFD_FFRCalculation.csv')
        self.ParaviewTablePath = os.path.join(self.TableDir, 'CFD_Paraview.csv')
        self.SaveVtuTablePath = os.path.join(self.TableDir, 'CFD_SaveVtu.csv')

    @staticmethod
    # generate windows command line
    def CopyCommand(source=None, dst=None):

        import shutil

        # copy
        source = source.replace('/', '\\')
        dst = dst.replace('/', '\\')
        shutil.copy(source, dst)

        # copy command
        cmd = 'copy {} {}'.format(source, dst)
        cmd = cmd.replace('/', '\\')

        return cmd

    @staticmethod
    def WriteAortaFixCsv(dstPath=None):
        header = ['3D Connectivity Filter', 'Morphological Process']
        first_col = ['first', '1']
        second_col = ['3D', 'No', 'Closing', '6', '0', '2000']
        csv = []
        csv.append(first_col)
        csv.append(second_col)
        df = pd.DataFrame(data=csv, columns=None, index=None)
        df = df.T
        df.columns = header
        df.to_csv(dstPath, index=False)

    # generate table used for batch generate nifti
    def GenerateCTATable(self):
        CSVIndex = ['DICOMDirectoryPath', 'OutputPath']
        Data = [self.DCMDirList, self.CTADirList]

        df = pd.DataFrame(data=Data, columns=None, index=None)
        df = df.T
        df.columns = CSVIndex

        df.to_csv(self.BatchDCMTablePath, index=False, mode='w')

    def BatchDCM2Nifti(self):
        self.model.convertDataType.BatchDICOMconvertNIFTI(tablePath=self.BatchDCMTablePath, refImg=False)

    def _load_batch_segmentation_records(self):
        self._ensure_batch_segmentation_record_path()
        record_path = Path(self.BatchSegmentationRecordPath)
        with interprocess_file_lock(str(record_path) + '.lock'):
            _, records = _read_csv_rows(record_path)
        return {
            (str(record.get('batch_name', '')).casefold(),
             str(record.get('case_id', '')).casefold()): record
            for record in records
            if record.get('batch_name') and record.get('case_id')
        }

    def _save_batch_segmentation_record_map(self, record_map):
        self._ensure_batch_segmentation_record_path()
        return write_batch_segmentation_records(
            self.BatchSegmentationRecordPath,
            list(record_map.values()),
        )

    def _ensure_batch_segmentation_record_path(self):
        if not self.TableDir:
            if not self.WorkingDir:
                raise ValueError('Working directory is required for the batch ledger')
            self.TableDir = os.path.join(self.WorkingDir, 'Table')
        record_folder = Path(self.TableDir) / SEGMENTATION_RECORD_FOLDER
        record_folder.mkdir(parents=True, exist_ok=True)
        expected_path = record_folder / SEGMENTATION_RECORD_FILE
        if not self.BatchSegmentationRecordPath:
            self.BatchSegmentationRecordPath = str(expected_path)
        return Path(self.BatchSegmentationRecordPath)

    def _ensure_case_record(self, record_map, case_path):
        case_path = Path(case_path)
        key = (self.BatchName.casefold(), case_path.name.casefold())
        record = record_map.get(key)
        if record is None:
            source_path = case_path / 'Preproc' / 'OriData' / 'CTA.nii.gz'
            try:
                source_sha256 = _sha256_file(source_path)
            except Exception:
                source_sha256 = ''
            record = {
                'case_id': case_path.name,
                'batch_name': self.BatchName,
                'case_relative_path': '{0}/{1}'.format(
                    self.BatchName, case_path.name
                ),
                'source_sha256': source_sha256,
                'centerline_json_source_sha256': '',
                'filter_mask_input_sha256': '',
            }
            record.update(SEGMENTATION_STATUS_DEFAULTS)
            record['complete_segmentation'] = int(
                is_complete_segmentation(case_path)
            )
            record_map[key] = record
        else:
            record.setdefault('centerline_json_source_sha256', '')
            record.setdefault('filter_mask_input_sha256', '')
            for field_name, default_value in SEGMENTATION_STATUS_DEFAULTS.items():
                record.setdefault(field_name, default_value)
        return record

    @staticmethod
    def _append_record_error(record, error_field, message):
        messages = [record.get(error_field, ''), message]
        record[error_field] = _clean_error_message(
            '; '.join(item for item in messages if item)
        )

    @staticmethod
    def _append_segmentation_record_error(record, message):
        BatchCoronary._append_record_error(record, 'seg_exe_er_msg', message)

    def _move_failed_cases(
            self,
            failed_cases,
            record_map,
            error_field,
            stage_label):
        failed_batch_path = (
            Path(self.FailedRootPath)
            / FAILED_SEGMENTATION_FOLDER_NAME
            / self.BatchName
        )
        records_changed = False

        seen_case_paths = set()
        for case_path in failed_cases:
            case_path = Path(case_path)
            case_path_key = _normalized_path_key(case_path)
            if case_path_key in seen_case_paths:
                continue
            seen_case_paths.add(case_path_key)
            key = (self.BatchName.casefold(), case_path.name.casefold())
            record = self._ensure_case_record(record_map, case_path)
            destination = failed_batch_path / case_path.name
            try:
                if not _path_is_within(case_path, self.WorkingDir):
                    raise ValueError('Refusing to move a case outside the working batch')
                if _path_is_within(failed_batch_path, self.WorkingDir):
                    raise ValueError(
                        'Failed-case destination cannot be inside the working batch'
                    )
                if _path_is_within(destination, case_path):
                    raise ValueError('Failed-case destination cannot be inside the case folder')
                if destination.exists():
                    collision_suffix = datetime.now().strftime('%Y%m%d%H%M%S%f')
                    alternate_name = '{0}__retry_{1}_{2}'.format(
                        case_path.name, collision_suffix, os.getpid()
                    )
                    destination = failed_batch_path / alternate_name
                    counter = 1
                    while destination.exists():
                        destination = failed_batch_path / '{0}_{1}'.format(
                            alternate_name, counter
                        )
                        counter += 1
                    self._append_record_error(
                        record,
                        error_field,
                        'Existing failed case was preserved; moved this failure to {0}'.format(
                            destination
                        ),
                    )
                    records_changed = True
                failed_batch_path.mkdir(parents=True, exist_ok=True)
                shutil.move(str(case_path), str(destination))
                self.UpdateMsgLog(
                    'Moved failed {0} case {1} to {2}'.format(
                        stage_label, case_path.name, destination
                    )
                )
            except Exception as error:
                self._append_record_error(
                    record,
                    error_field,
                    'Failed to move case: {0}'.format(error),
                )
                records_changed = True
                self.UpdateMsgLog(
                    'Could not move failed {0} case {1}: {2}'.format(
                        stage_label, case_path.name, error
                    )
                )

        if records_changed:
            self._save_batch_segmentation_record_map(record_map)

    def _move_failed_segmentation_cases(self, failed_cases, record_map):
        return self._move_failed_cases(
            failed_cases,
            record_map,
            'seg_exe_er_msg',
            'segmentation',
        )

    def _run_tracked_segmentation(self, safe_runner):
        """Run every discovered case independently and persist each result."""

        record_path = self._ensure_batch_segmentation_record_path()
        record_folder = record_path.parent

        # Reconcile records left by an interrupted earlier run before starting.
        try:
            merge_batch_segmentation_records(
                self.ManifestPath,
                self.InitialSegmentationRoot,
            )
        except Exception as error:
            self.UpdateMsgLog(
                'Manifest reconciliation warning; batch processing will continue: {0}'.format(
                    error
                )
            )

        failed_batch_path = (
            Path(self.FailedRootPath)
            / FAILED_SEGMENTATION_FOLDER_NAME
            / self.BatchName
        )
        if _path_is_within(failed_batch_path, self.WorkingDir):
            raise ValueError(
                'Failed segmentation output must be outside the working batch: {0}'.format(
                    failed_batch_path
                )
            )
        for case_path in self.DCMDirList:
            if (os.path.normcase(os.path.abspath(case_path))
                    == os.path.normcase(os.path.abspath(self.WorkingDir))):
                raise ValueError(
                    'The working batch itself cannot be processed as a case: {0}'.format(
                        self.WorkingDir
                    )
                )

        run_lock_path = record_folder / (SEGMENTATION_RECORD_FILE + '.run.lock')
        with interprocess_file_lock(run_lock_path, timeout_seconds=1):
            record_map = self._load_batch_segmentation_records()
            case_paths = list(self.DCMDirList)
            complete_case_paths = []
            failed_case_paths = []
            current_records = []
            skipped_count = 0
            executed_count = 0

            def segmentation_progress():
                processed_count = len(complete_case_paths) + len(failed_case_paths)
                return (
                    'Batch progress: completed {0}, failed {1}, processed {2}/{3}.'
                    .format(
                        len(complete_case_paths),
                        len(failed_case_paths),
                        processed_count,
                        len(case_paths),
                    )
                )

            for case_path_text in case_paths:
                case_path = Path(case_path_text)
                case_id = case_path.name
                key = (self.BatchName.casefold(), case_id.casefold())
                case_relative_path = '{0}/{1}'.format(self.BatchName, case_id)
                previous_record = record_map.get(key)
                if previous_record is not None:
                    previous_record = self._ensure_case_record(
                        record_map, case_path
                    )
                source_cta_path = case_path / 'Preproc' / 'OriData' / 'CTA.nii.gz'
                source_sha256 = ''
                source_hash_error = ''
                try:
                    source_sha256 = _sha256_file(source_cta_path)
                except Exception as error:
                    source_hash_error = 'Could not fingerprint input CTA: {0}'.format(
                        error
                    )

                if (previous_record is not None
                        and str(previous_record.get('complete_segmentation', '')) == '1'
                        and str(previous_record.get('seg_exe_failed', '')) == '0'
                        and source_sha256
                        and previous_record.get('source_sha256', '') == source_sha256
                        and is_complete_segmentation(case_path)):
                    skipped_count += 1
                    complete_case_paths.append(str(case_path))
                    current_records.append(previous_record)
                    self.UpdateMsgLog(
                        'Skip completed segmentation case: {0}\n{1}'.format(
                            case_id, segmentation_progress()
                        )
                    )
                    continue

                executed_count += 1
                record = {
                    'case_id': case_id,
                    'batch_name': self.BatchName,
                    'case_relative_path': case_relative_path,
                    'source_sha256': source_sha256,
                    'centerline_json_source_sha256': '',
                    'filter_mask_input_sha256': '',
                }
                record.update(SEGMENTATION_STATUS_DEFAULTS)
                record.update({
                    'run_segmentation': 1,
                    'seg_exe_failed': 1,
                    'seg_exe_er_msg': 'Segmentation started but did not finish.',
                    'coronary_segment_failed': 1,
                    'complete_segmentation': 0,
                })
                record_map[key] = record
                self._save_batch_segmentation_record_map(record_map)
                self.UpdateMsgLog('Segmentation start: {0}'.format(case_id))

                if source_hash_error:
                    record['seg_exe_er_msg'] = _clean_error_message(
                        source_hash_error
                    )
                    self._save_batch_segmentation_record_map(record_map)
                    current_records.append(record)
                    failed_case_paths.append(str(case_path))
                    self.UpdateMsgLog(
                        'Invalid CTA input; continuing to next case: {0}\n{1}'.format(
                            case_id, segmentation_progress()
                        )
                    )
                    continue

                if (not self.CGEXEPath
                        or not Path(self.CGEXEPath).is_file()):
                    record['seg_exe_er_msg'] = _clean_error_message(
                        'Centerline generation executable does not exist: {0}'
                        .format(self.CGEXEPath)
                    )
                    self._save_batch_segmentation_record_map(record_map)
                    current_records.append(record)
                    failed_case_paths.append(str(case_path))
                    self.UpdateMsgLog(
                        'Segmentation executable unavailable; continuing to '
                        'next case: {0}\n{1}'.format(
                            case_id, segmentation_progress()
                        )
                    )
                    continue

                mask_path = case_path / SEGMENTATION_MASK_RELATIVE_PATH
                stale_mask_path = None
                try:
                    if mask_path.exists():
                        if not mask_path.is_file():
                            raise ValueError(
                                'Existing segmentation output is not a file: {0}'.format(
                                    mask_path
                                )
                            )
                        stale_mask_path = mask_path.with_name(
                            '{0}.pre_run_{1}_{2}'.format(
                                mask_path.name,
                                os.getpid(),
                                datetime.now().strftime('%Y%m%d%H%M%S%f'),
                            )
                        )
                        os.replace(str(mask_path), str(stale_mask_path))
                except Exception as error:
                    record['seg_exe_er_msg'] = _clean_error_message(
                        'Could not preserve existing segmentation before rerun: {0}'.format(
                            error
                        )
                    )
                    self._save_batch_segmentation_record_map(record_map)
                    current_records.append(record)
                    failed_case_paths.append(str(case_path))
                    self.UpdateMsgLog(
                        'Segmentation preparation failed; continuing to next case: '
                        '{0}\n{1}'.format(
                            case_id, segmentation_progress()
                        )
                    )
                    continue

                try:
                    result = safe_runner(
                        destination=str(case_path / 'Preproc' / 'OriData'),
                        exe_path=self.CGEXEPath,
                        timeout_seconds=self.SegmentationTimeoutSeconds,
                        output_limit=MAX_SEGMENTATION_ERROR_LENGTH,
                    )
                    mask_complete = is_complete_segmentation(case_path)
                    exe_failed = bool(result.get('exe_failed', True))
                    complete = mask_complete and not exe_failed
                    record['seg_exe_failed'] = int(exe_failed)
                    record['seg_exe_er_msg'] = _clean_error_message(
                        result.get('error_message', '')
                    )
                    record['coronary_segment_failed'] = int(not complete)
                    record['complete_segmentation'] = int(complete)
                    if not mask_complete and not record['seg_exe_er_msg']:
                        record['seg_exe_er_msg'] = (
                            'sc_labelled_coronary_segment.nii.gz is missing or empty.'
                        )
                    elif exe_failed and not record['seg_exe_er_msg']:
                        record['seg_exe_er_msg'] = (
                            'Segmentation executable reported a failed stage.'
                        )
                except Exception as error:
                    record['seg_exe_failed'] = 1
                    record['seg_exe_er_msg'] = _clean_error_message(
                        'Unexpected segmentation runner error: {0}'.format(error)
                    )
                    record['coronary_segment_failed'] = 1
                    record['complete_segmentation'] = 0

                if record['complete_segmentation'] == 1 and stale_mask_path is not None:
                    try:
                        stale_mask_path.unlink()
                    except OSError as error:
                        self._append_segmentation_record_error(
                            record,
                            'Could not remove preserved old segmentation: {0}'.format(
                                error
                            ),
                        )

                self._save_batch_segmentation_record_map(record_map)
                current_records.append(record)
                if record['complete_segmentation'] == 1:
                    complete_case_paths.append(str(case_path))
                    outcome_message = 'Segmentation complete: {0}'.format(case_id)
                else:
                    failed_case_paths.append(str(case_path))
                    outcome_message = (
                        'Segmentation failed; continuing to next case: {0}'.format(
                            case_id
                        )
                    )
                self.UpdateMsgLog(
                    '{0}\n{1}'.format(outcome_message, segmentation_progress())
                )

            self.UpdateMsgLog(
                'Segmentation resume summary: skipped {0}, executed {1}, '
                'completed {2}, failed {3}, total {4}.'.format(
                    skipped_count,
                    executed_count,
                    len(complete_case_paths),
                    len(failed_case_paths),
                    len(case_paths),
                )
            )

            # Moving happens only after every case in the fixed snapshot was attempted.
            self._move_failed_segmentation_cases(failed_case_paths, record_map)
            # Also upgrades legacy ledgers when every case was resume-skipped.
            self._save_batch_segmentation_record_map(record_map)
            self.DCMDirList = complete_case_paths
            self.CTADirList = [
                os.path.join(case_path, 'CTA', 'CTA.nii.gz')
                for case_path in self.DCMDirList
            ]
            self.StoreListInformation()
            try:
                merge_batch_segmentation_records(
                    self.ManifestPath,
                    self.InitialSegmentationRoot,
                )
            except Exception as error:
                self.UpdateMsgLog(
                    'Manifest merge warning; batch record is preserved at {0}: {1}'.format(
                        self.BatchSegmentationRecordPath, error
                    )
                )
            return current_records

    # Move CTA file to specified folder, then generate centerline and mask
    def BatchCenterlineGen(self):
        df = pd.DataFrame(data=self.OriDataPathList, columns=['FolderPath'])

        BatchCCTablePath = os.path.join(self.TableDir, 'BatchCenterlineGeneration.csv')
        df.to_csv(BatchCCTablePath, index=False)

        safe_runner = getattr(
            self.model.CenterlineGeneration,
            'run_case_safely',
            None,
        )
        if callable(safe_runner):
            return self._run_tracked_segmentation(safe_runner)

        # Compatibility fallback for older injected CenterlineGeneration objects.
        self.model.CenterlineGeneration.InitCenterlineEx(
            exePath=self.CGEXEPath,
            TablePath=BatchCCTablePath,
        )
        self.model.CenterlineGeneration.BatchViaExcel()
        return None

    def BeginCenterlineCsvRun(self):
        """Select only cases whose complete centerline CSV outputs need repair."""

        self.CenterlineCsvOriginalCasePaths = list(self.DCMDirList)
        self.CenterlineCsvSkipPaths = set()
        self.CenterlineCsvSourceHashes = {}
        self.CenterlineJsonStageErrors = {}
        record_map = self._load_batch_segmentation_records()
        rerun_case_paths = []

        for case_path_text in self.CenterlineCsvOriginalCasePaths:
            case_path = Path(case_path_text)
            case_key = _normalized_path_key(case_path)
            record = self._ensure_case_record(record_map, case_path)
            fingerprint = ''
            try:
                fingerprint, _resolved = centerline_json_source_fingerprint(
                    case_path
                )
                self.CenterlineCsvSourceHashes[case_key] = fingerprint
            except Exception as error:
                self.CenterlineJsonStageErrors[case_key] = _clean_error_message(
                    error
                )

            can_resume = (
                str(record.get('complete_centerline_csv', '')) == '1'
                and str(record.get('centerline_csv_failed', '')) == '0'
                and bool(fingerprint)
                and record.get('centerline_json_source_sha256', '') == fingerprint
                and is_complete_centerline_csv(case_path)
            )
            if can_resume:
                self.CenterlineCsvSkipPaths.add(case_key)
                self.UpdateMsgLog(
                    'Skip completed centerline CSV case: {0}'.format(
                        case_path.name
                    )
                )
            else:
                rerun_case_paths.append(str(case_path))

        if self.CenterlineCsvOriginalCasePaths:
            self._save_batch_segmentation_record_map(record_map)
        self.DCMDirList = rerun_case_paths
        self.CTADirList = [
            os.path.join(case_path, 'CTA', 'CTA.nii.gz')
            for case_path in rerun_case_paths
        ]
        self.StoreListInformation()
        self.UpdateMsgLog(
            'Centerline CSV resume summary: skipped {0}, rerun {1}, total {2}.'
            .format(
                len(self.CenterlineCsvSkipPaths),
                len(rerun_case_paths),
                len(self.CenterlineCsvOriginalCasePaths),
            )
        )
        return {
            'skipped': len(self.CenterlineCsvSkipPaths),
            'rerun': len(rerun_case_paths),
            'total': len(self.CenterlineCsvOriginalCasePaths),
        }

    def FinalizeCenterlineCsvRun(self):
        """Validate every fixed-snapshot case, record it, then move failures."""

        original_case_paths = list(self.CenterlineCsvOriginalCasePaths)
        if not original_case_paths:
            original_case_paths = list(self.DCMDirList)
        record_map = self._load_batch_segmentation_records()
        ready_case_paths = []
        failed_case_paths = []
        outcomes = {}

        for checked_count, case_path_text in enumerate(original_case_paths, 1):
            case_path = Path(case_path_text)
            case_key = _normalized_path_key(case_path)
            record = self._ensure_case_record(record_map, case_path)
            errors = []
            fingerprint = self.CenterlineCsvSourceHashes.get(case_key, '')
            try:
                current_fingerprint, _resolved = (
                    centerline_json_source_fingerprint(case_path)
                )
                if fingerprint and current_fingerprint != fingerprint:
                    errors.append(
                        'Centerline JSON source changed during processing.'
                    )
                fingerprint = current_fingerprint
            except Exception as error:
                errors.append(str(error))

            stage_error = self.CenterlineJsonStageErrors.get(case_key, '')
            if stage_error:
                errors.append(stage_error)
            inspection = inspect_centerline_csv_case(case_path)
            if not inspection['complete']:
                errors.append(inspection['error_message'])

            error_message = _clean_error_message(
                '; '.join(dict.fromkeys(error for error in errors if error))
            )
            complete = not bool(error_message)
            record['run_centerline_csv_check'] = 1
            record['centerline_csv_failed'] = int(not complete)
            record['centerline_csv_er_msg'] = error_message
            record['complete_centerline_csv'] = int(complete)
            record['centerline_json_source_sha256'] = fingerprint

            if case_key not in self.CenterlineCsvSkipPaths:
                record['run_filter_mask'] = 0
                record['filter_mask_failed'] = 0
                record['filter_mask_er_msg'] = ''
                record['complete_filter_mask'] = 0
                record['filter_mask_input_sha256'] = ''

            if complete:
                ready_case_paths.append(str(case_path))
            else:
                failed_case_paths.append(str(case_path))
            outcomes[str(case_path)] = {
                'complete': complete,
                'error_message': error_message,
                'source_sha256': fingerprint,
            }
            self._save_batch_segmentation_record_map(record_map)
            self.UpdateMsgLog(
                'CSV safeguard: ready {0}, failed {1}, checked {2}/{3}.'.format(
                    len(ready_case_paths),
                    len(failed_case_paths),
                    checked_count,
                    len(original_case_paths),
                )
            )

        # Moving happens only after every case in the fixed snapshot was checked.
        # If a multi-case batch has zero successful cases, hold every folder in
        # place.  A unanimous failure is much more likely to be a shared input,
        # configuration, or software problem than independent bad cases.
        hold_entire_batch = (
            len(original_case_paths) > 1
            and not ready_case_paths
            and len(failed_case_paths) == len(original_case_paths)
        )
        if hold_entire_batch:
            self.UpdateMsgLog(
                'SAFETY STOP: all {0} cases failed the centerline CSV '
                'safeguard. No case folders were moved; review the recorded '
                'errors before retrying.'.format(len(original_case_paths))
            )
        else:
            self._move_failed_cases(
                failed_case_paths,
                record_map,
                'centerline_csv_er_msg',
                'centerline CSV',
            )
        self.DCMDirList = ready_case_paths
        self.CTADirList = [
            os.path.join(case_path, 'CTA', 'CTA.nii.gz')
            for case_path in ready_case_paths
        ]
        self.StoreListInformation()
        try:
            merge_batch_segmentation_records(
                self.ManifestPath,
                self.InitialSegmentationRoot,
            )
        except Exception as error:
            self.UpdateMsgLog(
                'Manifest merge warning after CSV safeguard; batch record is '
                'preserved at {0}: {1}'.format(
                    self.BatchSegmentationRecordPath, error
                )
            )
        return outcomes

    # copy centerline json file to specified folder
    @timmer
    def CopyCenterlineJson(self):
        results = {}
        for case_path_text in list(self.DCMDirList):
            case_path = Path(case_path_text)
            case_key = _normalized_path_key(case_path)
            try:
                fingerprint = stage_centerline_json_pair(case_path)
                self.CenterlineCsvSourceHashes[case_key] = fingerprint
                self.CenterlineJsonStageErrors.pop(case_key, None)
                results[str(case_path)] = {
                    'complete': True,
                    'source_sha256': fingerprint,
                    'error_message': '',
                }
                self.UpdateMsgLog(
                    'Centerline JSON staged: {0}'.format(case_path.name)
                )
            except Exception as error:
                error_message = _clean_error_message(error)
                self.CenterlineJsonStageErrors[case_key] = error_message
                results[str(case_path)] = {
                    'complete': False,
                    'source_sha256': '',
                    'error_message': error_message,
                }
                self.UpdateMsgLog(
                    'Centerline JSON staging failed; continuing: {0}: {1}'.format(
                        case_path.name, error_message
                    )
                )
        return results

    # copy cta file from cta folder to branches folder
    def CopyCTA(self):
        self.CTANewFilePathList = [y.replace('\\CTA\\', '\\PreProc\\OriData\\', 1) for y in self.CTADirList]
        self.CTANewPathList = [y.replace('CTA\\CTA.nii.gz', 'PreProc\\OriData', 1) for y in self.CTADirList]

        # copy cta.nii.gz to Preproc/OriData folder
        for i in range(len(self.CTANewFilePathList)):
            cmd = self.CopyCommand(self.CTADirList[i], self.CTANewPathList[i])
            print(cmd)

        # disabled JZ 13Mar24 !!!
        # self.BranchPathList = [x.replace('\\CTA.nii.gz', '\\Branches') for x in self.CTANewFilePathList]
        self.CTAPathList = [x.replace('\\CTA\\', '\\Preproc\\OriData\\Branches\\', 1) for x in self.CTADirList]

        for dstCTAPath in self.CTAPathList:
            if not os.path.exists(dstCTAPath):
                cmd = self.CopyCommand(self.CTADirList[self.CTAPathList.index(dstCTAPath)], dstCTAPath)
                print(cmd)
            else:
                pass
        # copy cta file from cta folder to XsectionFix folder
        self.XsecFixCTAPathList = [y.replace('\\CTA\\', '\\Preproc\\XsectionFixExt\\', 1) for y in self.CTADirList]
        for dstFixCTAPath in self.XsecFixCTAPathList:
            if not os.path.exists(dstFixCTAPath):
                cmd = self.CopyCommand(self.CTADirList[self.XsecFixCTAPathList.index(dstFixCTAPath)], dstFixCTAPath)
                print(cmd)
            else:
                pass

        # copy cta file from cta folder to OriVolume folder
        self.OriVolCTAPathList = [z.replace('\\CTA\\', '\\Preproc\\OriVolume\\', 1) for z in self.CTADirList]
        for dstOriVolCTAPath in self.OriVolCTAPathList:
            if not os.path.exists(dstOriVolCTAPath):
                cmd = self.CopyCommand(self.CTADirList[self.OriVolCTAPathList.index(dstOriVolCTAPath)],
                                       dstOriVolCTAPath)
                print(cmd)
            else:
                pass

    # copy aorta file which extracted from final label file to Orivolume folder
    def CopyAorta2OriVolumeFolder(self):
        # self.AortaPathList = [os.path.join(x, 'AortaVal.nii.gz') for x in self.BranchPathList]
        # self.NewAortaPathList = [x.replace('OriData/Branches', 'OriVolume') for x in self.AortaPathList]
        self.AortaPathList = [os.path.join(x, 'AortaFixVal.nii.gz') for x in self.XsecFixCTAPathList]
        self.AortaPathList = [
            x.replace(r'XsectionFixExt\CTA.nii.gz', 'XsectionFix')
            for x in self.AortaPathList
        ]
        self.NewAortaPathList = [x.replace('XsectionFix', 'OriVolume') for x in self.AortaPathList]
        print(self.AortaPathList)
        print(self.NewAortaPathList)
        for oldPath, newPath in zip(self.AortaPathList, self.NewAortaPathList):
            if not os.path.exists(newPath):
                cmd = self.CopyCommand(oldPath, newPath)
                print(cmd)
                os.system(cmd)

    # input the root directory, extract all the centerlines into the same folder with json files
    def BatchExtractCenterline(self):
        for dir in self.CSVPathList:
            print("Extract centerline in: {}".format(dir))
            try:
                self.model.CenterlineExtraction.BatchCenterlineProcessing(dir=dir)
            except Exception as error:
                self.UpdateMsgLog(
                    'Centerline extraction failed; continuing: {0}: {1}'.format(
                        dir, error
                    )
                )

    # generate table used for batch Jason update
    def GenerateJsonUpdateTable(self):
        # heading
        CSVIndex = ["CSVDirectory", "JsonFile", "UpdateType"]

        # check csv and left and right same length
        if not ((len(self.CSVPathList) == len(self.leftCenterlineJsonList))
                and (len(self.CSVPathList) == len(self.rightCenterlineJsonList))):
            print('!!!Warning Jason file list is not the same length as the CSV folder list')

        CSVFolderLst = []
        JsonFileLst = []
        UpdateTypeLst = []

        for i in range(len(self.CSVPathList)):
            # append results
            # left
            CSVFolderLst.append(self.CSVPathList[i])
            JsonFileLst.append(self.leftCenterlineJsonList[i])
            UpdateTypeLst.append("Replace with CSV")
            # right
            CSVFolderLst.append(self.CSVPathList[i])
            JsonFileLst.append(self.rightCenterlineJsonList[i])
            UpdateTypeLst.append("Replace with CSV")

        Data = [CSVFolderLst, JsonFileLst, UpdateTypeLst]

        # print to check results
        for j in range(numpy.min([10, len(self.CSVPathList)])):
            print(CSVFolderLst[j], JsonFileLst[j], UpdateTypeLst[j])

        df = pd.DataFrame(data=Data, columns=None, index=None)
        df = df.T
        df.columns = CSVIndex

        df.to_csv(self.JasonUpdateTablePath, index=False, mode='w')

    # Batch Jason Update
    def BatchJsonUpdate(self):
        update_one = getattr(
            self.model.CenterlineJSONUpdate,
            'UpdateCenterlineJson',
            None,
        )
        if not callable(update_one):
            self.model.CenterlineJSONUpdate.BatchUpdateJson(
                tablePath=self.JasonUpdateTablePath
            )
            return

        update_table = Pd_Funs.OpenDF(self.JasonUpdateTablePath, header=0)
        for csv_directory, json_file, update_type in zip(
                update_table['CSVDirectory'].tolist(),
                update_table['JsonFile'].tolist(),
                update_table['UpdateType'].tolist()):
            try:
                update_one(
                    jsonFile=json_file,
                    CSVDir=csv_directory,
                    updateType=update_type,
                )
            except Exception as error:
                self.UpdateMsgLog(
                    'Centerline JSON update failed; continuing: {0}: {1}'.format(
                        json_file, error
                    )
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
        # input list
        self.BranchDictionaryReadyCSVPaths = set()
        branchLst = []
        if self.ui.LM_CB_CPP.isChecked(): branchLst.append("LM")
        if self.ui.IMB_CB_CPP.isChecked(): branchLst.append("IMB")
        if self.ui.pLAD_CB_CPP.isChecked(): branchLst.append("pLAD")
        if self.ui.pLCX_CB_CPP.isChecked(): branchLst.append("pCx")
        if self.ui.mLAD_CB_CPP.isChecked(): branchLst.append("mLAD")
        if self.ui.dLAD_CB_CPP.isChecked(): branchLst.append("dLAD")
        if self.ui.D1_CB_CPP.isChecked(): branchLst.append("D1")
        if self.ui.D2_CB_CPP.isChecked(): branchLst.append("D2")
        if self.ui.mLCX_CB_CPP.isChecked(): branchLst.append("LCx")
        if self.ui.dLCX_CB_CPP.isChecked(): branchLst.append("L-PDA")
        if self.ui.OM1_CB_CPP.isChecked(): branchLst.append("OM1")
        if self.ui.OM2_CB_CPP.isChecked(): branchLst.append("OM2")
        if self.ui.pRCA_CB_CPP.isChecked(): branchLst.append("pRCA")
        if self.ui.mRCA_CB_CPP.isChecked(): branchLst.append("mRCA")
        if self.ui.dRCA_CB_CPP.isChecked(): branchLst.append("dRCA")
        if self.ui.V_CB_CPP.isChecked(): branchLst.append("V")
        if self.ui.AM_CB_CPP.isChecked(): branchLst.append("AM")
        if self.ui.CB_CB_CPP.isChecked(): branchLst.append("CB")

        # ignore coordinate list
        ignoreCoordsLst = Preprocess_Mask.StrToLst(strIn=self.ui.ignoreCheck_Txt_BPP.toPlainText())["listOut"]

        if not branchLst:  # empty list
            branchLst = ['LM', "pLAD", "pCx", "pRCA", "mRCA"]
            print("No Selection in branches use standard five branches:\n{}".format(branchLst))
        else:
            print("Selected branches:\n{}".format(branchLst))

        BranchDictionary.WriteDic(Outpath=self.DictionaryPath, branchLst=branchLst)
        for CSVPath in self.CSVPathList:
            # validate which branch can be used for each case
            tempJsonList = self.ReturnJsonList(CSVPath)
            UseJsonPath = os.path.join(CSVPath, 'coronarySegmentsUse.json')
            try:
                BranchDictionary.ValidateUse(dicPath=self.DictionaryPath,
                                             jsonList=tempJsonList,
                                             outputPath=UseJsonPath,
                                             ignoreCoordsLst=ignoreCoordsLst)
                self.BranchDictionaryReadyCSVPaths.add(
                    os.path.normcase(os.path.abspath(CSVPath))
                )
            except Exception as error:
                self.UpdateMsgLog(
                    'Branch dictionary failed; continuing: {0}: {1}'.format(
                        CSVPath, error
                    )
                )

    def RetainTableReadyCases(self):
        """Keep only cases with a valid branch-use JSON before table generation."""

        ready_case_paths = []
        for case_path, csv_path, area_volume_path in zip(
                self.DCMDirList, self.CSVPathList, self.ArVolCalPathList):
            use_json_path = Path(csv_path) / 'coronarySegmentsUse.json'
            try:
                if (self.BranchDictionaryReadyCSVPaths is not None
                        and os.path.normcase(os.path.abspath(csv_path))
                        not in self.BranchDictionaryReadyCSVPaths):
                    raise ValueError('branch dictionary was not generated in this run')
                if not use_json_path.is_file() or use_json_path.stat().st_size == 0:
                    raise FileNotFoundError('missing or empty coronarySegmentsUse.json')
                with use_json_path.open('r') as json_file:
                    case_dictionary = json.load(json_file)
                if not isinstance(case_dictionary, list):
                    raise ValueError('coronarySegmentsUse.json must contain a list')
                required_branch_fields = {
                    'use', 'original_volume_mask_fix', 'label',
                    'original_branch', 'original_csv',
                    'cross_section_image_LI', '3D_csv',
                    'cross_section_mask_NN', 'cross_section_mask_fix',
                    'cross_section_mask_fix_extension', 'branch_name',
                    'branch_label',
                }
                for branch_index, branch in enumerate(case_dictionary):
                    if not isinstance(branch, dict):
                        raise ValueError(
                            'branch {0} must be a dictionary'.format(branch_index)
                        )
                    if 'use' not in branch:
                        raise ValueError(
                            'branch {0} is missing use'.format(branch_index)
                        )
                    if not branch['use']:
                        continue
                    missing_fields = sorted(required_branch_fields.difference(branch))
                    if missing_fields:
                        raise ValueError(
                            'branch {0} is missing fields: {1}'.format(
                                branch_index, ', '.join(missing_fields)
                            )
                        )
                    if not isinstance(branch['label'], list) or not branch['label']:
                        raise ValueError(
                            'branch {0} has no label value'.format(branch_index)
                        )
                    string_fields = required_branch_fields.difference(
                        {'use', 'label', 'branch_label'}
                    )
                    invalid_string_fields = sorted(
                        field_name
                        for field_name in string_fields
                        if not isinstance(branch[field_name], str)
                    )
                    if invalid_string_fields:
                        raise ValueError(
                            'branch {0} fields must be text: {1}'.format(
                                branch_index, ', '.join(invalid_string_fields)
                            )
                        )
                for output_directory in (Path(csv_path), Path(area_volume_path)):
                    if not output_directory.is_dir():
                        raise NotADirectoryError(
                            'table output directory does not exist: {0}'.format(
                                output_directory
                            )
                        )
                    file_descriptor, write_test_name = tempfile.mkstemp(
                        prefix='.batch_table_write_test.',
                        dir=str(output_directory),
                    )
                    os.close(file_descriptor)
                    Path(write_test_name).unlink()
                ready_case_paths.append(case_path)
            except Exception as error:
                self.UpdateMsgLog(
                    'Skip table generation for case {0}: {1}'.format(
                        Path(case_path).name, error
                    )
                )

        self.DCMDirList = ready_case_paths
        self.CTADirList = [
            os.path.join(case_path, 'CTA', 'CTA.nii.gz')
            for case_path in self.DCMDirList
        ]
        self.StoreListInformation()

    # create Filter Mask Object and batch process
    def GenerateTable(self):
        """Build tables case by case so one bad case cannot stop later cases."""

        case_paths = list(self.DCMDirList)
        for attribute_name in GENERATED_TABLE_ATTRIBUTES:
            setattr(self, attribute_name, [])

        successful_case_paths = []
        for case_path in case_paths:
            table_lengths = {
                attribute_name: len(getattr(self, attribute_name))
                for attribute_name in GENERATED_TABLE_ATTRIBUTES
            }
            try:
                self.DCMDirList = [case_path]
                self.StoreListInformation()
                self._GenerateTableCurrentCases(write_shared_tables=False)
            except Exception as error:
                for attribute_name, original_length in table_lengths.items():
                    del getattr(self, attribute_name)[original_length:]
                self.UpdateMsgLog(
                    'Table generation failed; continuing: {0}: {1}'.format(
                        Path(case_path).name,
                        error,
                    )
                )
            else:
                successful_case_paths.append(case_path)

        # Write the shared batch tables once, after every case has been attempted.
        # An empty case list prevents the builder from appending duplicate rows.
        self.DCMDirList = []
        self.StoreListInformation()
        try:
            self._GenerateTableCurrentCases(write_shared_tables=True)
        finally:
            self.DCMDirList = successful_case_paths
            self.CTADirList = [
                os.path.join(case_path, 'CTA', 'CTA.nii.gz')
                for case_path in successful_case_paths
            ]
            self.StoreListInformation()

        return successful_case_paths

    def _GenerateTableCurrentCases(self, write_shared_tables=True):
        # define table column titles
        FM_Head = ['MaskPath', 'MaskVal', 'NewMaskVal', 'OutputFolder', 'FileNameRef']

        Xsection_Head = ['inputDir', 'inputCSVDir', 'outputDir', 'outputCSVDir', 'resampleDepth', 'CPRAngle', 'cpus',
                         'casePath', 'EXEPath']

        Case_Head = ['Interpolation', 'OutCSV', 'inputImagePath', 'inputCSVPath', 'outputImagePath', 'outputCSVPath']

        XsecFix_Head = ['Image Path', 'Mask Path', 'Process Path', 'Fill Original Value', 'Output Directory',
                        'Output Name']

        Extension_Head = ['Input Mask Path', 'Output Mask Path', 'First Automation', 'First Direction',
                          'First Use Circle', 'First Slice Start', 'First Slice Stop', 'First Reference Slice',
                          'First Reference Mask Depth', 'First Extension Total Slices', 'Second Extension',
                          'Second Automation', 'Second Direction', 'Second Use Circle', 'Second Slice Start',
                          'Second Slice Stop', 'Second Reference Slice', 'Second Reference Mask Depth',
                          'Second Extension Total Slices']

        OriVolume_Head = ['LoadFolder', 'RefImgName', 'XsecImgName', 'OriVolCSV', 'FillDistance', 'Processors',
                          'OutputFolder', 'OutImgName', 'Interpolation']

        XsecFixOri_Head = ['Image Path', 'Mask Path', 'Process Path', 'Fill Original Value', 'Output Directory',
                           'Output Name']

        SS_Head = ['RefImgPath', 'LoadFolder', 'RefSegmentationName', 'ThresholdValues', 'NewValues',
                   'OutputFolder', 'OutRefName']

        SG_Head = ['EXEPath', 'InputFilePath', 'OutputFilePath']

        AVCParam_Head = ['Total Inflow', 'Threshold', 'Step', 'Output Folder', 'Exponential', 'Multiple']

        AVCBranM_Head = ['Name Reference', 'Branch Label', 'Mask Path', 'Start Slice', 'Finish Slice', 'Mask Label',
                         'Area Choice', 'Reference File']

        AVCBatch_Head = ["Branch Map Path", "Parameters Path"]

        CFD_Mesh_Head = ["InputFolder", "OutputFolder", "MeshCut", "Mesh", "MeshExe", "FindFace", "Tolerance(FindFace)",
                         "Set(FindFace)", "GenerateInput", "AUIPath", "TimeFunction(GenerateInput)"]

        CFD_STLcut_Head = ["MeshCut", "InputStl(MeshCut)", "OutputFolder"]

        CFD_STLmesh_Head = ["Mesh", "InputStl(Mesh)", "MeshExe", "OutputFolder"]

        CFD_FindFace_Head = ["FindFace", "InputNas(FindFace)", "OutputFolder", "Set(FindFace)", "Tolerance(FindFace)"]

        CFD_GenerateIn_Head = ["GenerateInput", "InputNas(GenerateInput)", "OutputFolder",
                               "TimeFunction(GenerateInput)", "EndingArea(GenerateInput)", "AUIPath"]

        CFD_Postprogress_Head = ["OutputFolder", "DataOutput", "Idb Path(DataOutput)", "Por Path(DataOutput)",
                                 "AUI Path(DataOutput)", "TimeStart(DataOutput)", "TimeStop(DataOutput)",
                                 "CFD Post-processing", "TimeStart(CFD Post-processing)",
                                 "TimeStop(CFD Post-processing)", "Mask Path", "Extract Skin", "Value Exclusion",
                                 "Input Directory", "Input NPY Path", "Leaf Size", "CPU", "Filter Zero",
                                 "Match Ball Radius", "Output Directory", "Extract Skin", "Output Image File",
                                 "FFRCalculation", "InputMask(FFRCalculation)", "Keep Values(FFRCalculation)",
                                 "Paraview_processing", "Savevtu", "nasPath(Savevtu)", "ndInterestDictPath(Savevtu)",
                                 "ParamsNpyPath(Savevtu)", "savevtuName(Savevtu)", "addFFR(Savevtu)",
                                 "savesingle(Savevtu)", "TAPnpyPath(Savevtu)", "FFRValuePath(Savevtu)",
                                 "FFRarrayName(Savevtu)", "FFRSavename(Savevtu)", "ParamsPath(Savevtu)",
                                 "pvdName(Savevtu)", "SavePVD(Savevtu)", "SaveFFRvtu(Savevtu)", "Savevtufile(Savevtu)"]

        CFD_DataOutput_Head = ["OutputFolder", "DataOutput", "Idb Path(DataOutput)", "Por Path(DataOutput)",
                               "AUI Path(DataOutput)", "TimeStart(DataOutput)", "TimeStop(DataOutput)"]

        CFD_PostNpy_Head = ["OutputFolder", "CFD Post-processing", "InputFolder(CFD Post-processing)",
                            "TimeStart(CFD Post-processing)", "TimeStop(CFD Post-processing)"]

        CFD_Map_Result_Head = ["Mask Path", "Extract Skin", "Value Exclusion", "Input Directory", "Input NPY Path",
                               "Leaf Size", "CPU", "Filter Zero", "Match Ball Radius", "Output Directory",
                               "Extract Skin", "Output Image File"]

        CFD_FFRCalculation_Head = ["OutputFolder", "FFRCalculation", "InputMask(FFRCalculation)",
                                   "Keep Values(FFRCalculation)", "Load Value image for FFR Folder(FFRCalculation)"]

        CFD_Paraview_Head = ["OutputFolder", "Paraview_processing", "Nodecoo(Paraview_processing)",
                             "TetraElmnt(Paraview_processing)", "HexElmnt(Paraview_processing)",
                             "Times(Paraview_processing)", "ParamsDic(Paraview_processing)",
                             "PostResults(Paraview_processing)", "WSSParamDict(Paraview_processing)",
                             "WssDict(Paraview_processing)", "FFRIQRAverageValue(Paraview_processing)"]

        CFD_Savevtu_Head = ["OutputFolder", "Savevtu", "nasPath(Savevtu)", "ndInterestDictPath(Savevtu)",
                            "ParamsNpyPath(Savevtu)", "savevtuName(Savevtu)", "addFFR(Savevtu)", "savesingle(Savevtu)",
                            "TAPnpyPath(Savevtu)", "FFRValuePath(Savevtu)", "FFRarrayName(Savevtu)",
                            "FFRSavename(Savevtu)", "ParamsPath(Savevtu)", "pvdName(Savevtu)", "SavePVD(Savevtu)",
                            "SaveFFRvtu(Savevtu)", "Savevtufile(Savevtu)"]

        ## parameters
        resampleDepth = '0.1'
        CPRAngle = '0'
        cpus = '2'
        fillDistance = '0.8'
        processors = '4'
        interpolation = '1'
        # Xsection case setting
        interpolation_xsimg = '0'  # linear interpolation
        outputCSV_xsimg = '0'  # no ouput
        interpolation_xsmsk = '1'  # nearest neighbour
        outputCSV_xsmsk = '1'  # csv output

        # Murray's Law
        startSlc_AVC = '0'
        stopSlc_AVC = '10000'
        areaChc_AVC = 'averageArea'
        refFile_RCA_AVC = self.ui.AVCPath_Txt_BPP.toPlainText() + '/RCA.csv'
        # aorOut_AVC = self.ui.AVCPath_Txt_BPP.toPlainText() + '/Aor_in.txt'
        # aorIn_AVC = self.ui.AVCPath_Txt_BPP.toPlainText() + '/Aor_out.txt'
        # aorOutFFR_AVC = self.ui.AVCPath_Txt_BPP.toPlainText() + '/Aor_outFFR.txt'
        refFile_LCA_AVC = self.ui.AVCPath_Txt_BPP.toPlainText() + '/LCA.csv'
        refFile_N_AVC = 'N'
        totInflow_AVC = "3261.670409"
        diffThres_AVC = "0.01"
        step_AVC = "126"
        exo_AVC = "2.6"
        mul_AVC = "1,4.1667"
        # CFD_Mesh
        Tolerance_FindFace_Mesh = 45
        MeshExe = self.ui.MeshExePath_Txt_BPP.toPlainText()
        AUIExe = self.ui.AUIPath_Txt_BPP.toPlainText()
        # CFD_Postprogress
        TimeStart_DataOutput = 4.5
        TimeStop_DataOutput = 5.4
        TimeStart_IFR = 5.2
        TimeStop_IFR = 5.4
        FFR_KeepValue = 4

        for CSVPath in self.CSVPathList:
            loopIndex = self.CSVPathList.index(CSVPath)
            tmpMaskPath = os.path.join(self.OriDataPathList[loopIndex], 'sc_labelled_coronary_segment.nii.gz')
            AortaMaskPath = os.path.join(self.OriDataPathList[loopIndex], 'sc_final_labelmap2.nii.gz')
            tmpOutputPath = self.BranchPathList[loopIndex]
            # initialize case table
            tempCaseTable = []
            tempBranchList = []
            tempBranchLblist = []
            AVCBranMTable = []  # Murray's Law batch
            DCMDirPath = self.DCMDirList[loopIndex]

            # # validate which branch can be used for each case
            # tempJsonList = self.ReturnJsonList(CSVPath)
            UseJsonPath = os.path.join(CSVPath, 'coronarySegmentsUse.json')
            # BranchDictionary.ValidateUse(dicPath=self.DictionaryPath, jsonList=tempJsonList, outputPath=UseJsonPath)

            ## Aorta
            # filter aorta which label value is 4
            self.FilterMaskTable.append([AortaMaskPath, '4', '1', tmpOutputPath, 'Aorta'])
            # Aorta lumen correction
            tmpAortaXsecFixColumn = [self.AortaPathList[loopIndex], self.AortaPathList[loopIndex],
                                     self.AortaFixTablePath, 'Yes', self.XsectionFixPathList[loopIndex], 'AortaFix']
            self.XsectionFixTable.append(tmpAortaXsecFixColumn)
            # stack segmentation
            tempBranchList.append('AortaFixVal.nii.gz')
            tempBranchLblist.append(str(4))

            # validate the existence of json file first
            with open(UseJsonPath) as f:
                case = json.load(f)

            for branch in case:
                if branch['use']:

                    # store used branch for later Stack Segmentation use
                    tempBranchList.append(branch['original_volume_mask_fix'].replace('.nii.gz', 'Val.nii.gz'))
                    tempBranchLblist.append(str(branch['label'][0]))

                    # filter mask table
                    tempFMColumn = [tmpMaskPath, str(branch['label'][0]), '1', tmpOutputPath, branch['original_branch']]
                    self.FilterMaskTable.append(tempFMColumn)

                    # generate Xsection table columns for branches for ease case
                    tempCaseColumn1 = [interpolation_xsimg,
                                       outputCSV_xsimg,
                                       'CTA.nii.gz',
                                       branch['original_csv'],
                                       branch['cross_section_image_LI'],
                                       branch['3D_csv']]
                    tempCaseTable.append(tempCaseColumn1)

                    tempCaseColumn2 = [interpolation_xsmsk,
                                       outputCSV_xsmsk,
                                       branch['original_branch'] + 'Val.nii.gz',
                                       branch['original_csv'],
                                       branch['cross_section_mask_NN'], branch['3D_csv']]
                    tempCaseTable.append(tempCaseColumn2)

                    # insert columns into xsection fix table
                    XsecCTAPath = os.path.join(self.XsectionPathList[loopIndex], branch['cross_section_image_LI'])
                    XsecMaskPath = os.path.join(self.XsectionPathList[loopIndex], branch['cross_section_mask_NN'])
                    tmpXsecFixColumn = [XsecCTAPath, XsecMaskPath, self.LCProcessPath, 'Yes',
                                        self.XsectionFixPathList[loopIndex],
                                        branch['cross_section_mask_fix'].replace('Val.nii.gz', '')]

                    self.XsectionFixTable.append(tmpXsecFixColumn)

                    # insert columns into extension table
                    tmpExtInputMaskPath = os.path.join(self.XsectionFixPathList[loopIndex],
                                                       branch['cross_section_mask_fix'])
                    tmpExtOutputMaskPath = os.path.join(self.XsectionFixExtPathList[loopIndex],
                                                        branch['cross_section_mask_fix_extension'])
                    tmpExtColumn = [tmpExtInputMaskPath, tmpExtOutputMaskPath, 'Yes', 'X', 'No', '0', '0', '0', '46',
                                    '169', 'Yes', 'Yes', 'X', 'No', '0', '0', '0', '66', '169']
                    self.XsectionFixExtTable.append(tmpExtColumn)

                    tmpOriColumn = [self.XsectionFixExtPathList[loopIndex],
                                    'CTA.nii.gz',
                                    branch['cross_section_mask_fix_extension'],
                                    os.path.join(self.XsectionPathList[loopIndex], branch['3D_csv']),
                                    fillDistance,
                                    processors,
                                    self.OriVolumePathList[loopIndex],
                                    branch['original_volume_mask_fix'],
                                    interpolation]
                    self.OriVolumeTable.append(tmpOriColumn)

                    ## area volume calculation branch map
                    refFile_AVC = refFile_N_AVC
                    if branch['branch_name'] == 'pRCA':
                        refFile_AVC = refFile_RCA_AVC
                    elif branch['branch_name'] == 'LM':
                        refFile_AVC = refFile_LCA_AVC

                    tmpAVCBranchMCol = [
                        branch['branch_name'],
                        branch["branch_label"],
                        tmpExtInputMaskPath,
                        startSlc_AVC,
                        stopSlc_AVC,
                        str(branch['label'][0]),
                        areaChc_AVC,
                        refFile_AVC
                    ]
                    AVCBranMTable.append(tmpAVCBranchMCol)

                    # insert columns into OriVolume fix table
                    OriVCTAPath = os.path.join(self.OriVolumePathList[loopIndex], "CTA.nii.gz")
                    OriVMaskPath = os.path.join(self.OriVolumePathList[loopIndex], branch['original_volume_mask_fix'])
                    tmpOriVFixColumn = [OriVCTAPath,
                                        OriVMaskPath,
                                        self.OCProcessPath,
                                        'Yes',
                                        self.OriVolumePathList[loopIndex],
                                        branch['original_volume_mask_fix'].replace('.nii.gz', '')]

                    self.OriVolumeFixTable.append(tmpOriVFixColumn)

                else:
                    continue

            ## create centerline to xsection two tables
            # case table csv file for each case
            caseFilePath = os.path.join(CSVPath, 'case.csv')
            case_df = pd.DataFrame(tempCaseTable, columns=Case_Head)
            _write_dataframe_atomic(case_df, caseFilePath)

            # add record to overall Xsection table
            self.XsectionTable.append([self.BranchPathList[loopIndex], CSVPath, self.XsectionPathList[loopIndex],
                                       self.XsectionPathList[loopIndex], resampleDepth, CPRAngle, cpus,
                                       caseFilePath, self.CPREXEPath])

            # create Stack Segmentation Column for each case
            tmpSSColumn = [os.path.join(self.OriDataPathList[loopIndex], 'CTA.nii.gz'),
                           self.OriVolumePathList[loopIndex],
                           ','.join(tempBranchList),
                           '0',
                           ','.join(tempBranchLblist),
                           self.OriVolumePathList[loopIndex],
                           'OriVolumeCA']
            self.SSTable.append(tmpSSColumn)

            # mask to stl table
            tmpSGColumn = [self.STLEXEPath, self.FinalMaskPathList[loopIndex], self.StlFilePathList[loopIndex]]
            self.SGTable.append(tmpSGColumn)

            # CFD table
            tmpCFDColumn = [self.StlFilePathList[loopIndex], os.path.join(DCMDirPath, 'CFD'), True, True, MeshExe,
                            True, Tolerance_FindFace_Mesh, self.CFD_setTablePath, True, AUIExe,
                            os.path.join(self.ArVolCalPathList[loopIndex], 'Flow_1_0')]
            tmpFFRColumn = [self.MeshFilePathList[loopIndex], os.path.join(DCMDirPath, 'FFR'), False, False, MeshExe,
                            True, Tolerance_FindFace_Mesh, self.CFD_setTablePath, True, AUIExe,
                            os.path.join(self.ArVolCalPathList[loopIndex], 'Flow_4_1667')]
            self.CFD_MeshTable.append(tmpCFDColumn)
            if self.getFFRresult:
                self.CFD_MeshTable.append(tmpFFRColumn)

            # STLcut table
            tmpSTLcutColumn = [True, os.path.join(DCMDirPath, 'CFD/STL/Raw.stl'), os.path.join(DCMDirPath, 'CFD')]
            self.CFD_STLcutTable.append(tmpSTLcutColumn)

            # STLmesh table
            tmpSTLmeshColumn = [True, os.path.join(DCMDirPath, 'CFD/MeshCut/lumenCut.stl'), MeshExe,
                                os.path.join(DCMDirPath, 'CFD')]
            self.CFD_STLmeshTable.append(tmpSTLmeshColumn)

            # FindFace table
            tmpFindFaceColumn = [True, os.path.join(DCMDirPath, 'CFD/Mesh/FluidMeshInit.nas'),
                                 os.path.join(DCMDirPath, 'CFD'), self.CFD_setTablePath, Tolerance_FindFace_Mesh]
            tmpFFRFindFaceColumn = [True, os.path.join(DCMDirPath, 'CFD/Mesh/FluidMeshInit.nas'),
                                    os.path.join(DCMDirPath, 'FFR'), self.CFD_setTablePath, Tolerance_FindFace_Mesh]
            self.CFD_FindFaceTable.append(tmpFindFaceColumn)
            if self.getFFRresult:
                self.CFD_FindFaceTable.append(tmpFFRFindFaceColumn)

            # GenerateIn table
            tmpGenerateInColumn = [True, os.path.join(DCMDirPath, 'CFD/FindFace/CFD.nas'),
                                   os.path.join(DCMDirPath, 'CFD'),
                                   os.path.join(self.ArVolCalPathList[loopIndex], 'Flow_1_0'),
                                   os.path.join(DCMDirPath, 'CFD/FindFace/area.csv'), AUIExe]
            tmpFFRGenerateInColumn = [True, os.path.join(DCMDirPath, 'FFR/FindFace/CFD.nas'),
                                      os.path.join(DCMDirPath, 'FFR'),
                                      os.path.join(self.ArVolCalPathList[loopIndex], 'Flow_4_1667'),
                                      os.path.join(DCMDirPath, 'FFR/FindFace/area.csv'), AUIExe]
            self.CFD_GenerateInTable.append(tmpGenerateInColumn)
            if self.getFFRresult:
                self.CFD_GenerateInTable.append(tmpFFRGenerateInColumn)

            # PostProgress table
            tmpPostProgressColumn = [os.path.join(DCMDirPath, 'PostProgress'), True,
                                     os.path.join(DCMDirPath, 'CFD/ADINA/CFD.idb'),
                                     os.path.join(DCMDirPath, 'CFD/result/CFD.por'), AUIExe, TimeStart_DataOutput,
                                     TimeStop_DataOutput, True, TimeStart_DataOutput, TimeStop_DataOutput,
                                     os.path.join(self.OriVolumePathList[loopIndex], 'OriVolumeCAVal.nii.gz'),
                                     False, 100, os.path.join(DCMDirPath, 'PostProgress/PostNpy'),
                                     'ndCooOSI_2dndarr.npy,'
                                     'ndCooRRT_2dndarr.npy,ndCooTAWSS_2dndarr.npy,ndCooTAWSS_VectorMag_2dndarr.npy,'
                                     'ndTimeAvePressureCoo_2dndarr.npy,ndTimeIQRPressureCoo_2dndarr.npy,'
                                     'ndTimeKurtosisPressureCoo_2dndarr.npy,ndTimeMaxPressureCoo_2dndarr.npy,'
                                     'ndTimeMedianPressureCoo_2dndarr.npy,ndTimeMinPressureCoo_2dndarr.npy,'
                                     'ndTimePTPPressureCoo_2dndarr.npy,ndTimeQ1PressureCoo_2dndarr.npy,'
                                     'ndTimeQ3PressureCoo_2dndarr.npy,ndTimeSkewPressureCoo_2dndarr.npy,'
                                     'ndTimeSTDPressureCoo_2dndarr.npy,ndTimeVarPressureCoo_2dndarr.npy,'
                                     'ndTimeIQRWSSCoo_2dndarr.npy,ndTimeKurtosisWSSCoo_2dndarr.npy,'
                                     'ndTimeMaxWSSCoo_2dndarr.npy,ndTimeMedianWSSCoo_2dndarr.npy,'
                                     'ndTimeMinWSSCoo_2dndarr.npy,ndTimePTPWSSCoo_2dndarr.npy,'
                                     'ndTimeQ1WSSCoo_2dndarr.npy,ndTimeQ3WSSCoo_2dndarr.npy,'
                                     'ndTimeSkewWSSCoo_2dndarr.npy,ndTimeSTDWSSCoo_2dndarr.npy,'
                                     'ndTimeVarWSSCoo_2dndarr.npy', 666, 5, True, 1,
                                     os.path.join(DCMDirPath, 'PostProgress/MapCT'), False, 'CTOSI.nii.gz,'
                                                                                            'CTRRT.nii.gz,CTTAWSS.nii.gz,CTTAWSS_VectorMag.nii.gz,CTTimeAvePressure.nii.gz,'
                                                                                            'CTTimeIQRPressure.nii.gz,CTTimeKurtosisPressure.nii.gz,CTTimeMaxPressure.nii.gz,'
                                                                                            'CTTimeMedianPressure.nii.gz,CTTimeMinPressure.nii.gz,CTTimePTPPressure.nii.gz,'
                                                                                            'CTTimeQ1Pressure.nii.gz,CTTimeQ3Pressure.nii.gz,CTTimeSkewPressure.nii.gz,'
                                                                                            'CTTimeSTDPressure.nii.gz,CTTimeVarPressure.nii.gz,CTTimeIQRWSS.nii.gz,'
                                                                                            'CTTimeKurtosisWSS.nii.gz,CTTimeMaxWSS.nii.gz,CTTimeMedianWSS.nii.gz,'
                                                                                            'CTTimeMinWSS.nii.gz,CTTimePTPWSS.nii.gz,CTTimeQ1WSS.nii.gz,'
                                                                                            'CTTimeQ3WSS.nii.gz,CTTimeSkewWSS.nii.gz,CTTimeSTDWSS.nii.gz,CTTimeVarWSS.nii.gz',
                                     True, self.OriVolumePathList[loopIndex], FFR_KeepValue, True, True,
                                     os.path.join(DCMDirPath, 'CFD/FindFace/3Dmesh.nas'),
                                     os.path.join(DCMDirPath, 'PostProgress/PostNpy/ndInterestDict_dict.npy'),
                                     os.path.join(DCMDirPath, 'PostProgress/PostNpy'), 'CFDresult', True, False,
                                     os.path.join(DCMDirPath, 'PostProgress/PostNpy/ndTimeAvePressureCoo_2dndarr.npy'),
                                     os.path.join(DCMDirPath, 'PostProgress/FFR/CTFFRIQRAverageValue.npy'), 'PDPA', '',
                                     os.path.join(DCMDirPath, 'PostProgress/Npy/Fluid_lst_ParamsDic.npy'), 'CFDresult',
                                     True, False, True]
            tmpFFRPostProgressColumn = [os.path.join(DCMDirPath, 'FFRPostProgress'), True,
                                        os.path.join(DCMDirPath, 'FFR/ADINA/CFD.idb'),
                                        os.path.join(DCMDirPath, 'FFR/result/CFD.por'), AUIExe, TimeStart_DataOutput,
                                        TimeStop_DataOutput, True, TimeStart_DataOutput, TimeStop_DataOutput,
                                        os.path.join(self.OriVolumePathList[loopIndex], 'OriVolumeCAVal.nii.gz'), False,
                                        100, os.path.join(DCMDirPath, 'FFRPostProgress/PostNpy'),
                                        'ndCooOSI_2dndarr.npy,ndCooRRT_2dndarr.npy,ndCooTAWSS_2dndarr.npy,'
                                        'ndCooTAWSS_VectorMag_2dndarr.npy,ndTimeAvePressureCoo_2dndarr.npy,'
                                        'ndTimeIQRPressureCoo_2dndarr.npy,ndTimeKurtosisPressureCoo_2dndarr.npy,'
                                        'ndTimeMaxPressureCoo_2dndarr.npy,ndTimeMedianPressureCoo_2dndarr.npy,'
                                        'ndTimeMinPressureCoo_2dndarr.npy,ndTimePTPPressureCoo_2dndarr.npy,'
                                        'ndTimeQ1PressureCoo_2dndarr.npy,ndTimeQ3PressureCoo_2dndarr.npy,'
                                        'ndTimeSkewPressureCoo_2dndarr.npy,ndTimeSTDPressureCoo_2dndarr.npy,'
                                        'ndTimeVarPressureCoo_2dndarr.npy,ndTimeIQRWSSCoo_2dndarr.npy,'
                                        'ndTimeKurtosisWSSCoo_2dndarr.npy,ndTimeMaxWSSCoo_2dndarr.npy,'
                                        'ndTimeMedianWSSCoo_2dndarr.npy,ndTimeMinWSSCoo_2dndarr.npy,'
                                        'ndTimePTPWSSCoo_2dndarr.npy,ndTimeQ1WSSCoo_2dndarr.npy,'
                                        'ndTimeQ3WSSCoo_2dndarr.npy,ndTimeSkewWSSCoo_2dndarr.npy,'
                                        'ndTimeSTDWSSCoo_2dndarr.npy,ndTimeVarWSSCoo_2dndarr.npy', 666, 5, True, 1,
                                        os.path.join(DCMDirPath, 'FFRPostProgress/MapCT'), False,
                                        'CTOSI.nii.gz,CTRRT.nii.gz,CTTAWSS.nii.gz,CTTAWSS_VectorMag.nii.gz,'
                                        'CTTimeAvePressure.nii.gz,CTTimeIQRPressure.nii.gz,'
                                        'CTTimeKurtosisPressure.nii.gz,CTTimeMaxPressure.nii.gz,'
                                        'CTTimeMedianPressure.nii.gz,CTTimeMinPressure.nii.gz,'
                                        'CTTimePTPPressure.nii.gz,CTTimeQ1Pressure.nii.gz,CTTimeQ3Pressure.nii.gz,'
                                        'CTTimeSkewPressure.nii.gz,CTTimeSTDPressure.nii.gz,CTTimeVarPressure.nii.gz,'
                                        'CTTimeIQRWSS.nii.gz,CTTimeKurtosisWSS.nii.gz,CTTimeMaxWSS.nii.gz,'
                                        'CTTimeMedianWSS.nii.gz,CTTimeMinWSS.nii.gz,CTTimePTPWSS.nii.gz,'
                                        'CTTimeQ1WSS.nii.gz,CTTimeQ3WSS.nii.gz,CTTimeSkewWSS.nii.gz,'
                                        'CTTimeSTDWSS.nii.gz,CTTimeVarWSS.nii.gz', True,
                                        self.OriVolumePathList[loopIndex], FFR_KeepValue, True, True,
                                        os.path.join(DCMDirPath, 'PostProgress/Vtu/CFDresult.vtu'), '', '', '', '', '',
                                        os.path.join(DCMDirPath,
                                                     'FFRPostProgress/PostNpy/ndTimeAvePressureCoo_2dndarr.npy'),
                                        os.path.join(DCMDirPath, 'FFRPostProgress/FFR/CTFFRIQRAverageValue.npy'),
                                        'FFR', 'FFRresult', '', '', False, True, False]
            tmpiFRPostProgressColumn = [os.path.join(DCMDirPath, 'IFRPostProgress'), True,
                                        os.path.join(DCMDirPath, 'CFD/ADINA/CFD.idb'),
                                        os.path.join(DCMDirPath, 'CFD/result/CFD.por'), AUIExe, TimeStart_DataOutput,
                                        TimeStop_DataOutput, True, TimeStart_IFR, TimeStop_IFR,
                                        os.path.join(self.OriVolumePathList[loopIndex], 'OriVolumeCAVal.nii.gz'), False,
                                        100, os.path.join(DCMDirPath, 'IFRPostProgress/PostNpy'),
                                        'ndCooOSI_2dndarr.npy,ndCooRRT_2dndarr.npy,ndCooTAWSS_2dndarr.npy,'
                                        'ndCooTAWSS_VectorMag_2dndarr.npy,ndTimeAvePressureCoo_2dndarr.npy,'
                                        'ndTimeIQRPressureCoo_2dndarr.npy,ndTimeKurtosisPressureCoo_2dndarr.npy,'
                                        'ndTimeMaxPressureCoo_2dndarr.npy,ndTimeMedianPressureCoo_2dndarr.npy,'
                                        'ndTimeMinPressureCoo_2dndarr.npy,ndTimePTPPressureCoo_2dndarr.npy,'
                                        'ndTimeQ1PressureCoo_2dndarr.npy,ndTimeQ3PressureCoo_2dndarr.npy,'
                                        'ndTimeSkewPressureCoo_2dndarr.npy,ndTimeSTDPressureCoo_2dndarr.npy,'
                                        'ndTimeVarPressureCoo_2dndarr.npy,ndTimeIQRWSSCoo_2dndarr.npy,'
                                        'ndTimeKurtosisWSSCoo_2dndarr.npy,ndTimeMaxWSSCoo_2dndarr.npy,'
                                        'ndTimeMedianWSSCoo_2dndarr.npy,ndTimeMinWSSCoo_2dndarr.npy,'
                                        'ndTimePTPWSSCoo_2dndarr.npy,ndTimeQ1WSSCoo_2dndarr.npy,'
                                        'ndTimeQ3WSSCoo_2dndarr.npy,ndTimeSkewWSSCoo_2dndarr.npy,'
                                        'ndTimeSTDWSSCoo_2dndarr.npy,ndTimeVarWSSCoo_2dndarr.npy', 666, 5, True, 1,
                                        os.path.join(DCMDirPath, 'IFRPostProgress/MapCT'), False, 'CTOSI.nii.gz,'
                                                                                                  'CTRRT.nii.gz,CTTAWSS.nii.gz,CTTAWSS_VectorMag.nii.gz,CTTimeAvePressure.nii.gz,'
                                                                                                  'CTTimeIQRPressure.nii.gz,CTTimeKurtosisPressure.nii.gz,'
                                                                                                  'CTTimeMaxPressure.nii.gz,CTTimeMedianPressure.nii.gz,CTTimeMinPressure.nii.gz,'
                                                                                                  'CTTimePTPPressure.nii.gz,CTTimeQ1Pressure.nii.gz,CTTimeQ3Pressure.nii.gz,'
                                                                                                  'CTTimeSkewPressure.nii.gz,CTTimeSTDPressure.nii.gz,CTTimeVarPressure.nii.gz,'
                                                                                                  'CTTimeIQRWSS.nii.gz,CTTimeKurtosisWSS.nii.gz,CTTimeMaxWSS.nii.gz,'
                                                                                                  'CTTimeMedianWSS.nii.gz,CTTimeMinWSS.nii.gz,CTTimePTPWSS.nii.gz,'
                                                                                                  'CTTimeQ1WSS.nii.gz,CTTimeQ3WSS.nii.gz,CTTimeSkewWSS.nii.gz,'
                                                                                                  'CTTimeSTDWSS.nii.gz,CTTimeVarWSS.nii.gz',
                                        True,
                                        self.OriVolumePathList[loopIndex], FFR_KeepValue, True, True,
                                        os.path.join(DCMDirPath, 'FFRPostProgress/Vtu/FFRresult.vtu'), '', '', '', '',
                                        '',
                                        os.path.join(DCMDirPath,
                                                     'IFRPostProgress/PostNpy/ndTimeAvePressureCoo_2dndarr.npy'),
                                        os.path.join(DCMDirPath, 'IFRPostProgress/FFR/CTFFRIQRAverageValue.npy'),
                                        'IFR', 'IFRresult', '', '', False, True, False]
            self.CFD_PostProgressTable.append(tmpPostProgressColumn)
            if self.getFFRresult:
                self.CFD_PostProgressTable.append(tmpFFRPostProgressColumn)
            if self.getIFRresult:
                self.CFD_PostProgressTable.append(tmpiFRPostProgressColumn)

            # DataOutput table
            tmpDataOutputColumn = [os.path.join(DCMDirPath, 'PostProgress'), True,
                                   os.path.join(DCMDirPath, 'CFD/ADINA/CFD.idb'),
                                   os.path.join(DCMDirPath, 'CFD/result/CFD.por'), AUIExe, TimeStart_DataOutput,
                                   TimeStop_DataOutput]
            tmpFFRDataOutputColumn = [os.path.join(DCMDirPath, 'FFRPostProgress'), True,
                                      os.path.join(DCMDirPath, 'FFR/ADINA/CFD.idb'),
                                      os.path.join(DCMDirPath, 'FFR/result/CFD.por'), AUIExe, TimeStart_DataOutput,
                                      TimeStop_DataOutput]
            self.DataOutputTable.append(tmpDataOutputColumn)
            if self.getFFRresult:
                self.DataOutputTable.append(tmpFFRDataOutputColumn)

            # PostNpy table
            tmpPostNpyColumn = [os.path.join(DCMDirPath, 'PostProgress'), True,
                                os.path.join(DCMDirPath, 'PostProgress/Npy'), TimeStart_DataOutput, TimeStop_DataOutput]
            tmpFFRPostNpyColumn = [os.path.join(DCMDirPath, 'FFRPostProgress'), True,
                                   os.path.join(DCMDirPath, 'FFRPostProgress/Npy'), TimeStart_DataOutput,
                                   TimeStop_DataOutput]
            tmpIFRPostNpyColumn = [os.path.join(DCMDirPath, 'IFRPostProgress'), True,
                                   os.path.join(DCMDirPath, 'PostProgress/Npy'), TimeStart_IFR, TimeStop_IFR]
            self.PostNpyTable.append(tmpPostNpyColumn)
            if self.getFFRresult:
                self.PostNpyTable.append(tmpFFRPostNpyColumn)
            if self.getIFRresult:
                self.PostNpyTable.append(tmpIFRPostNpyColumn)

            # Map_Result table
            # tmpMap_ResultColumn = [os.path.join(DCMDirPath,'PostProgress'),True,
            #                        os.path.join(self.OriVolumePathList[loopIndex],'OriVolumeCAVal.nii.gz'),
            #                        os.path.join(DCMDirPath,'PostProgress/PostNpy')]
            # tmpFFRMap_ResultColumn = [os.path.join(DCMDirPath,'FFRPostProgress'),True,
            #                           os.path.join(self.OriVolumePathList[loopIndex],'OriVolumeCAVal.nii.gz'),
            #                           os.path.join(DCMDirPath,'FFRPostProgress/PostNpy')]
            ndarrstr = 'ndCooOSI_2dndarr.npy,ndCooRRT_2dndarr.npy,ndCooTAWSS_2dndarr.npy,' \
                       'ndCooTAWSS_VectorMag_2dndarr.npy,ndTimeAvePressureCoo_2dndarr.npy,' \
                       'ndTimeDecile1PressureCoo_2dndarr.npy,ndTimeDecile1WSSCoo_2dndarr.npy,' \
                       'ndTimeDecile2PressureCoo_2dndarr.npy,ndTimeDecile2WSSCoo_2dndarr.npy,' \
                       'ndTimeDecile3PressureCoo_2dndarr.npy,ndTimeDecile3WSSCoo_2dndarr.npy,' \
                       'ndTimeDecile4PressureCoo_2dndarr.npy,ndTimeDecile4WSSCoo_2dndarr.npy,' \
                       'ndTimeDecile6PressureCoo_2dndarr.npy,ndTimeDecile6WSSCoo_2dndarr.npy,' \
                       'ndTimeDecile7PressureCoo_2dndarr.npy,ndTimeDecile7WSSCoo_2dndarr.npy,' \
                       'ndTimeDecile8PressureCoo_2dndarr.npy,ndTimeDecile8WSSCoo_2dndarr.npy,' \
                       'ndTimeDecile9PressureCoo_2dndarr.npy,ndTimeDecile9WSSCoo_2dndarr.npy,' \
                       'ndTimeEnergyPressureCoo_2dndarr.npy,ndTimeEnergyWSSCoo_2dndarr.npy,' \
                       'ndTimeEntropyPressureCoo_2dndarr.npy,ndTimeEntropyWSSCoo_2dndarr.npy,' \
                       'ndTimeGMeanPressureCoo_2dndarr.npy,ndTimeGMeanWSSCoo_2dndarr.npy,' \
                       'ndTimeHMeanPressureCoo_2dndarr.npy,ndTimeHMeanWSSCoo_2dndarr.npy,' \
                       'ndTimeIQRPressureCoo_2dndarr.npy,ndTimeIQRWSSCoo_2dndarr.npy,' \
                       'ndTimeKurtosisPressureCoo_2dndarr.npy,ndTimeKurtosisWSSCoo_2dndarr.npy,' \
                       'ndTimeMaxPressureCoo_2dndarr.npy,ndTimeMaxWSSCoo_2dndarr.npy,' \
                       'ndTimeMedianPressureCoo_2dndarr.npy,ndTimeMedianWSSCoo_2dndarr.npy,' \
                       'ndTimeMinPressureCoo_2dndarr.npy,ndTimeMinWSSCoo_2dndarr.npy,' \
                       'ndTimeModePressureCoo_2dndarr.npy,ndTimeModeWSSCoo_2dndarr.npy,' \
                       'ndTimePTPPressureCoo_2dndarr.npy,ndTimePTPWSSCoo_2dndarr.npy,' \
                       'ndTimeQ1PressureCoo_2dndarr.npy,ndTimeQ1WSSCoo_2dndarr.npy,ndTimeQ3PressureCoo_2dndarr.npy,' \
                       'ndTimeQ3WSSCoo_2dndarr.npy,ndTimeRMSPressureCoo_2dndarr.npy,ndTimeRMSWSSCoo_2dndarr.npy,' \
                       'ndTimeSEPressureCoo_2dndarr.npy,ndTimeSEWSSCoo_2dndarr.npy,ndTimeSkewPressureCoo_2dndarr.npy,' \
                       'ndTimeSkewWSSCoo_2dndarr.npy,ndTimeSTDPressureCoo_2dndarr.npy,ndTimeSTDWSSCoo_2dndarr.npy,' \
                       'ndTimeTriMeanPressureCoo_2dndarr.npy,ndTimeTriMeanWSSCoo_2dndarr.npy,' \
                       'ndTimeVarPressureCoo_2dndarr.npy,ndTimeVarWSSCoo_2dndarr.npy'
            niistr = 'CTOSI.nii.gz,CTRRT.nii.gz,CTTAWSS.nii.gz,CTTAWSS_VectorMag.nii.gz,CTTimeAvePressure.nii.gz,' \
                     'CTTimeDecile1Pressure.nii.gz,CTTimeDecile1WSS.nii.gz,CTTimeDecile2Pressure.nii.gz,' \
                     'CTTimeDecile2WSS.nii.gz,CTTimeDecile3Pressure.nii.gz,CTTimeDecile3WSS.nii.gz,' \
                     'CTTimeDecile4Pressure.nii.gz,CTTimeDecile4WSS.nii.gz,CTTimeDecile6Pressure.nii.gz,' \
                     'CTTimeDecile6WSS.nii.gz,CTTimeDecile7Pressure.nii.gz,CTTimeDecile7WSS.nii.gz,' \
                     'CTTimeDecile8Pressure.nii.gz,CTTimeDecile8WSS.nii.gz,CTTimeDecile9Pressure.nii.gz,' \
                     'CTTimeDecile9WSS.nii.gz,CTTimeEnergyPressure.nii.gz,CTTimeEnergyWSS.nii.gz,' \
                     'CTTimeEntropyPressure.nii.gz,CTTimeEntropyWSS.nii.gz,CTTimeGMeanPressure.nii.gz,' \
                     'CTTimeGMeanWSS.nii.gz,CTTimeHMeanPressure.nii.gz,CTTimeHMeanWSS.nii.gz,' \
                     'CTTimeIQRPressure.nii.gz,CTTimeIQRWSS.nii.gz,CTTimeKurtosisPressure.nii.gz,' \
                     'CTTimeKurtosisWSS.nii.gz,CTTimeMaxPressure.nii.gz,CTTimeMaxWSS.nii.gz,' \
                     'CTTimeMedianPressure.nii.gz,CTTimeMedianWSS.nii.gz,CTTimeMinPressure.nii.gz,' \
                     'CTTimeMinWSS.nii.gz,CTTimeModePressure.nii.gz,CTTimeModeWSS.nii.gz,CTTimePTPPressure.nii.gz,' \
                     'CTTimePTPWSS.nii.gz,CTTimeQ1Pressure.nii.gz,CTTimeQ1WSS.nii.gz,CTTimeQ3Pressure.nii.gz,' \
                     'CTTimeQ3WSS.nii.gz,CTTimeRMSPressure.nii.gz,CTTimeRMSWSS.nii.gz,CTTimeSEPressure.nii.gz,' \
                     'CTTimeSEWSS.nii.gz,CTTimeSkewPressure.nii.gz,CTTimeSkewWSS.nii.gz,CTTimeSTDPressure.nii.gz,' \
                     'CTTimeSTDWSS.nii.gz,CTTimeTriMeanPressure.nii.gz,CTTimeTriMeanWSS.nii.gz,' \
                     'CTTimeVarPressure.nii.gz,CTTimeVarWSS.nii.gz'
            tmpMap_ResultColumn = [os.path.join(self.OriVolumePathList[loopIndex], 'OriVolumeCAVal.nii.gz'), False, 100,
                                   os.path.join(DCMDirPath, 'PostProgress/PostNpy'), ndarrstr, 666, 5, True, 1,
                                   os.path.join(DCMDirPath, 'PostProgress/MapCT'), False, niistr]
            tmpFFRMap_ResultColumn = [os.path.join(self.OriVolumePathList[loopIndex], 'OriVolumeCAVal.nii.gz'), False,
                                      100, os.path.join(DCMDirPath, 'FFRPostProgress/PostNpy'), ndarrstr, 666, 5, True,
                                      1, os.path.join(DCMDirPath, 'FFRPostProgress/MapCT'), False, niistr]
            tmpIFRMap_ResultColumn = [os.path.join(self.OriVolumePathList[loopIndex], 'OriVolumeCAVal.nii.gz'), False,
                                      100, os.path.join(DCMDirPath, 'IFRPostProgress/PostNpy'), ndarrstr, 666, 5, True,
                                      1, os.path.join(DCMDirPath, 'IFRPostProgress/MapCT'), False, niistr]
            self.Map_ResultTable.append(tmpMap_ResultColumn)
            if self.getFFRresult:
                self.Map_ResultTable.append(tmpFFRMap_ResultColumn)
            if self.getIFRresult:
                self.Map_ResultTable.append(tmpIFRMap_ResultColumn)

            # FFRCalculation table
            tmpFFRCalculationColumn = [os.path.join(DCMDirPath, 'PostProgress'), True,
                                       os.path.join(self.OriVolumePathList[loopIndex], 'OriVolumeCAVal.nii.gz'),
                                       FFR_KeepValue,
                                       os.path.join(DCMDirPath, 'PostProgress/MapCT/CTTimeAvePressure.nii.gz')]
            tmpFFRFFRCalculationColumn = [os.path.join(DCMDirPath, 'FFRPostProgress'), True,
                                          os.path.join(self.OriVolumePathList[loopIndex], 'OriVolumeCAVal.nii.gz'),
                                          FFR_KeepValue,
                                          os.path.join(DCMDirPath, 'FFRPostProgress/MapCT/CTTimeAvePressure.nii.gz')]
            tmpIFRFFRCalculationColumn = [os.path.join(DCMDirPath, 'IFRPostProgress'), True,
                                          os.path.join(self.OriVolumePathList[loopIndex], 'OriVolumeCAVal.nii.gz'),
                                          FFR_KeepValue,
                                          os.path.join(DCMDirPath, 'IFRPostProgress/MapCT/CTTimeAvePressure.nii.gz')]
            self.FFRCalculationTable.append(tmpFFRCalculationColumn)
            if self.getFFRresult:
                self.FFRCalculationTable.append(tmpFFRFFRCalculationColumn)
            if self.getIFRresult:
                self.FFRCalculationTable.append(tmpIFRFFRCalculationColumn)

            # Paraview table
            tmpParaviewColumn = [os.path.join(DCMDirPath, 'PostProgress'), True,
                                 os.path.join(DCMDirPath, 'PostProgress/Npy/Fluid_Nodecoo_Dic.npy'),
                                 os.path.join(DCMDirPath, 'PostProgress/Npy/Fluid_TetraElmnt_NdIfo_Dic.npy'),
                                 os.path.join(DCMDirPath, 'PostProgress/Npy/Fluid_HexElmnt_NdIfo_Dic.npy'),
                                 os.path.join(DCMDirPath, 'PostProgress/Npy/Fluid_Times_Dic.npy'),
                                 os.path.join(DCMDirPath, 'PostProgress/Npy/Fluid_lst_ParamsDic.npy'),
                                 os.path.join(DCMDirPath, 'PostProgress/postNpy/fluid_PostResults_Dict_dict.npy'),
                                 os.path.join(DCMDirPath, 'PostProgress/postNpy/fluid_TimeAvePressureDict_dict.npy'),
                                 os.path.join(DCMDirPath, 'PostProgress/postNpy/fluid_PostResults_WssDict_dict.npy'),
                                 os.path.join(DCMDirPath, 'PostProgress/FFR/CTFFRIQRAverageValue.npy')]
            tmpFFRParaviewColumn = [os.path.join(DCMDirPath, 'FFRPostProgress'), True,
                                    os.path.join(DCMDirPath, 'FFRPostProgress/Npy/Fluid_Nodecoo_Dic.npy'),
                                    os.path.join(DCMDirPath, 'FFRPostProgress/Npy/Fluid_TetraElmnt_NdIfo_Dic.npy'),
                                    os.path.join(DCMDirPath, 'FFRPostProgress/Npy/Fluid_HexElmnt_NdIfo_Dic.npy'),
                                    os.path.join(DCMDirPath, 'FFRPostProgress/Npy/Fluid_Times_Dic.npy'),
                                    os.path.join(DCMDirPath, 'FFRPostProgress/Npy/Fluid_lst_ParamsDic.npy'),
                                    os.path.join(DCMDirPath, 'FFRPostProgress/postNpy/fluid_PostResults_Dict_dict.npy'),
                                    os.path.join(DCMDirPath,
                                                 'FFRPostProgress/postNpy/fluid_TimeAvePressureDict_dict.npy'),
                                    os.path.join(DCMDirPath,
                                                 'FFRPostProgress/postNpy/fluid_PostResults_WssDict_dict.npy'),
                                    os.path.join(DCMDirPath, 'FFRPostProgress/FFR/CTFFRIQRAverageValue.npy')]
            tmpIFRParaviewColumn = [os.path.join(DCMDirPath, 'IFRPostProgress'), True,
                                    os.path.join(DCMDirPath, 'PostProgress/Npy/Fluid_Nodecoo_Dic.npy'),
                                    os.path.join(DCMDirPath, 'PostProgress/Npy/Fluid_TetraElmnt_NdIfo_Dic.npy'),
                                    os.path.join(DCMDirPath, 'PostProgress/Npy/Fluid_HexElmnt_NdIfo_Dic.npy'),
                                    os.path.join(DCMDirPath, 'PostProgress/Npy/Fluid_Times_Dic.npy'),
                                    os.path.join(DCMDirPath, 'PostProgress/Npy/Fluid_lst_ParamsDic.npy'),
                                    os.path.join(DCMDirPath, 'IFRPostProgress/postNpy/fluid_PostResults_Dict_dict.npy'),
                                    os.path.join(DCMDirPath,
                                                 'IFRPostProgress/postNpy/fluid_TimeAvePressureDict_dict.npy'),
                                    os.path.join(DCMDirPath,
                                                 'IFRPostProgress/postNpy/fluid_PostResults_WssDict_dict.npy'),
                                    os.path.join(DCMDirPath, 'IFRPostProgress/FFR/CTFFRIQRAverageValue.npy')]
            self.ParaviewTable.append(tmpParaviewColumn)
            if self.getFFRresult:
                self.ParaviewTable.append(tmpFFRParaviewColumn)
            if self.getIFRresult:
                self.ParaviewTable.append(tmpIFRParaviewColumn)

            # Savevtu table
            tmpSavevtuColumn = [os.path.join(DCMDirPath, 'PostProgress'), True,
                                os.path.join(DCMDirPath, 'CFD/FindFace/3Dmesh.nas'),
                                os.path.join(DCMDirPath, 'PostProgress/PostNpy/ndInterestDict_dict.npy'),
                                os.path.join(DCMDirPath, 'PostProgress/PostNpy'), 'CFDresult', True, False,
                                os.path.join(DCMDirPath, 'PostProgress/PostNpy/ndTimeAvePressureCoo_2dndarr.npy'),
                                os.path.join(DCMDirPath, 'PostProgress/FFR/CTFFRIQRAverageValue.npy'), 'PDPA', '',
                                os.path.join(DCMDirPath, 'PostProgress/Npy/Fluid_lst_ParamsDic.npy'), 'CFDresult',
                                True, False, True]
            tmpFFRSavevtuColumn = [os.path.join(DCMDirPath, 'FFRPostProgress'), True,
                                   os.path.join(DCMDirPath, 'PostProgress/Vtu/CFDresult.vtu'), '', '', '', '', '',
                                   os.path.join(DCMDirPath, 'FFRPostProgress/PostNpy/ndTimeAvePressureCoo_2dndarr.npy'),
                                   os.path.join(DCMDirPath, 'FFRPostProgress/FFR/CTFFRIQRAverageValue.npy'),
                                   'FFR', 'FFRresult', '', '', False, True, False]
            tmpIFRSavevtuColumn = [os.path.join(DCMDirPath, 'IFRPostProgress'), True,
                                   os.path.join(DCMDirPath, 'FFRPostProgress/Vtu/FFRresult.vtu'), '', '', '', '', '',
                                   os.path.join(DCMDirPath, 'IFRPostProgress/PostNpy/ndTimeAvePressureCoo_2dndarr.npy'),
                                   os.path.join(DCMDirPath, 'IFRPostProgress/FFR/CTFFRIQRAverageValue.npy'), 'IFR',
                                   'IFRresult', '', '', False, True, False]
            self.SaveVtuTable.append(tmpSavevtuColumn)
            if self.getFFRresult:
                self.SaveVtuTable.append(tmpFFRSavevtuColumn)
            if self.getIFRresult:
                self.SaveVtuTable.append(tmpIFRSavevtuColumn)

            ## Area volume calculation table
            # save branch table
            branchMFilePath = os.path.join(self.ArVolCalPathList[loopIndex], 'AVCBranM.csv')
            branchM_df = pd.DataFrame(AVCBranMTable, columns=AVCBranM_Head)
            _write_dataframe_atomic(branchM_df, branchMFilePath)

            # save parameter table
            ParamFilePath = os.path.join(self.ArVolCalPathList[loopIndex], 'AVCParam.csv')
            Param_df = pd.DataFrame([[totInflow_AVC, diffThres_AVC, step_AVC, self.ArVolCalPathList[loopIndex],
                                      exo_AVC, mul_AVC]], columns=AVCParam_Head)
            _write_dataframe_atomic(Param_df, ParamFilePath)

            # batch files
            self.AVCTable.append([branchMFilePath, ParamFilePath])

        if not write_shared_tables:
            return

        # generate csv used for batch filter mask processing
        FM_df = pd.DataFrame(self.FilterMaskTable, columns=FM_Head)
        _write_dataframe_atomic(FM_df, self.FilterMaskTablePath)
        # generate csv used for Batch Xsection
        Xsec_df = pd.DataFrame(self.XsectionTable, columns=Xsection_Head)
        _write_dataframe_atomic(Xsec_df, self.XsectionTablePath)
        # generate csv used for Batch Lumen Correction
        XsecFix_df = pd.DataFrame(self.XsectionFixTable, columns=XsecFix_Head)
        _write_dataframe_atomic(XsecFix_df, self.XsectionFixTablePath)
        # generate csv used for batch extension
        XsecFixExt_df = pd.DataFrame(self.XsectionFixExtTable, columns=Extension_Head)
        _write_dataframe_atomic(XsecFixExt_df, self.XsectionFixExtTablePath)
        # generate csv used for batch xsection original volume
        OriVolume_df = pd.DataFrame(self.OriVolumeTable, columns=OriVolume_Head)
        _write_dataframe_atomic(OriVolume_df, self.OriVolumeTablePath)
        # stack segmentation
        SS_df = pd.DataFrame(self.SSTable, columns=SS_Head)
        _write_dataframe_atomic(SS_df, self.SSTablePath)
        # stl generation
        SG_df = pd.DataFrame(self.SGTable, columns=SG_Head)
        _write_dataframe_atomic(SG_df, self.SGTablePath)
        # murray's law
        AVC_df = pd.DataFrame(self.AVCTable, columns=AVCBatch_Head)
        _write_dataframe_atomic(AVC_df, self.AVCTablePath)
        # generate csv used for Batch Lumen Correction
        XsecFix_df = pd.DataFrame(self.OriVolumeFixTable, columns=XsecFixOri_Head)
        _write_dataframe_atomic(XsecFix_df, self.OriVolumeFixTablePath)

        CFD_Mesh_df = pd.DataFrame(self.CFD_MeshTable, columns=CFD_Mesh_Head)
        _write_dataframe_atomic(CFD_Mesh_df, self.CFD_MeshTablePath)

        CFD_STLcut_df = pd.DataFrame(self.CFD_STLcutTable, columns=CFD_STLcut_Head)
        _write_dataframe_atomic(CFD_STLcut_df, self.CFD_STLcutTablePath)

        CFD_STLmesh_df = pd.DataFrame(self.CFD_STLmeshTable, columns=CFD_STLmesh_Head)
        _write_dataframe_atomic(CFD_STLmesh_df, self.CFD_STLmeshTablePath)

        CFD_FindFace_df = pd.DataFrame(self.CFD_FindFaceTable, columns=CFD_FindFace_Head)
        _write_dataframe_atomic(CFD_FindFace_df, self.CFD_FindFaceTablePath)

        CFD_GenerateIn_df = pd.DataFrame(self.CFD_GenerateInTable, columns=CFD_GenerateIn_Head)
        _write_dataframe_atomic(CFD_GenerateIn_df, self.CFD_GenerateInTablePath)

        CFD_PostProgress_df = pd.DataFrame(self.CFD_PostProgressTable, columns=CFD_Postprogress_Head)
        _write_dataframe_atomic(CFD_PostProgress_df, self.CFD_PostProgressTablePath)

        CFD_DataOutput_df = pd.DataFrame(self.DataOutputTable, columns=CFD_DataOutput_Head)
        _write_dataframe_atomic(CFD_DataOutput_df, self.DataOutputTablePath)

        CFD_PostNpy_df = pd.DataFrame(self.PostNpyTable, columns=CFD_PostNpy_Head)
        _write_dataframe_atomic(CFD_PostNpy_df, self.PostNpyTablePath)

        CFD_Map_Result_df = pd.DataFrame(self.Map_ResultTable, columns=CFD_Map_Result_Head)
        _write_dataframe_atomic(CFD_Map_Result_df, self.Map_ResultTablePath)

        CFD_FFRCalculation_df = pd.DataFrame(self.FFRCalculationTable, columns=CFD_FFRCalculation_Head)
        _write_dataframe_atomic(CFD_FFRCalculation_df, self.FFRCalculationTablePath)

        CFD_Paraview_df = pd.DataFrame(self.ParaviewTable, columns=CFD_Paraview_Head)
        _write_dataframe_atomic(CFD_Paraview_df, self.ParaviewTablePath)

        CFD_SaveVtu_df = pd.DataFrame(self.SaveVtuTable, columns=CFD_Savevtu_Head)
        _write_dataframe_atomic(CFD_SaveVtu_df, self.SaveVtuTablePath)

    def BatchFilterMask(self):
        """Run Filter Mask independently per case with validation and resume."""

        case_paths = list(self.DCMDirList)
        plan_warnings = []
        plans = _load_filter_mask_case_plans(
            self.FilterMaskTablePath,
            case_paths,
            warning_messages=plan_warnings,
        )
        for warning in plan_warnings:
            self.UpdateMsgLog(warning)
        record_map = self._load_batch_segmentation_records()
        ready_case_paths = []
        failed_case_paths = []
        outcomes = {}

        for processed_count, case_path_text in enumerate(case_paths, 1):
            case_path = Path(case_path_text)
            case_key = _normalized_path_key(case_path)
            rows = plans.get(case_key, [])
            record = self._ensure_case_record(record_map, case_path)
            fingerprint = ''
            expected_paths = []
            error_message = ''
            skipped = False

            try:
                fingerprint, expected_paths = _filter_mask_case_fingerprint(
                    case_path, rows
                )
                if (
                        str(record.get('complete_filter_mask', '')) == '1'
                        and str(record.get('filter_mask_failed', '')) == '0'
                        and record.get('filter_mask_input_sha256', '') == fingerprint
                        and _filter_mask_outputs_complete(expected_paths)):
                    skipped = True
            except Exception as error:
                error_message = _clean_error_message(error)

            if skipped:
                record['run_filter_mask'] = 1
                record['filter_mask_failed'] = 0
                record['filter_mask_er_msg'] = ''
                record['complete_filter_mask'] = 1
                record['filter_mask_input_sha256'] = fingerprint
                ready_case_paths.append(str(case_path))
                outcomes[str(case_path)] = {
                    'complete': True,
                    'skipped': True,
                    'error_message': '',
                    'input_sha256': fingerprint,
                }
                self.UpdateMsgLog(
                    'Skip completed Filter Mask case: {0}'.format(
                        case_path.name
                    )
                )
            elif error_message:
                record['run_filter_mask'] = 1
                record['filter_mask_failed'] = 1
                record['filter_mask_er_msg'] = error_message
                record['complete_filter_mask'] = 0
                record['filter_mask_input_sha256'] = fingerprint
                failed_case_paths.append(str(case_path))
                outcomes[str(case_path)] = {
                    'complete': False,
                    'skipped': False,
                    'error_message': error_message,
                    'input_sha256': fingerprint,
                }
            else:
                record['run_filter_mask'] = 1
                record['filter_mask_failed'] = 1
                record['filter_mask_er_msg'] = (
                    'Filter Mask started but did not finish.'
                )
                record['complete_filter_mask'] = 0
                record['filter_mask_input_sha256'] = fingerprint
                self._save_batch_segmentation_record_map(record_map)

                backup_paths = []
                temporary_table_path = None
                try:
                    backup_suffix = '.pre_run_{0}_{1}'.format(
                        os.getpid(),
                        datetime.now().strftime('%Y%m%d%H%M%S%f'),
                    )
                    for output_index, expected_path in enumerate(expected_paths):
                        expected_path = Path(expected_path)
                        if not expected_path.exists():
                            continue
                        if not expected_path.is_file():
                            raise ValueError(
                                'Existing Filter Mask output is not a file: {0}'
                                .format(expected_path)
                            )
                        backup_path = expected_path.with_name(
                            expected_path.name
                            + backup_suffix
                            + '_{0}'.format(output_index)
                        )
                        counter = 1
                        while backup_path.exists():
                            backup_path = expected_path.with_name(
                                expected_path.name
                                + backup_suffix
                                + '_{0}_{1}'.format(output_index, counter)
                            )
                            counter += 1
                        os.replace(str(expected_path), str(backup_path))
                        backup_paths.append(backup_path)

                    file_descriptor, temporary_name = tempfile.mkstemp(
                        prefix='.filter_mask_{0}.'.format(case_path.name),
                        suffix='.csv',
                        dir=str(self.TableDir),
                        text=True,
                    )
                    os.close(file_descriptor)
                    temporary_table_path = Path(temporary_name)
                    case_rows = [
                        {
                            column: row[column]
                            for column in FILTER_MASK_REQUIRED_COLUMNS
                        }
                        for row in rows
                    ]
                    _write_dataframe_atomic(
                        pd.DataFrame(
                            case_rows,
                            columns=FILTER_MASK_REQUIRED_COLUMNS,
                        ),
                        temporary_table_path,
                    )
                    self.model.filterMask.BatchProcessing(
                        str(temporary_table_path)
                    )
                    if not _filter_mask_outputs_complete(expected_paths):
                        missing_outputs = [
                            str(path) for path in expected_paths
                            if not _is_nonempty_regular_file(path)
                        ]
                        raise ValueError(
                            'Filter Mask outputs are missing or empty: {0}'.format(
                                ', '.join(missing_outputs)
                            )
                        )
                    for backup_path in backup_paths:
                        backup_path.unlink()
                    record['filter_mask_failed'] = 0
                    record['filter_mask_er_msg'] = ''
                    record['complete_filter_mask'] = 1
                    ready_case_paths.append(str(case_path))
                    outcomes[str(case_path)] = {
                        'complete': True,
                        'skipped': False,
                        'error_message': '',
                        'input_sha256': fingerprint,
                    }
                except Exception as error:
                    error_message = _clean_error_message(error)
                    record['filter_mask_failed'] = 1
                    record['filter_mask_er_msg'] = error_message
                    record['complete_filter_mask'] = 0
                    failed_case_paths.append(str(case_path))
                    outcomes[str(case_path)] = {
                        'complete': False,
                        'skipped': False,
                        'error_message': error_message,
                        'input_sha256': fingerprint,
                    }
                finally:
                    if (temporary_table_path is not None
                            and temporary_table_path.exists()):
                        try:
                            temporary_table_path.unlink()
                        except OSError as cleanup_error:
                            self.UpdateMsgLog(
                                'Could not remove per-case Filter Mask table; '
                                'continuing: {0}: {1}'.format(
                                    temporary_table_path, cleanup_error
                                )
                            )

            self._save_batch_segmentation_record_map(record_map)
            self.UpdateMsgLog(
                'Filter Mask progress: completed {0}, failed {1}, '
                'processed {2}/{3}.'.format(
                    len(ready_case_paths),
                    len(failed_case_paths),
                    processed_count,
                    len(case_paths),
                )
            )

        # Moving happens only after every case in the fixed snapshot was attempted.
        self._move_failed_cases(
            failed_case_paths,
            record_map,
            'filter_mask_er_msg',
            'Filter Mask',
        )
        self.DCMDirList = ready_case_paths
        self.CTADirList = [
            os.path.join(case_path, 'CTA', 'CTA.nii.gz')
            for case_path in ready_case_paths
        ]
        self.StoreListInformation()
        try:
            merge_batch_segmentation_records(
                self.ManifestPath,
                self.InitialSegmentationRoot,
            )
        except Exception as error:
            self.UpdateMsgLog(
                'Manifest merge warning after Filter Mask; batch record is '
                'preserved at {0}: {1}'.format(
                    self.BatchSegmentationRecordPath, error
                )
            )
        return outcomes

    # create Xsection Object and batch process
    def BatchXsection(self):
        self.model.centerlineXsection.BatchConvert(InputCSVPath=self.XsectionTablePath)

    # create Lumen Correction Object and batch process
    def BatchLumenCorrection(self):
        # lumen correction
        cmd1 = self.CopyCommand(
            self.ui.LumenCorrectionPath_Txt_BPP.toPlainText(),
            self.TableDir
        )
        print(cmd1)

        self.model.lumenCorrect.BatchCorrect(CSVPath=self.XsectionFixTablePath)

    # create Ori-Volume Correction Object and batch process
    def BatchOriVolumeCorrection(self):
        # lumen correction
        cmd1 = self.CopyCommand(
            self.ui.OriVolumePath_Txt_BPP.toPlainText(),
            self.TableDir
        )
        os.system(cmd1)

        # OriVCorrect = LumenCorrectMain.LumCorrect()
        # OriVCorrect.BatchCorrect(CSVPath=self.OriVolumeFixTablePath)
        self.model.lumenCorrect.BatchCorrect(CSVPath=self.OriVolumeFixTablePath)

    # create Extension Object and execute batch process
    def BatchExtension(self):
        print('in')
        # Ext = MaskExtensionMain.MaskExtension()
        print(self.XsectionFixExtTablePath)
        # Ext.BatchProcess(path=self.XsectionFixExtTablePath)
        self.model.maskExtension.BatchProcess(path=self.XsectionFixExtTablePath)

    # create OriVolume Object and batch process
    def BatchXsecOriVolume(self):
        # OriVol = XsectionOriVolumeMain.OriVolume()
        # OriVol.BatchProcessing(CSVPath=self.OriVolumeTablePath, EXEPath=self.CPREXEPath)
        self.model.oriVolume.BatchProcessing(CSVPath=self.OriVolumeTablePath, EXEPath=self.CPREXEPath)

    # create SS object and batch process
    def SSSSGridMan(self):
        # copy aorta
        for loopIndex in range(len(self.XsectionFixPathList)):
            cmd1 = self.CopyCommand(
                os.path.join(self.XsectionFixPathList[loopIndex], 'AortaFixVal.nii.gz'),
                self.OriVolumePathList[loopIndex]
            )
            os.system(cmd1)

        # SS = StackSegmentMain.StackSegment()
        # SS.BatchProcessing(CSVPath=self.SSTablePath)
        self.model.stackSegment.BatchProcessing(CSVPath=self.SSTablePath)

    # create Batch Mask2STL object and execute batch process
    def BatchMaskStl(self):
        # MS = MaskSTLMain.MaskSTL()
        # MS.BatchConvert(CSVPath=self.SGTablePath, choice=3)
        self.model.maskSTL.BatchConvert(CSVPath=self.SGTablePath, choice=3)

    def BatchMurraysLaw(self):
        # areaVolCalcs = AreaVolumeCalcsMain.AreaVolCalcs(self.UI)
        # areaVolCalcs.BatchAutoType3(tablePath=self.AVCTablePath)
        self.model.areaVolCalcs.BatchAutoType3(tablePath=self.AVCTablePath)

        # copy Aorta files
        aorOut_AVC = self.ui.AVCPath_Txt_BPP.toPlainText() + '/Aor_in.txt'
        aorIn_AVC = self.ui.AVCPath_Txt_BPP.toPlainText() + '/Aor_out.txt'
        aorOutFFR_AVC = self.ui.AVCPath_Txt_BPP.toPlainText() + '/Aor_outFFR.txt'
        for loopIndex in range(len(self.ArVolCalPathList)):
            # copy to all subdirectories
            subfolderPaths = [f.path for f in os.scandir(self.ArVolCalPathList[loopIndex]) if f.is_dir()]
            for subfolder in subfolderPaths:
                # copy aorta files
                cmd1 = self.CopyCommand(aorOut_AVC, subfolder)
                cmd2 = self.CopyCommand(aorIn_AVC, subfolder)
                cmd3 = self.CopyCommand(aorOutFFR_AVC, subfolder)
                os.system(cmd1)
                os.system(cmd2)
                os.system(cmd3)
                self.UpdateMsgLog(str(cmd1))
                self.UpdateMsgLog(str(cmd2))
                self.UpdateMsgLog(str(cmd3))

        BloodPressureAndHeartRate_Head = ['Blood Pressure And Heart Rate', 'Reference CSV path', 'Output CSV path',
                                          'CalculatePressurePa', 'CalculatePressure', 'CalculateCardiacCycle',
                                          'CalculatePressureCoefficent', 'CalculateHeartRateCoefficent',
                                          'Linear Regression', 'Linear Regression Timestart',
                                          'Systolic Blood Pressure (mmHg)', 'Diastolic Blood Pressure (mmHg)',
                                          'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Mean Blood Pressure',
                                          'Pressure Difference', 'Mean Presure Coefficient',
                                          'Pressure Difference Coefficient', 'Heart Rate', 'Cardiac Cycle',
                                          'Heart Rate Coefficient']

        BloodPressureAndHeartRatePath = self.ui.BloodPressureAndHeartRatePath_Txt_BPP.toPlainText()
        # load DFs
        dfBloodPressureAndHeartRateTable = Pd_Funs.OpenDF(inPath=BloodPressureAndHeartRatePath, header=0, usecols=None)

        inflow = ['Aor_in']
        MeanPresureCoefficient = ''
        BloodPressureCoefficient = ''
        HeartRateCoefficient = 1
        SystolicBloodPressuremmHg = 122.5
        DiastolicBloodPressuremmHg = 62.2
        HeartRate = 67
        for loopIndex in range(len(self.ArVolCalPathList)):
            BloodPressureAndHeartRate = dfBloodPressureAndHeartRateTable.iloc[loopIndex]
            # copy to all subdirectories
            subfolderPaths = [f.path for f in os.scandir(self.ArVolCalPathList[loopIndex]) if f.is_dir()]
            BloodPressureAndHeartRateTable = []
            for subfolder in subfolderPaths:
                flowpaths = [f.path for f in os.scandir(subfolder) if f.is_file()]
                try:
                    if BloodPressureAndHeartRate["Systolic Blood Pressure (mmHg)"]:
                        SystolicBloodPressuremmHg = BloodPressureAndHeartRate["Systolic Blood Pressure (mmHg)"]
                except:
                    pass
                try:
                    if BloodPressureAndHeartRate["Diastolic Blood Pressure (mmHg)"]:
                        DiastolicBloodPressuremmHg = BloodPressureAndHeartRate["Diastolic Blood Pressure (mmHg)"]
                except:
                    pass
                try:
                    if BloodPressureAndHeartRate["Heart Rate"]:
                        HeartRate = BloodPressureAndHeartRate["Heart Rate"]
                except:
                    pass
                for flowpath in flowpaths:
                    (flowpathonly, flowpathname) = os.path.split(flowpath)
                    (flowpathnameonly, flowpathnamesuffix) = os.path.splitext(flowpathname)
                    if flowpathnameonly not in inflow:
                        MeanPresureCoefficient = 1
                        BloodPressureCoefficient = 1
                    tmpBloodPressureAndHeartRateTable = [True, flowpath, flowpath, True, True, True, True, True, True,
                                                         2, SystolicBloodPressuremmHg, DiastolicBloodPressuremmHg, '',
                                                         '', '', '', MeanPresureCoefficient, BloodPressureCoefficient,
                                                         HeartRate, '', HeartRateCoefficient]
                    BloodPressureAndHeartRateTable.append(tmpBloodPressureAndHeartRateTable)

            # save branch table
            BloodPressureAndHeartRateFilePath = os.path.join(self.ArVolCalPathList[loopIndex],
                                                             'BloodPressureAndHeartRate.csv')
            BloodPressureAndHeartRate_df = pd.DataFrame(BloodPressureAndHeartRateTable,
                                                        columns=BloodPressureAndHeartRate_Head)
            BloodPressureAndHeartRate_df.to_csv(BloodPressureAndHeartRateFilePath, index=None)

    def BatchBloodPressureAndHeartRate(self):
        for loopIndex in range(len(self.ArVolCalPathList)):
            BloodPressureAndHeartRateFilePath = os.path.join(self.ArVolCalPathList[loopIndex],
                                                             'BloodPressureAndHeartRate.csv')
            self.model.BloodPressureAndHeartRate.batchBloodPressureAndHeartRate(
                tablePath=BloodPressureAndHeartRateFilePath)

    # create MeshCut Object and batch process
    def BatchMeshCut(self):
        # MeshCut = CuttingMeshTest.Interaction(self.UI)
        # MeshCut.batchrun(CSVPath=self.CFD_STLcutTablePath)
        self.model.MeshCut.batchrun(CSVPath=self.CFD_STLcutTablePath)

    # create Mesh Object and batch process
    def BatchMesh(self):
        # Mesh = MeshGeneratingMain.MeshGenerating(self.UI)
        # Mesh.batchrun(CSVPath=self.CFD_STLmeshTablePath)
        self.model.Meshgeneration.batchrun(CSVPath=self.CFD_STLmeshTablePath)

    # create FindFace Object and batch process
    def BatchFindFace(self):
        # FindFace = SplitFaceElemMain.SplitFaceElem(self.UI)
        # FindFace.batchrun(CSVPath=self.CFD_FindFaceTablePath)
        self.model.SplitFaceElem.batchrun(CSVPath=self.CFD_FindFaceTablePath)

    # create GenerateInput Object and batch process
    def BatchGenerateInput(self):
        # GenerateInput = Generate_In_File.GenerateInputFile(self.UI)
        # GenerateInput.batchrun(CSVPath=self.CFD_GenerateInTablePath)
        self.model.Finfilegneration.batchrun(CSVPath=self.CFD_GenerateInTablePath)

    # create DataOutput Object and batch process
    def BatchDataOutput(self):
        # DataOutput = ADINA_DataOutputMain.ADINA_DataOutput(self.UI)
        # DataOutput.batchrun(CSVPath=self.DataOutputTablePath)
        self.model.adina_DataOutput.batchrun(CSVPath=self.DataOutputTablePath)

    # create CFD_PostProcessing Object and batch process
    def BatchCFD_PostProcessing(self):
        # CFD_PostProcessing = CalcCFDResultMain.CFDParameters(self.UI)
        # CFD_PostProcessing.batchrun(CSVPath=self.PostNpyTablePath)
        self.model.CFDParameters.batchrun(CSVPath=self.PostNpyTablePath)

    # create MapResult Object and batch process
    def BatchMapResult(self):
        # MapResult = MapResultsMain.MapResults(self.UI)
        # MapResult.Batch(DFPath=self.Map_ResultTablePath)
        self.model.mapResults.Batch(DFPath=self.Map_ResultTablePath)

    # create FFRcalculation Object and batch process
    def BatchFFRcalculation(self):
        # FFRcalculation = FFRCalcsMain.FFRCalcs(self.UI)
        # FFRcalculation.batchrun(CSVPath=self.FFRCalculationTablePath)
        self.model.FFR.batchrun(CSVPath=self.FFRCalculationTablePath)

    # create Paraview-processing Object and batch process
    def BatchParaview_processing(self):
        # Paraview_processing = PostProcessingMain.PostProcessing(self.UI)
        # Paraview_processing.batchrun(CSVPath=self.ParaviewTablePath)
        self.model.ParaviewPostProcessing.batchrun(CSVPath=self.ParaviewTablePath)

    # create SaveVTU Object and batch process
    def BatchSaveVTU(self):
        # SaveVTU = SaveVTUMain.SaveVTUorPVD(self.UI)
        # SaveVTU.batchrun(CSVPath=self.SaveVtuTablePath)
        self.model.SaveVTU.batchrun(CSVPath=self.SaveVtuTablePath)

    '''
    ##############################################################################
    # Main Function
    ##############################################################################
    '''

    def mainFunction(self):
        self.initProcess()
        if self.ui:
            self.WorkingDir = _widget_text(self.ui.LoadFolderPath_Txt_BPP)
            self.CGEXEPath = _widget_text(self.ui.CGExePath_Txt_BPP)
            self.CPREXEPath = _widget_text(self.ui.CPRExePath_Txt_BPP)
            self.STLEXEPath = _widget_text(self.ui.STLGExePath_Txt_BPP)
        else:
            self.UpdateMsgLog('No UI')

        if not self.WorkingDir:
            raise ValueError('Working directory is required.')
        working_path = Path(self.WorkingDir).expanduser().resolve()
        if not working_path.is_dir():
            raise NotADirectoryError(
                'Working directory does not exist: {0}'.format(working_path)
            )
        self.WorkingDir = str(working_path)
        self.BatchName = working_path.name
        self.InitialSegmentationRoot = str(working_path.parent)

        manifest_widget = getattr(self.ui, 'ManifestPath_Txt_BPP', None)
        manifest_path = _widget_text(manifest_widget)
        if not manifest_path:
            manifest_path = str(working_path.parent / DEFAULT_MANIFEST_FILE_NAME)
        self.ManifestPath = str(Path(manifest_path).expanduser())

        failed_root_widget = getattr(self.ui, 'FailedRootPath_Txt_BPP', None)
        failed_root_path = _widget_text(failed_root_widget)
        if not failed_root_path:
            if working_path.parent.name == '0_Initial_Segmentation':
                failed_root_path = str(working_path.parent.parent)
            else:
                failed_root_path = str(working_path.parent)
        self.FailedRootPath = str(Path(failed_root_path).expanduser())

        timeout_minutes = _widget_integer(
            getattr(self.ui, 'SegmentationTimeout_SB_BPP', None),
            DEFAULT_SEGMENTATION_TIMEOUT_MINUTES,
        )
        if timeout_minutes <= 0:
            raise ValueError('Segmentation timeout must be greater than zero.')
        self.SegmentationTimeoutSeconds = timeout_minutes * 60

        # create overall table directory
        self.TableDir = os.path.join(self.WorkingDir, 'Table')
        os.makedirs(self.TableDir, exist_ok=True)

        # cmd english
        os.system('chcp 437')

        # check boxes
        self.CheckCheckBoxStatus()

        # get all case folders & create folders
        self.UpdateMsgLog('Scan DCM Start')
        self.ScanDCMDir()
        print("Find DCM folders")
        print(str(self.DCMDirList))
        self.UpdateMsgLog('Scan DCM Done')

        # create folders
        if self.CreateFolderFlag:
            self.UpdateMsgLog('Create Folders Start')
            self.CreateFoldersDCM()
            self.UpdateMsgLog('Create Folders Done')

        # create all paths
        self.StoreListInformation()

        # DCM to Nifti
        if self.D2NFlag:
            self.UpdateMsgLog('DCM to Nifti Convert Start')
            # create new DICOM to nifti table
            self.GenerateCTATable()
            # DCM to nifti
            self.BatchDCM2Nifti()
            self.UpdateMsgLog('DCM to Nifti Convert Done')

        # copy CTA need to be done
        if self.CopyCTAFlag:
            self.UpdateMsgLog('Copy CTA Start')
            self.CopyCTA()
            self.UpdateMsgLog('Copy CTA Done')

        # centerline generation
        if self.CGFlag:
            self.UpdateMsgLog('Centerline Generation start')
            self.BatchCenterlineGen()
            self.UpdateMsgLog('Centerline Generation done')

        centerline_csv_lifecycle = bool(
            self.CenterlineEFlag or self.UpdateJasonFlag
        )
        if centerline_csv_lifecycle:
            self.BeginCenterlineCsvRun()

        # copy Jason for same lists
        if self.CopyJSONFlag:
            self.UpdateMsgLog('Copy JASON Start')
            self.CopyCenterlineJson()
            self.UpdateMsgLog('Copy JASON done')

        # center line extraction
        if self.CenterlineEFlag:
            self.UpdateMsgLog('Centerline Extraction start')
            self.BatchExtractCenterline()
            self.UpdateMsgLog('Centerline Extraction done')

        # run update jason
        if self.UpdateJasonFlag:
            self.UpdateMsgLog('Jason Update start')
            # create table for update Jason centerline
            self.GenerateJsonUpdateTable()
            # batch update
            self.BatchJsonUpdate()
            self.UpdateMsgLog('Jason Update done')

            # auto centerline extraction
            self.UpdateMsgLog('Centerline Extraction start')
            self.BatchExtractCenterline()
            self.UpdateMsgLog('Centerline Extraction done')

        if centerline_csv_lifecycle:
            self.UpdateMsgLog('Centerline CSV safeguard start')
            self.FinalizeCenterlineCsvRun()
            self.UpdateMsgLog('Centerline CSV safeguard done')

        # create branch to be used json file
        if self.BranchDictFlag:
            self.UpdateMsgLog('Write Branch Dictionary start')
            self.CheckWriteDict()
            self.UpdateMsgLog('Write Branch Dictionary done')

        # generate table
        if self.GenerateTFlag:
            self.UpdateMsgLog('Generate Table start')
            self.RetainTableReadyCases()
            print(self.BranchPathList)
            self.GenerateTable()
            self.UpdateMsgLog('Generate Table done')

        # Filter mask
        if self.FMFlag:
            self.UpdateMsgLog('Filter Mask start')
            self.BatchFilterMask()
            self.UpdateMsgLog('Filter Mask done')
            # copy aorta
            # self.CopyAorta2OriVolumeFolder()

        if self.XsecFlag:
            self.UpdateMsgLog('Xsection start')
            self.BatchXsection()
            self.UpdateMsgLog('Xsection Done')

        if self.LCFlag:
            self.UpdateMsgLog('LC TEST')
            BatchCoronary.WriteAortaFixCsv(dstPath=self.AortaFixTablePath)
            self.UpdateMsgLog('Write Aorta Fix CSV done')

            self.BatchLumenCorrection()
            self.UpdateMsgLog('Lumen Correction Done')

        if self.ExtFlag:
            self.UpdateMsgLog('Extension Start')
            self.BatchExtension()
            self.UpdateMsgLog('Extension Done')

        if self.XsecOriFlag:
            self.BatchXsecOriVolume()
            self.UpdateMsgLog('Xsection OriVolume Done')

        if self.OriVolumeFixFlag:
            self.UpdateMsgLog('OriVolume Correction Start')
            self.BatchOriVolumeCorrection()
            self.UpdateMsgLog('OriVolume Correction Done')

        if self.SSFlag:
            self.SSSSGridMan()
            self.UpdateMsgLog('Stack Segmentation Done')

        if self.M2SFlag:
            self.BatchMaskStl()
            self.UpdateMsgLog('Mask to STL done')

        if self.ArVolCalFlag:
            self.BatchMurraysLaw()
            self.UpdateMsgLog("Murray's Law done")

        if self.ui.BloodPressureAndHeartRate_BPP.isChecked():
            self.BatchBloodPressureAndHeartRate()
            self.UpdateMsgLog("Blood Pressure And Heart Rate Input")

        if self.MeshCut:
            self.BatchMeshCut()
            self.UpdateMsgLog("MeshCut done")

        if self.Mesh:
            self.BatchMesh()
            self.UpdateMsgLog("Mesh done")

        if self.FindFace:
            cmd1 = self.CopyCommand(self.ui.setPath_Txt_BPP.toPlainText(), self.TableDir)
            os.system(cmd1)
            self.BatchFindFace()
            self.UpdateMsgLog("FindFace done")

        if self.GenerateInput:
            self.BatchGenerateInput()
            self.UpdateMsgLog("GenerateInput done")

        if self.DataOutput:
            self.BatchDataOutput()
            self.UpdateMsgLog("DataOutput done")

        if self.CFD_PostProcessing:
            self.BatchCFD_PostProcessing()
            self.UpdateMsgLog("CFD_PostProcessing done")

        if self.MapResult:
            self.BatchMapResult()
            self.UpdateMsgLog("MapResult done")

        if self.FFRcalculation:
            self.BatchFFRcalculation()
            self.UpdateMsgLog("FFRcalculation done")

        if self.Paraview_processing:
            self.BatchParaview_processing()
            self.UpdateMsgLog("Paraview_processing done")

        if self.SaveVTU:
            self.BatchSaveVTU()
            self.UpdateMsgLog("SaveVTU done")

        # init process
        self.initProcess()

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
    a = BatchCoronary()
    a.mainFunction()
