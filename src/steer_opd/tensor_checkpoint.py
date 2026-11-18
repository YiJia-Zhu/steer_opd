"""Columnar candidate checkpoints with unchanged legacy engine-call recovery.

Kept separate from experiment_runtime: active runs may freeze that module's SHA.
Existing stage identities select their original reader; new candidates default
to Parquet/Zstd and commit all shards together after the complete engine call.
"""
import os
from pathlib import Path
import uuid

from steer_opd.experiment_runtime import StageCheckpoint as LegacyStageCheckpoint
from steer_opd.resumable_core import atomic_json, read_json
from steer_opd.tensor_archive import FORMATS, SCHEMA, read_archive, write_archive

DEFAULT_CANDIDATE_FORMAT = 'parquet_zstd'
SUPPORTED_FORMATS = ('pt', 'pt_gzip', *FORMATS)
ALIASES = {'pt': 'none', 'pt_gzip': 'gzip'}


class StageCheckpoint(LegacyStageCheckpoint):
    def __init__(self, output, binding, *, tensor_format=None, tensor_compression=None, shard_size=None):
        if tensor_format is not None and tensor_compression is not None:
            raise ValueError('Choose tensor_format or legacy tensor_compression, not both')
        identity = Path(output)/'stage_input.json'
        saved = read_json(identity) if identity.exists() else None
        storage = tensor_format if tensor_format is not None else tensor_compression
        if storage is None:
            storage = saved.get('tensor_format', saved.get('tensor_compression', 'none')) if saved else 'none'
        storage = ALIASES.get(storage, storage)
        self.tensor_format = storage
        if storage in ('none', 'gzip'):
            super().__init__(output, binding, tensor_compression=storage)
            return
        if storage not in FORMATS:
            raise ValueError('Unknown lossless tensor format')
        self.output, self.binding = Path(output), binding
        self.path = self.output/'complete.json'
        self.tensor_compression = storage
        self.shard_size = shard_size if shard_size is not None else (saved or {}).get('tensor_shard_size', 64)
        if type(self.shard_size) is not int or self.shard_size < 1:
            raise ValueError('Tensor shard size must be a positive integer')
        expected = {'schema': self.schema, 'binding': binding, 'tensor_format': storage,
                    'tensor_archive_schema': SCHEMA, 'tensor_shard_size': self.shard_size}
        if saved is not None and saved != expected:
            raise ValueError('Stage input identity changed')
        if saved is None:
            atomic_json(identity, expected)

    def save_tensors(self, records, extra_files=()):
        if self.tensor_format in ('none', 'gzip'):
            return super().save_tensors(records, extra_files)
        if self.path.exists():
            raise ValueError('A committed stage cannot be overwritten')
        keys = list(records)
        if any(not isinstance(k, str) or Path(k).name != k or k in ('.', '..') for k in keys):
            raise ValueError('Unsafe response key')
        files, shards = list(extra_files), []
        extension = '.parquet' if self.tensor_format == 'parquet_zstd' else '.npz'
        for number, start in enumerate(range(0, len(keys), self.shard_size)):
            selected = keys[start:start+self.shard_size]
            target = self.output/f'records_{number:05d}{extension}'
            temporary = target.with_name(target.name+'.tmp.'+uuid.uuid4().hex)
            try:
                write_archive(temporary, {k: records[k] for k in selected}, self.tensor_format)
                with temporary.open('rb') as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            files.append(target)
            shards.append({'file': target.name, 'record_keys': selected})
        self.commit(files, record_keys=keys, tensor_format=self.tensor_format,
                    tensor_archive_schema=SCHEMA, tensor_shard_size=self.shard_size, record_shards=shards)

    def _load_records(self, keys, fields=None, *, require_all=True):
        keys = list(keys)
        value = self.load()
        if value is None:
            return None
        if len(keys) != len(set(keys)) or not set(keys).issubset(value['record_keys']):
            raise ValueError('Committed stage response keys changed')
        if require_all and set(value['record_keys']) != set(keys):
            raise ValueError('Committed stage response keys changed')
        if self.tensor_format in ('none', 'gzip'):
            result = super().load_tensors(value['record_keys'])
            if fields is not None:
                if any(field not in result[key] for key in keys for field in fields):
                    raise ValueError('Requested tensor fields are absent')
                return {key: {field: result[key][field] for field in fields} for key in keys}
            return {key: result[key] for key in keys}
        if (value.get('tensor_format') != self.tensor_format or value.get('tensor_archive_schema') != SCHEMA
                or value.get('tensor_shard_size') != self.shard_size):
            raise ValueError('Committed tensor format identity changed')
        seen, result = set(), {}
        for shard in value['record_shards']:
            if shard['file'] not in value['files'] or seen.intersection(shard['record_keys']):
                raise ValueError('Uncommitted or duplicate tensor shard')
            seen.update(shard['record_keys'])
            selected = [key for key in keys if key in shard['record_keys']]
            if selected:
                result.update(read_archive(self.output/shard['file'], self.tensor_format,
                                           keys=selected, fields=fields))
        if seen != set(value['record_keys']) or set(result) != set(keys):
            raise ValueError('Committed tensor shard keys changed')
        return {key: result[key] for key in keys}

    def load_tensors(self, keys):
        if self.tensor_format in ('none', 'gzip'):
            return super().load_tensors(keys)
        return self._load_records(keys)

    def load_tensor_fields(self, keys, fields):
        """Select records and top-level fields, retaining complete file-SHA checks."""
        return self._load_records(keys, list(fields), require_all=False)


def candidate_checkpoint(output, binding, job):
    """Use the saved format on resume; default only new candidate stores to Parquet."""
    if 'candidate_tensor_format' in job and 'candidate_tensor_compression' in job:
        raise ValueError('Choose candidate_tensor_format or legacy candidate_tensor_compression')
    storage = job.get('candidate_tensor_format', job.get('candidate_tensor_compression'))
    if storage is None and not (Path(output)/'stage_input.json').exists():
        storage = DEFAULT_CANDIDATE_FORMAT
    return StageCheckpoint(output, binding, tensor_format=storage,
                           shard_size=job.get('candidate_tensor_shard_size'))
