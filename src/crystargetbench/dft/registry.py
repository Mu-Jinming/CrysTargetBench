"""Metadata-only discovery. Only an explicitly selected installed entry is loaded."""
from functools import lru_cache
from importlib import metadata
from pathlib import PurePosixPath

from .validation import read_record, validate_manifest

GROUP='crystargetbench.dft_adapters'

@lru_cache(maxsize=1)
def _discover():
    found={}
    for dist in metadata.distributions():
        for ep in dist.entry_points:
            if ep.group!=GROUP: continue
            if ep.name in found: raise ValueError(f'duplicate DFT provider: {ep.name}')
            if ep.name in {'symmetry','mattersim','vasp_atomate2','qe','alignn2','synthetic'}:raise ValueError('reserved builtin provider identifier')
            module=ep.value.split(':')[0]
            if not module or any(not part.isidentifier() for part in module.split('.')):
                raise ValueError('invalid installed entry point')
            relative=PurePosixPath(*module.split('.')[:-1],'ctb_dft_adapter.json') if '.' in module else PurePosixPath(module,'ctb_dft_adapter.json')
            if str(relative) not in {str(p) for p in dist.files or []}:
                raise ValueError(f'{ep.name}: missing static adapter manifest {relative}')
            manifest=validate_manifest(read_record(dist.locate_file(relative)))
            canonical=lambda n:n.lower().replace('_','-').replace('.','-')
            if manifest['provider_id']!=ep.name or canonical(manifest['distribution'])!=canonical(dist.metadata['Name']) or manifest['adapter_version']!=dist.version:
                raise ValueError('entry point/distribution/manifest identity mismatch')
            found[ep.name]=(manifest,ep)
    return found

def discover():
    from copy import deepcopy
    return {name:deepcopy(value[0]) for name,value in sorted(_discover().items())}

def load_adapter(provider):
    selected=_discover().get(provider)
    if selected is None: raise ValueError(f'No installed adapter selected: {provider}')
    manifest,ep=selected
    adapter=ep.load()()
    if validate_manifest(adapter.describe())!=manifest: raise ValueError('runtime describe differs from installed static manifest')
    return adapter,discover()[provider]

def refresh():
    """Explicit metadata cache invalidation after installing a plugin."""
    _discover.cache_clear()
