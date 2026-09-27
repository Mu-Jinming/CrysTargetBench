"""Curated public source tree; no private history or physical payloads.

Run from the project root. Re-running requires a new/empty destination so a
previous reviewed artifact is never silently overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from crystargetbench.public_export import export_public

TOP=['OWNER_METADATA_TODO.md','CTB_REPOSITORY_METADATA.json','MANIFEST.in','pyproject.toml','README.md','LICENSE','CITATION.cff','CONTRIBUTING.md','SECURITY.md','THIRD_PARTY.md','CHANGELOG.md','.gitignore','.github/workflows/ci.yml']
DOCS=['tasks_and_metrics.md','backends.md','protocol_comparability.md','DFT_ADAPTER_API.md',
      'ABACUS_SUPPORT_MATRIX.json','ABACUS_PARSER_EVIDENCE.md','release_checklist.md','PUBLIC_REPOSITORY_READINESS.md',
      'CTB_S3_1_REPORT.md','CTB_S3_1_TEST_RESULTS.json','CTB_S3_1_MIGRATION.md',
      'CTB_S3_1_REPRO_BEFORE.json','CTB_S3_1_REPRO_AFTER.json']
TOOLS=['generate_s2_numerical.py','cleanroom_verify.py','s3_examples.py','s3_1_evidence.py','s3_release_check.py','s3_public_export.py']

def allowlist(root):
    selected=[p for p in TOP if (root/p).is_file()]
    selected+=['docs/'+p for p in DOCS if (root/'docs'/p).is_file()]
    selected+=['tools/'+p for p in TOOLS if (root/'tools'/p).is_file()]
    for tree in ['src/crystargetbench','tests','examples']:
        for p in (root/tree).rglob('*'):
            if not p.is_file() or p.suffix not in {'.py','.json','.cif','.md','.toml'}:continue
            if any(x in {'__pycache__','build','dist'} or x.endswith('.egg-info') for x in p.parts):continue
            selected.append(str(p.relative_to(root)))
    selected+=['CTB_S2_Addendum/examples/MLFF_LIVE_PERMIT.disabled.json']
    selected += [p for p in ['docs/s3_1/environment.lock.json','docs/s3_1/BACKEND_STATUS.json','docs/s3_1/DISTRIBUTION_AUDIT.json','docs/s3_1/public-roundtrip.json','docs/s3_1/public-tree-validation.json'] if (root/p).is_file()]
    # Only project-authored synthetic fixtures and an explicit empty reference
    # notice. Historical runs, permits, sessions and raw audit logs are excluded.
    example=root/'docs/s3_1/examples'
    if example.exists():
        for p in example.rglob('*'):
            if p.is_file() and p.name!='.exchange.lock' and p.suffix!='.lock':
                if any(x in {'cache','returned'} for x in p.relative_to(example).parts):continue
                # JSON records in analytic returned caches are redundant; keep
                # the documented numerical artifact, not large raw cache trees.
                if 'analytic-phonopy' in p.parts and p.name!='numerical.json':continue
                selected.append(str(p.relative_to(root)))
    # A portable exchange journal references received artifacts; include those
    # explicitly, never generic runs or unrelated caches.
    if example.exists():
        for p in example.rglob('received/*/*'):
            if p.is_file() and p.suffix in {'.json','.log'}:selected.append(str(p.relative_to(root)))
    return sorted(set(selected))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='public-export-0.3.1');p.add_argument('--archive',default='dist/CTB-0.3.1-public-source.zip')
    a=p.parse_args();root=Path.cwd();target=root/a.output
    audit=export_public(root,target,allowlist(root))
    from crystargetbench.public_audit import audit_public_tree
    independent=audit_public_tree(target)
    if independent['findings']:raise ValueError('independent post-export findings: '+str(independent['findings']))
    archive=root/a.archive;archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists():raise ValueError('refuse to overwrite reviewed source archive')
    entries=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(target.rglob('*')):
            if not f.is_file():continue
            relative=str(f.relative_to(target));z.write(f,relative)
            entries.append({'path':relative,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'size_bytes':f.stat().st_size})
    record={'archive':str(archive.relative_to(root)),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':entries}
    archive.with_suffix('.manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    if (root/'docs/s3_1').is_dir():(root/'docs/CTB_PUBLIC_EXPORT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'files':len(entries),'archive':record['archive'],'sha256':record['sha256'],'redacted_files':len(audit['redactions'])}))

if __name__=='__main__':main()
