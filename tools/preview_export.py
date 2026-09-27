"""Current preview allowlist; historical private evidence stays in place."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from tools.s3_public_export import allowlist
from crystargetbench.public_export import export_public
from crystargetbench.public_audit import audit_public_tree


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--archive',required=True)
    a=p.parse_args();root=Path.cwd();target=root/a.output;archive=root/a.archive
    if archive.exists():raise ValueError('refuse to overwrite an existing artifact')
    selected=[name for name in allowlist(root) if not name.startswith('docs/s3_1/') and not name.startswith('docs/CTB_S3_1_')]
    selected += ['docs/CTB_S3_1_MIGRATION.md','docs/PREVIEW_SDK_WALKTHROUGH.md',
                 'tools/preview_demo.py','tools/preview_journeys.py','tools/preview_export.py','tools/preview_ci.py']
    for name in ['CTB_PUBLIC_PREVIEW_REPORT.md','USER_JOURNEY_RESULTS.json','CAPABILITY_MATRIX.json','RELEASE_READINESS.json','OWNER_METADATA_TODO.md','TEST_RESULTS.json','environment.lock.json']:
        path='docs/preview/'+name
        if (root/path).is_file():selected.append(path)
    # Curated current user outputs; original command logs and historical runs
    # are deliberately not swept into the distribution.
    artifacts=root/'docs/preview/artifacts'
    if artifacts.exists():
        for path in artifacts.rglob('*'):
            if path.is_file() and path.name!='.exchange.lock' and not any(x in {'cache','incomplete','conflicting'} for x in path.relative_to(artifacts).parts):
                selected.append(str(path.relative_to(root)))
    audit=export_public(root,target,selected)
    scan=audit_public_tree(target)
    if scan['findings']:raise ValueError('independent public scan failed')
    archive.parent.mkdir(parents=True,exist_ok=True);entries=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(target.rglob('*')):
            if path.is_file():
                relative=str(path.relative_to(target));data=path.read_bytes();z.writestr(relative,data)
                entries.append({'path':relative,'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    record={'path':str(archive.relative_to(root)),'size_bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':entries}
    archive.with_suffix('.manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    evidence=root/'docs/preview';evidence.mkdir(exist_ok=True)
    (evidence/'PUBLIC_EXPORT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (evidence/'INDEPENDENT_PUBLIC_POSTCHECK.json').write_text(json.dumps(scan,indent=2)+'\n')
    print(json.dumps({'files':len(entries),'archive':record['path'],'sha256':record['sha256'],'redactions':len(audit['redactions']),'findings':len(scan['findings'])}))

if __name__=='__main__':main()
