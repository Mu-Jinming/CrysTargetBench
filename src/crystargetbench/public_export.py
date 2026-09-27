"""Allowlisted public repository export with recursive data sanitization.

Original files and private audit trails are never rewritten. This is not a
license checker; only explicitly listed, project-owned files may be exported.
"""
from pathlib import Path
import hashlib
import json
import re
from .reporting import write_json

_DROP=object()
_PRIVATE_PATH=re.compile(r'(?:/home/[A-Za-z0-9_.-]+/[^\s\"\'<>]+|/Users/[A-Za-z0-9_.-]+/[^\s\"\'<>]+|[A-Za-z]:\\\\Users\\\\[^\s\"\'<>]+)')
_SENSITIVE={'session','sessions','session_id','full_session','live_session','private_notes','access_token',
    'api_key','password','secret','owner_identity','hostname','account_name','authorization_token'}

# Reviewed, schema-complete disabled template, not a runtime permission. The
# semantic hash pins every field, null binding and false flag. Any extension,
# conflicting flag, filled binding or changed text requires a new review.
_DISABLED_TEMPLATE_SHA256 = '8731b105a6901e9e12f38a45baf3da64b4c7d6e750e27a0e3c5a0e679caf285e'

def approved_disabled_template(value):
    if value.get('schema_version')!='ctb.s2.live_permit_template.1':return False
    wire=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    return (value.get('schema_version')=='ctb.s2.live_permit_template.1'
            and value.get('authorized') is False
            and hashlib.sha256(wire).hexdigest()==_DISABLED_TEMPLATE_SHA256)


def redact_embedded(text):
    """Also handles JSON embedded inside JSON/YAML string values."""
    decoder=json.JSONDecoder();position=0;removed=[]
    while position<len(text):
        start=min((i for i in (text.find('{',position),text.find('[',position)) if i>=0),default=-1)
        if start<0:break
        try:data,length=decoder.raw_decode(text[start:])
        except ValueError:position=start+1;continue
        marks=[];clean=redact(data,removed=marks)
        if marks:
            replacement='[REDACTED_PRIVATE_RECORD]' if clean is _DROP else json.dumps(clean,ensure_ascii=False)
            text=text[:start]+replacement+text[start+length:];removed+=marks;position=start+len(replacement)
        else:position=start+length
    return text,removed

def redact(value,path='$',removed=None):
    removed=removed if removed is not None else []
    if isinstance(value,dict):
        schema=str(value.get('schema_version',value.get('schema',''))).lower()
        fields={str(k).lower().replace('-','_') for k in value}
        if approved_disabled_template(value):return value
        if (any(token in schema for token in ('session','permit','authorization'))
                or 'authorized' in fields
                or fields & {'authorization_id','approval_reference','owner_approval_reference','allowed_operations'}):
            removed.append(path);return _DROP
        result={}
        for key,item in value.items():
            normalized=str(key).lower().replace('-','_')
            # This exact property definition describes a nullable path, not a
            # permission instance. Preserve core JSON Schema bytes on export.
            if (path.endswith('.properties') and normalized=='live_permit'
                    and item=={'type':['string','null'],'minLength':1}):
                result[key]=item;continue
            if normalized in _SENSITIVE:
                removed.append(path+'.'+str(key));continue
            if normalized in {'permit','live_permit','authorization'} and isinstance(item,dict) and not approved_disabled_template(item):
                removed.append(path+'.'+str(key));continue
            child=redact(item,path+'.'+str(key),removed)
            if child is not _DROP:result[key]=child
        return result
    if isinstance(value,list):
        children=[redact(item,f'{path}[{i}]',removed) for i,item in enumerate(value)]
        return [child for child in children if child is not _DROP]
    if isinstance(value,str):
        value,marks=redact_embedded(value)
        if marks:removed.append(path+':embedded_json')
        if _PRIVATE_PATH.search(value):
            removed.append(path);value=_PRIVATE_PATH.sub('[REDACTED_PRIVATE_PATH]',value)
    return value


def export_public(source,output,allowlist):
    source=Path(source).resolve();output=Path(output).resolve()
    if output==source or not output.is_relative_to(source):raise ValueError('public export must be a distinct workspace subdirectory')
    if output.exists() and any(output.iterdir()):raise ValueError('use an empty export destination')
    output.mkdir(parents=True,exist_ok=True);entries=[];removals=[];excluded=[]
    for relative in sorted(set(allowlist)):
        parts=Path(relative).parts
        if not parts or Path(relative).is_absolute() or '..' in parts:raise ValueError('unsafe export path')
        if any(p in {'.git','.venvs','.ctb','__pycache__','build','dist','runs','wheelhouse'} or p.endswith('.egg-info') for p in parts):
            raise ValueError('forbidden export tree')
        target=source/relative
        if any((source/Path(*parts[:i])).is_symlink() for i in range(1,len(parts)+1)):raise ValueError('export symlink')
        if not target.is_file():raise ValueError('allowlist entry missing')
        if target.suffix.lower() in {'.pth','.pt','.ckpt','.upf','.orb','.exe','.so','.zip','.whl'}:raise ValueError('engine/asset/archive payload forbidden')
        text=target.read_text();changed=[]
        if target.suffix=='.json':
            clean=redact(json.loads(text),removed=changed)
            if clean is _DROP:excluded.append(relative);continue
            if changed:text=json.dumps(clean,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
        elif target.suffix in {'.yaml','.yml'}:
            import yaml
            data=yaml.safe_load(text)
            clean=redact(data,removed=changed)
            if clean is _DROP:excluded.append(relative);continue
            if changed:text=yaml.safe_dump(clean,sort_keys=False)
            if _PRIVATE_PATH.search(text):raise ValueError('private path in YAML after sanitization')
        elif target.suffix not in {'.py','.toml'}:
            # Text may contain fenced/nested JSON, independent of its filename.
            text,marks=redact_embedded(text);changed+=marks
            if _PRIVATE_PATH.search(text):
                text=_PRIVATE_PATH.sub('[REDACTED_PRIVATE_PATH]',text);changed.append('private_path_in_text')
        elif _PRIVATE_PATH.search(text):
            if target.suffix in {'.py','.toml','.yml','.yaml'}:raise ValueError('private absolute path in public source: '+relative)
            text=_PRIVATE_PATH.sub('[REDACTED_PRIVATE_PATH]',text);changed=['private_path_in_text']
        destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(text)
        entries.append({'path':relative,'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'size_bytes':destination.stat().st_size})
        if changed:removals.append({'path':relative,'fields':changed})
    from .public_audit import audit_public_tree
    postcheck=audit_public_tree(output)
    if postcheck['findings']:raise ValueError('independent public postcheck rejected export: '+str(postcheck['findings']))
    audit={'schema_version':'ctb.public_export_audit.v2','allowlist_only':True,'files':entries,
        'independent_postcheck':postcheck,
        'redactions':removals,'excluded_sensitive_records':excluded,'originals_modified':False,
        'engines_or_assets_included':False,'remote_created':False,'published':False}
    write_json(output/'PUBLIC_EXPORT_AUDIT.json',audit)
    return audit
