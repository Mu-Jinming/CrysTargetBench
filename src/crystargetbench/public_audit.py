"""Independent read-only post-export inspection, not another redactor pass.

Inspects serialized data and embedded JSON. Python/TOML source is inventory
reviewed, not interpreted as private runtime data (tests contain attack cases).
This is a bounded data/permission check, not a comprehensive secret scanner.
"""
from pathlib import Path
import hashlib
import json
import re


def audit_public_tree(root):
    findings=[];scanned=[];source_inventory=[];templates=[]
    def inspect(value,location):
        if isinstance(value,dict):
            fields={str(k).casefold().replace('-','_'):v for k,v in value.items()}
            schema=str(fields.get('schema_version',fields.get('schema',''))).casefold()
            # Independently pin the approved disabled template's entire shape.
            signature=(hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
                       if schema=='ctb.s2.live_permit_template.1' else None)
            if (schema=='ctb.s2.live_permit_template.1' and fields.get('authorized') is False
                    and signature=='8731b105a6901e9e12f38a45baf3da64b4c7d6e750e27a0e3c5a0e679caf285e'):
                templates.append(location);return
            if (re.search(r'permit|session|authorization',schema)
                    or set(fields)&{'authorized','allowed_operations','authorization_id','approval_reference','owner_approval_reference'}):
                findings.append({'location':location,'reason':'permission/session record is not an approved disabled template'})
            for key,item in fields.items():
                if (location.endswith('.properties') and key=='live_permit'
                        and item=={'type':['string','null'],'minLength':1}):continue
                if key in {'session','sessions','session_id','full_session','live_session','permit','live_permit','authorization',
                           'access_token','api_key','password','secret','authorization_token','owner_identity','private_notes','hostname','account_name'}:
                    # A permit key may contain the exact approved template; all
                    # other sensitive fields are forbidden even if set false.
                    if key in {'permit','live_permit','authorization'} and isinstance(item,dict):
                        template_hash=hashlib.sha256(json.dumps(item,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
                        if template_hash=='8731b105a6901e9e12f38a45baf3da64b4c7d6e750e27a0e3c5a0e679caf285e':
                            inspect(item,location+'.'+key);continue
                    findings.append({'location':location+'.'+key,'reason':'private field'})
                else:inspect(item,location+'.'+key)
        elif isinstance(value,list):
            for i,item in enumerate(value):inspect(item,f'{location}[{i}]')
        elif isinstance(value,str):embedded(value,location)
    def embedded(text,location):
        decoder=json.JSONDecoder();index=0
        while index<len(text):
            starts=[i for i in (text.find('{',index),text.find('[',index)) if i>=0]
            if not starts:break
            start=min(starts)
            try:value,length=decoder.raw_decode(text[start:])
            except ValueError:index=start+1;continue
            inspect(value,location+':embedded');index=start+length
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():findings.append({'location':str(path.relative_to(root)),'reason':'symlink'});continue
        if not path.is_file():continue
        relative=str(path.relative_to(root))
        if path.suffix in {'.py','.toml'}:
            source_inventory.append({'path':relative,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});continue
        text=path.read_text()
        try:
            if path.suffix=='.json':inspect(json.loads(text),relative)
            elif path.suffix in {'.yaml','.yml'}:
                import yaml
                inspect(yaml.safe_load(text),relative)
            else:embedded(text,relative)
        except (ValueError,RecursionError) as exc:findings.append({'location':relative,'reason':'uninspectable data: '+str(exc)})
        scanned.append(relative)
    return {'schema_version':'ctb.independent_public_postcheck.v1','findings':findings,
            'data_files_scanned':scanned,'source_inventory':source_inventory,'approved_disabled_templates':templates,
            'scope':'data schemas/fields and embedded JSON; no execution; not a comprehensive secret or license audit'}
