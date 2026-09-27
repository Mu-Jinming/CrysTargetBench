"""Test-only adapter: reads authored JSON observations, never fabricates physics.

No imports from CTB core implementation internals except the public SDK record
validators/helpers. This package is built and installed separately in conformance.
"""
from importlib.resources import files
from pathlib import Path
import json
from crystargetbench.dft.validation import (validate_request, artifact, identity_mapping,
    read_record, observation, result_record)

class Adapter:
    def describe(self):
        return json.loads(files('ctb_second_example').joinpath('ctb_dft_adapter.json').read_text())

    def prepare(self, request, directory, configuration):
        validate_request(request)
        if configuration != {}:raise ValueError('test adapter accepts no configuration overrides')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        (directory/'fixture-input.json').write_text(json.dumps(request,sort_keys=True))
        return {'schema_version':'ctb.dft.prepared.v1','request_id':request['request_id'],
            'request_digest':request['request_digest'],'status':'ready',
            'files':[artifact(directory,'fixture-input.json','input')],
            'mapping':identity_mapping(len(request['geometry']['species'])),'reasons':[]}

    def collect(self, request, directory, configuration):
        validate_request(request)
        if configuration != {}:raise ValueError('test adapter accepts no configuration overrides')
        data=read_record(Path(directory)/'authored-observations.json')
        if data['request_digest']!=request['request_digest'] or data['method']!=request['method']:
            raise ValueError('fixture binding mismatch')
        refs=[artifact(directory,'authored-observations.json')]
        observations=[observation(q, v['value'], definition_id='synthetic.second.'+q,
            diagnostics=v.get('diagnostics',{}),raw_artifacts=refs,evidence_kind='synthetic',
            qualified=data['scf']) for q,v in data['observations'].items()]
        return result_record(request,self.describe(),observations,
            observed_geometry=data.get('observed_geometry',request['geometry']),
            scf=data['scf'],ionic=data.get('ionic'),response=data.get('response'),evidence_mode='fixture')
