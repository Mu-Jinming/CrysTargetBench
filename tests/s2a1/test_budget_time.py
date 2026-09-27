"""R2 persistent windows with controlled clocks; no physical execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
from crystargetbench.budget import Budget, BudgetExceeded, Cancelled, LedgerMigrationRequired, DEFAULT_TEST_LIMITS


@pytest.fixture
def clock(monkeypatch):
    value={'wall':1000.,'mono':100.}
    monkeypatch.setattr('crystargetbench.budget.time.time',lambda:value['wall'])
    monkeypatch.setattr('crystargetbench.budget.time.monotonic',lambda:value['mono'])
    return value


def budget(path,**limits):
    return Budget(path,{**DEFAULT_TEST_LIMITS,'max_sample_wall_seconds':10,**limits},identity='same-test-permit')


def advance(clock,seconds):
    clock['wall']+=seconds;clock['mono']+=seconds


@pytest.mark.parametrize('elapsed',[2,10,11])
def test_R2_resume_keeps_deadline_and_worker_remaining(tmp_path,clock,elapsed):
    path=tmp_path/'ledger.json'; b=budget(path);b.start_sample('geometry')
    original=b.snapshot()['samples']['geometry']
    advance(clock,elapsed); resumed=budget(path)
    if elapsed>=10:
        with pytest.raises(BudgetExceeded,match='max_sample_wall'): resumed.start_sample('geometry')
    else:
        resumed.start_sample('geometry')
        with resumed.deadline_scope(3):
            lease=resumed.reserve_evaluation('node',1)
            assert lease['remaining_seconds']=={'global':118.,'sample':8.,'node':3.}
            assert lease['max_wall_seconds']==3
            resumed.release(lease,success=True)
    assert resumed.snapshot()['samples']['geometry']==original


def test_R2_repeated_geometry_new_sample_and_global_bound(tmp_path,clock):
    b=budget(tmp_path/'b',max_total_wall_seconds=12);b.start_sample('a')
    advance(clock,8);b.start_sample('a');assert b.snapshot()['samples']['a']['deadline_wall']==1010
    b.start_sample('b');assert b.snapshot()['samples']['b']['deadline_wall']==1018
    lease=b.reserve_evaluation('n',1);assert lease['max_wall_seconds']==4;b.release(lease,success=True)
    advance(clock,4)
    with pytest.raises(BudgetExceeded,match='max_total_wall'): b.start_sample('b')


def test_R2_zero_and_cancelled_windows_do_not_renew(tmp_path,clock):
    p=tmp_path/'zero';b=budget(p,max_sample_wall_seconds=0)
    with pytest.raises(BudgetExceeded,match='max_sample_wall'):b.start_sample('g')
    assert b.snapshot()['samples']['g']['started_wall']==b.snapshot()['samples']['g']['deadline_wall']
    p=tmp_path/'cancel';b=budget(p);b.start_sample('g');b.cancel()
    with pytest.raises(Cancelled):budget(p).start_sample('g')


@pytest.mark.parametrize('anomaly',['wall_back','mono_back','wall_nan','mono_inf'])
def test_R2_clock_anomaly_latches_fail_closed(tmp_path,clock,anomaly):
    p=tmp_path/'ledger';b=budget(p);b.start_sample('g')
    if anomaly=='wall_back':clock['wall']-=1
    elif anomaly=='mono_back':clock['mono']-=1
    elif anomaly=='wall_nan':clock['wall']=float('nan')
    else:clock['mono']=float('inf')
    with pytest.raises(BudgetExceeded,match='clock_rollback'):b.check()
    clock.update(wall=1001.,mono=101.)
    with pytest.raises(BudgetExceeded,match='clock_rollback'):budget(p).start_sample('g')


def test_R2_frozen_wall_does_not_pause_active_monotonic_time(tmp_path,clock):
    b=budget(tmp_path/'b');b.start_sample('g');clock['mono']+=10
    with pytest.raises(BudgetExceeded,match='max_sample_wall'):b.check()


def test_R2_frozen_wall_across_instances_and_monotonic_reset(tmp_path,clock):
    p=tmp_path/'ledger';b=budget(p);b.start_sample('g');clock['mono']+=10
    with pytest.raises(BudgetExceeded,match='max_sample_wall'):budget(p).start_sample('g')
    clock['mono']=1;clock['wall']=1001
    with pytest.raises(BudgetExceeded,match='clock_rollback'):budget(p).start_sample('g')


def test_R2_v1_ledger_untouched_new_identity_requires_explicit_lineage(tmp_path,clock):
    old=tmp_path/'old';raw=b'{"schema_version":"ctb.budget.v1","identity":"old-permit"}'
    old.write_bytes(raw)
    with pytest.raises(LedgerMigrationRequired):budget(old)
    assert old.read_bytes()==raw
    new=Budget(tmp_path/'new',DEFAULT_TEST_LIMITS,identity='new-test-only-authorization',predecessor_ledger=old)
    assert new.snapshot()['authorization_lineage']=={'previous_identity':'old-permit','previous_ledger_sha256':hashlib.sha256(raw).hexdigest()}
    assert old.read_bytes()==raw
    with pytest.raises(ValueError,match='distinct'):Budget(tmp_path/'bad',DEFAULT_TEST_LIMITS,identity='old-permit',predecessor_ledger=old)


def test_R2_missing_v2_sample_records_cannot_reset_time(tmp_path,clock):
    p=tmp_path/'ledger';b=budget(p);b.start_sample('g');s=b.snapshot();s['samples']={};p.write_text(json.dumps(s))
    with pytest.raises(LedgerMigrationRequired):budget(p)


def test_R2_cross_actual_process_restore_without_sleep(tmp_path):
    script='''
import json,sys
from unittest.mock import patch
from crystargetbench.budget import Budget,DEFAULT_TEST_LIMITS,BudgetExceeded
with patch('crystargetbench.budget.time.time',return_value=float(sys.argv[2])),patch('crystargetbench.budget.time.monotonic',return_value=100.):
 b=Budget(sys.argv[1],dict(DEFAULT_TEST_LIMITS,max_sample_wall_seconds=10),identity='shared-permit')
 try:
  b.start_sample('same-geometry');l=b.reserve_evaluation('next',1);b.release(l,success=True)
  print(json.dumps({'status':'reserved','seconds':l['max_wall_seconds']}))
 except BudgetExceeded as e:print(json.dumps({'status':'blocked','reason':str(e)}))
'''
    def run(t):return json.loads(subprocess.check_output([sys.executable,'-c',script,str(tmp_path/'ledger'),str(t)],text=True))
    assert run(1000)=={'status':'reserved','seconds':10.}
    assert run(1007)=={'status':'reserved','seconds':3.}
    assert run(1010)['status']=='blocked'
    assert run(1009)['status']=='blocked'  # rollback after observed expiry cannot reopen
