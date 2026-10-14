"""Stdlib lease tests: no WSL, processes, keys, model or GPU."""
import contextlib
import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock

source=pathlib.Path(__file__).resolve().parents[1]/'keepalive.py'
spec=importlib.util.spec_from_file_location('keepalive_fixture',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
GEN='a'*32;LEASE='b'*32;START='synthetic-owner-created'
GOOD={'Generation':GEN,'SupervisorPid':12345,'SupervisorStarted':START,
      'LifetimeJobArmed':True,'HeartbeatSequence':1,'KeepaliveLeaseId':LEASE}


class KeepaliveTests(unittest.TestCase):
    def lease(self):return m.HeartbeatLease(GEN,LEASE,12345,START,0)

    def test_first_heartbeat_and_exact_stale_boundary(self):
        lease=self.lease();self.assertIsNone(lease.observe(GOOD,0))
        self.assertIsNone(lease.observe(GOOD,29.999))
        self.assertEqual(lease.observe(GOOD,30),'heartbeat_stale')

    def test_increasing_sequence_extends_only_this_lease(self):
        lease=self.lease();lease.observe(GOOD,0)
        self.assertIsNone(lease.observe({**GOOD,'HeartbeatSequence':2},25))
        self.assertIsNone(lease.observe({**GOOD,'HeartbeatSequence':2},54.999))
        self.assertEqual(lease.observe({**GOOD,'HeartbeatSequence':2},55),'heartbeat_stale')

    def test_updated_wallclock_does_not_extend_frozen_sequence(self):
        lease=self.lease();lease.observe(GOOD,0)
        self.assertEqual(lease.observe({**GOOD,'Updated':'2099-01-01','HeartbeatSequence':1},30),'heartbeat_stale')

    def test_new_owner_generation_pid_creation_all_exit(self):
        for change in ({'Generation':'c'*32},{'SupervisorPid':12346},{'SupervisorPid':True},{'SupervisorStarted':'other'}):
            self.assertEqual(self.lease().observe({**GOOD,**change},0),'owner_changed')

    def test_restart_within_same_owner_expires_previous_helper(self):
        self.assertEqual(self.lease().observe({**GOOD,'KeepaliveLeaseId':'d'*32},0),'lease_replaced')

    def test_job_unarmed_exits(self):
        self.assertEqual(self.lease().observe({**GOOD,'LifetimeJobArmed':False},0),'ownership_not_armed')

    def test_invalid_sequence_does_not_refresh_lease(self):
        for sequence in (None,True,-1,2**63,1.0,'2'):
            lease=self.lease();lease.observe(GOOD,0)
            self.assertIsNone(lease.observe({**GOOD,'HeartbeatSequence':sequence},29))
            self.assertEqual(lease.observe({**GOOD,'HeartbeatSequence':sequence},30),'state_unreadable')

    def test_regressed_sequence_exits(self):
        lease=self.lease();lease.observe({**GOOD,'HeartbeatSequence':10},0)
        self.assertEqual(lease.observe(GOOD,1),'heartbeat_regressed')

    def test_repeated_read_errors_cannot_extend_lifetime(self):
        lease=self.lease();lease.observe(GOOD,0)
        self.assertIsNone(lease.unreadable(10));self.assertIsNone(lease.unreadable(29))
        self.assertEqual(lease.unreadable(30),'state_unreadable')

    def test_non_dict_state_cannot_extend_lifetime(self):
        lease=self.lease();lease.observe(GOOD,0)
        self.assertEqual(lease.observe([],30),'state_unreadable')

    def test_state_read_is_size_bounded_and_malformed_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path=pathlib.Path(temp)/'state.json'
            path.write_text(json.dumps(GOOD));self.assertEqual(m.read_owner_state(path),GOOD)
            for raw in ('{', 'x'*(m.MAX_STATE_BYTES+1)):
                path.write_text(raw)
                with self.assertRaises(ValueError):m.read_owner_state(path)

    def test_proc_starttime_parses_comm_with_spaces_and_parentheses(self):
        raw='123 (name with ) parentheses) S '+' '.join(str(i) for i in range(4,23))
        with mock.patch.object(pathlib.Path,'read_text',return_value=raw):self.assertEqual(m.process_starttime(),22)

    def fake_run(self, reads, times):
        records=[]
        with contextlib.ExitStack() as stack:
            for obj,name,value in ((m.sys,'platform','linux'),):stack.enter_context(mock.patch.object(obj,name,value))
            stack.enter_context(mock.patch.object(m.os,'getuid',return_value=1000,create=True))
            stack.enter_context(mock.patch.object(m.os,'umask',return_value=0o077))
            stack.enter_context(mock.patch.object(m,'private_directory'))
            stack.enter_context(mock.patch.object(m,'process_starttime',return_value=54321))
            stack.enter_context(mock.patch.object(m,'write_record',side_effect=lambda path,row,create=False:records.append(dict(row))))
            stack.enter_context(mock.patch.object(m,'read_owner_state',side_effect=reads))
            stack.enter_context(mock.patch.object(m.time,'monotonic',side_effect=times))
            self.sleep=stack.enter_context(mock.patch.object(m.time,'sleep'))
            stack.enter_context(mock.patch.object(m.signal,'signal',return_value=m.signal.SIG_DFL))
            m.run(pathlib.Path('synthetic-state'),GEN,LEASE,12345,START,pathlib.Path('synthetic-records'))
        return records

    def test_loop_missing_state_exits_and_records_identity(self):
        records=self.fake_run([FileNotFoundError(),FileNotFoundError()],[0,0,0.2])
        self.assertEqual(records[-1]['reason'],'owner_state_removed')
        self.assertEqual(self.sleep.call_args_list[0].args,(m.MISSING_CONFIRM_SECONDS,))
        self.assertEqual(records[-1]['linux_starttime_ticks'],54321)
        self.assertEqual(records[-1]['state'],'exited')

    def test_transient_missing_state_then_valid_state_does_not_exit(self):
        # DrvFS ENOENT during a Windows File.Replace, then the replaced state.
        records=self.fake_run([FileNotFoundError(),GOOD,GOOD],[0,0,0.2,30.2])
        self.assertEqual(records[-1]['reason'],'heartbeat_stale')

    def test_second_miss_sooner_than_confirmation_interval_does_not_confirm(self):
        records=self.fake_run([FileNotFoundError()]*3,[0,0,0.1,0.3])
        self.assertEqual(records[-1]['reason'],'owner_state_removed')

    def test_miss_then_other_error_restarts_confirmation(self):
        records=self.fake_run([FileNotFoundError(),PermissionError("x"),FileNotFoundError(),FileNotFoundError()],[0,0,0.25,1.25,1.5])
        self.assertEqual(records[-1]['reason'],'owner_state_removed')

    def test_missing_state_cannot_extend_lifetime(self):
        records=self.fake_run([GOOD,FileNotFoundError()],[0,0,30])
        self.assertEqual(records[-1]['reason'],'state_unreadable')

    def test_loop_permission_error_is_bounded_and_sanitized(self):
        records=self.fake_run([GOOD,PermissionError('PRIVATE_PATH')],[0,0,30])
        self.assertEqual(records[-1]['reason'],'state_unreadable')
        self.assertNotIn('PRIVATE_PATH',json.dumps(records))

    def test_loop_stale_identity_record(self):
        records=self.fake_run([GOOD,GOOD],[0,0,30])
        self.assertEqual(records[-1]['reason'],'heartbeat_stale')
        self.assertEqual(records[-1]['lease_id'],LEASE)
        self.assertEqual(records[-1]['last_sequence'],1)


if __name__=='__main__':unittest.main(verbosity=2)
