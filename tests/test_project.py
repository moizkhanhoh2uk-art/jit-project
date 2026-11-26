import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from core import *
@pytest.fixture
def grant(): return Grant(frozenset(['ec2:StopInstances']),frozenset([resource_for('ec2:StopInstances')]),True,10,20)
@pytest.mark.parametrize('now,expected',[(9,False),(10,True),(19.999,True),(20,False),(21,False)])
def test_time_boundary(grant,now,expected): assert check('ec2:StopInstances',resource_for('ec2:StopInstances'),grant,now).allowed==expected
@pytest.mark.parametrize('action,resource,approved', [('ec2:StartInstances',resource_for('ec2:StopInstances'),True),('ec2:StopInstances','wrong',True),('ec2:StopInstances',resource_for('ec2:StopInstances'),False),('unknown',resource_for('ec2:StopInstances'),True)])
def test_denials(action,resource,approved):
 g=Grant(frozenset(['ec2:StopInstances']),frozenset([resource_for('ec2:StopInstances')]),approved,0,100)
 assert not check(action,resource,g,50).allowed
@pytest.mark.parametrize('action',ACTIONS)
def test_each_action_allowed(action):
 r=resource_for(action);g=Grant(frozenset([action]),frozenset([r]),True,0,100)
 assert check(action,r,g,50).allowed
@pytest.mark.parametrize('action,resource',[('s3:ListBucket',resource_for('s3:GetObject')),('s3:GetObject',resource_for('s3:ListBucket')),('ec2:StopInstances',resource_for('s3:GetObject'))])
def test_resource_type(action,resource):
 g=Grant(frozenset(ACTIONS),frozenset([resource]),True,0,100)
 assert not check(action,resource,g,50).allowed
def test_dataset_integrity():
 d,m=train();tr=d[d.split=='train'];te=d[d.split=='test']
 assert len(d)==256 and len(tr)==192 and len(te)==64
 assert not set(tr.family)&set(te.family)
 assert not set(tr.text.str.lower())&set(te.text.str.lower())
 assert set(tr.label)==set(te.label)==set(ACTIONS)
 assert set(m.predict(te.text))<=set(ACTIONS)
def test_policy_can_block_or_propagate_error():
 r=resource_for('ec2:StopInstances')
 narrow=Grant(frozenset(['ec2:StopInstances']),frozenset([r]),True,0,100)
 broad=Grant(frozenset(ACTIONS),frozenset([r]),True,0,100)
 assert not check('ec2:StartInstances',r,narrow,50).allowed
 assert check('ec2:StartInstances',r,broad,50).allowed
 assert check('ec2:StopInstances',r,broad,50).allowed
@pytest.mark.parametrize('text,action',[('please stop the instance','ec2:StopInstances'),('upload the object','s3:PutObject'),('reboot the instance','ec2:RebootInstances')])
def test_keyword_examples(text,action): assert keyword(text)==action
