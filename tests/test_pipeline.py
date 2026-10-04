"""Integration test: scripted model responses + real LangGraph, Blender and GLB validation."""
import json
import time
import uuid
from pathlib import Path
from backend import agent, db
from backend.examples import chair


def test_real_blender_pipeline_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DATA', tmp_path)
    monkeypatch.setattr(agent, 'DATA', tmp_path)
    assets=tmp_path/'assets';assets.mkdir()
    monkeypatch.setattr(agent, 'ASSETS', assets)
    cfg={'blender_path':'D:/blender/blender.exe','model':'scripted-test-provider'}
    monkeypatch.setattr(agent, 'settings', lambda:cfg)
    calls=[]
    def scripted_complete(system,text,image_path=None,max_tokens=0,**kwargs):
        calls.append(system)
        response=chair() if 'scene architect' in system else {'title':'Integration test chair','summary':'Scripted provider test','requirements':['chair','wood'],'optimized_prompt':'A wooden chair'}
        return json.dumps(response), {'input_tokens':1,'output_tokens':1}
    monkeypatch.setattr(agent,'complete',scripted_complete)
    db.init()
    tid=uuid.uuid4().hex
    (assets/tid).mkdir()
    db.create({'id':tid,'status':'queued','prompt':'A wooden chair','style':'minimal','quality':'draft','engine':'procedural','cancel_requested':False})
    runner=agent.Runner()
    try:
        runner._run(tid)
        result=db.get(tid)
        assert result['status']=='completed',result.get('error')
        assert result['report']['verified']
        assert result['report']['triangles']>100
        assert (assets/tid/'model.glb').read_bytes()[:4]==b'glTF'
        assert set(result['formats'])>={'glb','obj','stl','blend'}
        assert len(calls)==2
        snapshot=runner.graph.get_state({'configurable':{'thread_id':tid}})
        assert snapshot.values['task_id']==tid and not snapshot.next
        # Cached completed stages can be replayed without new model calls or geometry writes.
        timestamp=(assets/tid/'model.glb').stat().st_mtime_ns
        runner._run(tid)
        assert db.get(tid)['status']=='completed'
        assert (assets/tid/'model.glb').stat().st_mtime_ns==timestamp
        assert len(calls)==2
        assert sum(c['label']=='独立回读验证' for c in db.get(tid)['report']['checks'])==1
    finally:
        runner.close()
        runner.connection.close()
