import json
import sys
import time
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import db
from backend.config import ASSETS
from backend.schema import SceneSpec
from backend.examples import robot, lamp, chair
from backend.agent import generate, validate, finish, write_json

db.init()
for factory in (robot, lamp, chair):
    tid = uuid.uuid5(uuid.NAMESPACE_DNS, 'forma-studio-example-'+factory.__name__).hex
    existing = db.get(tid)
    if existing and existing['status'] == 'completed':
        print('EXISTS', factory.__name__, flush=True)
        continue
    spec = SceneSpec.model_validate(factory()).model_dump(exclude_defaults=True)
    folder = ASSETS/tid
    folder.mkdir(exist_ok=True)
    analysis = {'title':spec['title'],'summary':spec['description'],'requirements':['可交互展示','PBR 材质','独立部件'],'optimized_prompt':spec['description']}
    write_json(folder/'design.json',spec)
    write_json(folder/'analysis.json',analysis)
    task={'id':tid,'title':spec['title'],'prompt':spec['description'],'status':'queued','stage':'queued','progress':0,'cancel_requested':False,'engine':'procedural','quality':'standard','style':'industrial','is_example':True,'formats':[],'analysis':analysis,'materials':spec['materials'],'part_count':len(spec['parts']),'started_at':time.time(),'model':'预置程序化示例'}
    if existing: db.update(tid,**{k:v for k,v in task.items() if k!='id'})
    else: db.create(task)
    print('BUILD',factory.__name__,tid,flush=True)
    try:
        generate({'task_id':tid})
        report=validate({'task_id':tid})
        finish({'task_id':tid,**report})
        print('COMPLETE',factory.__name__,flush=True)
    except Exception as exc:
        db.update(tid,status='failed',error=str(exc))
        print('FAILED',factory.__name__,type(exc).__name__,str(exc),flush=True)
        raise
