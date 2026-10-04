import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'data/verification'
browser=json.loads((out/'browser-result.json').read_text())
suite=ET.parse(out/'pytest.xml').getroot().find('testsuite')
archive=out/'download-test.zip'
source=root/'data/assets/f950d435a0435bd1ad92e6205d22e883/asset.zip'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(archive)==sha(source), 'Downloaded ZIP differs from source'
with zipfile.ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    names=zipped.namelist()
    assert {'model.glb','model.obj','model.mtl','model.stl','model.fbx','scene.blend','report.json'}<=set(names)
assert not browser['errors'] and not browser['mobileOverflow']
report={
    'backend_tests':{k:suite.attrib[k] for k in ('tests','failures','errors','time')},
    'browser':browser,
    'download':{'sha256':sha(archive),'bytes':archive.stat().st_size,'matches_source':True,'zip_integrity':True,'files':names},
    'model_integration':'Scripted LLM replies + actual LangGraph, Blender and Trimesh verified. Cloud LLM not verified; existing credential returned HTTP 401.',
    'neural_reconstruction':'TripoSR / Hunyuan adapters implemented; weights and inference environments not installed or verified.',
}
(out/'release-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'tests':report['backend_tests'],'zip_bytes':report['download']['bytes'],'zip_matches':True,'browser_errors':len(browser['errors'])}))
