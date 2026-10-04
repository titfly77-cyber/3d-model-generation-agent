"""Scripted cloud output -> real LangGraph, Blender, textures and three renders.

This fixture is deliberately isolated from the user's assets. It verifies the
pipeline, not actual image-generation or neural reconstruction model quality.
"""
import json
import uuid
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from backend import agent, db, neural_pipeline as pipeline


def test_neural_graph_with_real_blender_and_embedded_texture(tmp_path, monkeypatch):
    assets, uploads = tmp_path/'assets', tmp_path/'uploads'
    assets.mkdir()
    uploads.mkdir()
    for module, attribute, value in [(db,'DATA',tmp_path), (agent,'DATA',tmp_path),
                                    (agent,'ASSETS',assets), (agent,'UPLOADS',uploads)]:
        monkeypatch.setattr(module, attribute, value)
    cfg = {'blender_path': 'D:/blender/blender.exe', 'model': 'scripted-vision',
           'visual_review_enabled': True, 'max_quality_retries': 0}
    monkeypatch.setattr(agent, 'settings', lambda: cfg)
    db.init()
    image_id, tid = uuid.uuid4().hex, uuid.uuid4().hex
    reference = Image.new('RGB', (384, 384), 'white')
    ImageDraw.Draw(reference).rectangle((80, 40, 300, 330), fill='#266a91')
    reference.save(uploads/(image_id+'.png'))
    (assets/tid).mkdir()
    db.create({'id': tid, 'engine': 'neural', 'status': 'queued', 'prompt': 'A textured box test fixture',
               'style': 'minimal', 'quality': 'draft', 'image_id': image_id, 'cancel_requested': False})

    def complete(system, text, **kwargs):
        result = ({'title': 'Scripted integration fixture', 'requirements': ['textured box'],
                   'optimized_prompt': 'A blue textured box'} if system.startswith('Plan a 3D asset')
                  else {'passed': True, 'summary': 'Scripted review result', 'score': 90, 'issues': []})
        return json.dumps(result), {'input_tokens': 1, 'output_tokens': 1}

    calls = []
    def reconstruct(views, output, cfg, check_cancel, *args):
        assert list(views) == ['front']
        check_cancel()
        calls.append(1)
        texture = Image.new('RGB', (64, 64), '#266a91')
        ImageDraw.Draw(texture).rectangle((0, 0, 31, 31), fill='#ef9237')
        mesh = trimesh.creation.box(extents=(1, .5, 1.6))
        uv = (mesh.vertices[:, (0, 2)] + (0.5, 0.8)) / (1, 1.6)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv,
            material=trimesh.visual.material.PBRMaterial(baseColorTexture=texture, metallicFactor=.1, roughnessFactor=.5))
        output.write_bytes(mesh.export(file_type='glb'))
        return {'model': 'scripted-textured-mesh', 'used_views': ['front'], 'textured': True}

    monkeypatch.setattr(pipeline, 'complete', complete)
    monkeypatch.setattr(pipeline, 'reconstruct_views', reconstruct)
    monkeypatch.setattr(pipeline, 'generate_sheet', lambda *args: (_ for _ in ()).throw(AssertionError('Photo route must not generate images')))
    runner = agent.Runner()
    try:
        runner._run(tid)
        task = db.get(tid)
        assert task['status'] == 'completed', task.get('error')
        out = assets/tid
        assert task['report']['verified']
        assert task['report']['texture_images'] == 1
        assert task['report']['review_renders'] == ['front', 'left', 'back']
        for view in ('front', 'left', 'back'):
            with Image.open(out/f'render_{view}.png') as rendered:
                assert rendered.size == (384, 384)
                assert np.asarray(rendered).std() > 5
        textures = list((out/'textures').glob('*.png'))
        assert len(textures) == 1
        assert 'textures' in (out/'model.mtl').read_text()
        scene = trimesh.load(out/'model.glb', force='scene')
        assert any(getattr(mesh.visual.material, 'baseColorTexture', None) is not None for mesh in scene.geometry.values())
        assert {'glb', 'obj', 'fbx', 'stl', 'blend'} <= set(task['formats'])
        timestamp = (out/'model.glb').stat().st_mtime_ns
        runner._run(tid)
        assert len(calls) == 1 and (out/'model.glb').stat().st_mtime_ns == timestamp
    finally:
        runner.close()
        runner.connection.close()
