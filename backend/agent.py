import base64
import json
import os
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict
import httpx
import trimesh
from pydantic import ValidationError
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from . import db
from .config import ROOT, DATA, ASSETS, UPLOADS, settings
from .llm import complete, extract_json, ModelError, ModelOutputError
from .schema import SceneSpec

DESIGN_SYSTEM = '''You are an expert industrial designer and Blender scene architect. Return ONLY a JSON object with title (Chinese, <=30 chars), description (Chinese design explanation), materials (dict), and parts (array). Construct a beautiful, coherent, detailed 3D asset using the scene DSL below. Never output code.
Coordinate system: Z UP, X width, Y depth, front is NEGATIVE Y. Ground z=0. Typical asset extent 2 to 4 units. Ensure structural connections, plausible supports, and tasteful layered details. Aim for 35-70 parts for robots/machines, 10-35 for furniture; use 5-8 materials. Use distinctive silhouettes and realistic proportions. Do not add floor, lights, cameras or presentation pedestals unless requested. No text labels made of fake geometry. Honor every explicit user constraint; for edits preserve unchanged geometry and material names.
Materials: {"shell":{"color":"#789564","metallic":0.25,"roughness":0.35,"emission":0}}. Hex colors, emission 0..5.
Part common fields: name (unique descriptive), type, material (existing name), position:[x,y,z], rotation:[degreesX,degreesY,degreesZ], scale:[x,y,z], bevel:0..0.5, segments:8..64. Omit defaults when not needed.
box: scale is FULL width/depth/height; default bevel .03.
sphere: scale is x/y/z RADII (unit sphere).
cylinder: radius and depth (total length), axis Z. Rotation [90,0,0] makes axle point Y, rotation [0,90,0] makes axle point X. Cylinder cap visible on local Z faces.
cone: radius (bottom), radius_top, depth, axis Z.
torus: radius (major), tube (minor), ring in XY plane. Wheel axles often X, so rotate wheel [0,90,0].
beam: start:[x,y,z],end:[x,y,z],radius creates a connected cylinder; omit position/rotation.
curve: points:[[x,y,z],...] with 2..16 points, tube for rounded cables or curved frames.
lathe: points:[[radius,z],...] generates a revolved profile about Z; use for vases, bottles, lamps and rounded shells. Profile should be sensible; position adds offset.
mesh: vertices:[[x,y,z],...],faces:[[index,index,index,...],...] for angular panels. All face indices must be valid. Use sparingly.
Use boxes and beams for mechanical structures, torus/cylinders for wheels, layered shapes for housings. Avoid unnecessary tessellation. Maximum 180 parts. Keep coordinates finite and geometry within +-20. Output compact valid JSON without markdown.'''


class State(TypedDict, total=False):
    task_id: str
    analysis: dict
    scene: dict
    report: dict


class Cancelled(Exception):
    pass


def check_cancel(task_id):
    if db.get(task_id).get('cancel_requested'):
        raise Cancelled()


def phase(task_id, stage, message, progress):
    check_cancel(task_id)
    db.update(task_id, stage=stage, progress=progress, status='running')
    db.event(task_id, stage, message)


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def add_usage(task_id, usage):
    task = db.get(task_id)
    current = task.get('usage', {})
    for key in ('input_tokens', 'output_tokens', 'prompt_tokens', 'completion_tokens'):
        if isinstance(usage.get(key), int):
            current[key] = current.get(key, 0) + usage[key]
    db.update(task_id, usage=current, model=settings()['model'])


def analyze(state):
    tid = state['task_id']
    phase(tid, 'analyze', '正在理解主体、外观和交付要求', 10)
    task = db.get(tid)
    path = ASSETS / tid / 'analysis.json'
    if path.exists():
        analysis = json.loads(path.read_text(encoding='utf-8'))
    else:
        context = '用户需求：' + task['prompt'] + '\n风格偏好：' + task['style']
        if task.get('parent_id'):
            parent = db.get(task['parent_id'])
            context += '\n这是对已有资产的修改。原始需求：' + parent['prompt'] + '\n原始设计：' + json.dumps(parent.get('analysis', {}), ensure_ascii=False)
        image_path = UPLOADS / (task['image_id'] + '.png') if task.get('image_id') else None
        output, usage = complete('Analyze a 3D asset request. Return JSON only: title (short Chinese), summary (Chinese), requirements (3-6 Chinese strings), optimized_prompt (detailed English generation description). Preserve user constraints. Do not claim printing readiness or engineering accuracy. For image inputs describe visible structure; hidden geometry is inferred. For edits summarize the resulting complete object.', context, image_path, max_tokens=1600, json_output=True)
        analysis = extract_json(output)
        if not isinstance(analysis, dict) or not isinstance(analysis.get('requirements'), list):
            raise ModelError('需求分析结果格式不正确，请重试。')
        analysis['title'] = str(analysis.get('title', task['prompt'][:24]))[:80]
        write_json(path, analysis)
        add_usage(tid, usage)
    db.update(tid, title=analysis['title'], analysis=analysis)
    db.event(tid, 'analyze', '设计需求已整理', {'requirements': analysis.get('requirements', [])})
    return {'analysis': analysis}


def design(state):
    tid = state['task_id']
    task = db.get(tid)
    phase(tid, 'design', '正在设计形体、部件与材质', 26)
    if task['engine'] != 'procedural':
        db.event(tid, 'design', '使用参考图进入网格重建引擎')
        return {'scene': {}}
    path = ASSETS / tid / 'design.json'
    if path.exists():
        scene = SceneSpec.model_validate_json(path.read_text(encoding='utf-8')).model_dump(exclude_defaults=True)
    else:
        prompt = json.dumps(state['analysis'], ensure_ascii=False) + '\nUser request: ' + task['prompt']
        if task.get('parent_id'):
            parent_path = ASSETS / task['parent_id'] / 'design.json'
            if parent_path.exists():
                prompt += '\nExisting scene to revise. Retain everything not affected by the request:\n' + parent_path.read_text(encoding='utf-8')
        image_path = UPLOADS / (task['image_id'] + '.png') if task.get('image_id') else None
        original_prompt = prompt
        diagnostic_folder = path.parent / 'diagnostics'
        diagnostic_folder.mkdir(exist_ok=True)
        run_id = str(time.time_ns())
        for attempt in range(2):
            check_cancel(tid)
            failure = None
            output, usage = '', {}
            try:
                output, usage = complete(DESIGN_SYSTEM, prompt, image_path, max_tokens=14000, json_output=True)
            except ModelOutputError as exc:
                output, usage, failure = exc.output, exc.usage, exc
            add_usage(tid, usage)
            check_cancel(tid)
            if failure is None:
                try:
                    scene = SceneSpec.model_validate(extract_json(output)).model_dump(exclude_defaults=True)
                except (ValidationError, ModelError) as exc:
                    failure = exc
            errors = []
            if isinstance(failure, ValidationError):
                errors = failure.errors(include_input=False, include_context=False, include_url=False)[:12]
            elif failure:
                errors = [{'type': getattr(failure, 'code', 'invalid_json'), 'msg': str(failure)}]
            diagnostic = {'attempt': attempt + 1, 'model': settings()['model'],
                          'valid': failure is None, 'usage': usage, 'errors': errors,
                          'final_output': output}
            # Store final JSON and error locations for diagnosis; never reasoning
            # content, API keys, request headers, or raw upstream error responses.
            diagnostic_path = diagnostic_folder / f'design-{run_id}-{attempt + 1}.json'
            write_json(diagnostic_path, diagnostic)
            if failure is None:
                break
            if isinstance(failure, ValidationError):
                first = errors[0]
                location = '.'.join(str(item) for item in first.get('loc', [])) or 'scene'
                detail = f"{location}: {first['msg']}"
                summary = '场景字段未通过校验：' + detail
            else:
                summary = str(failure)
            db.event(tid, 'design', summary[:350], {'errors': errors, 'response': usage.get('_response', {})})
            if attempt == 1:
                raise ModelError(summary[:350] + ' 自动修正后仍未完成；具体原因已记录在执行轨迹和任务诊断文件中。') from failure
            db.event(tid, 'design', '正在根据具体错误重新生成完整场景')
            if isinstance(failure, ModelOutputError):
                prompt = original_prompt + '\nThe previous response had no complete final JSON. Return a complete compact JSON object now. Preserve every explicit user constraint; omit optional decoration if needed to finish within the output limit. Do not output prose or reasoning.'
            else:
                prompt = original_prompt + '\nPrevious invalid scene:\n' + output[:60000] + '\nValidation errors to fix:\n' + json.dumps(errors, ensure_ascii=False) + '\nReturn the entire corrected JSON scene, not a patch.'
        write_json(path, scene)
    db.update(tid, part_count=len(scene['parts']), materials=scene['materials'])
    db.event(tid, 'design', f"场景设计完成：{len(scene['parts'])} 个部件，{len(scene['materials'])} 种材质")
    return {'scene': scene}


def process_command(tid, command, logfile, timeout=600, cwd=ROOT):
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with logfile.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), creationflags=flags)
        start = time.monotonic()
        try:
            while process.poll() is None:
                check_cancel(tid)
                if time.monotonic() - start > timeout:
                    raise RuntimeError('处理超时，已停止本次后台作业。可降低质量后重试。')
                time.sleep(.5)
            if process.returncode:
                raise RuntimeError('模型处理失败，请查看该任务的运行日志后重试。')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def reconstruct(task, folder):
    cfg = settings()
    image_path = UPLOADS / (task['image_id'] + '.png')
    raw = folder / 'raw.glb'
    if raw.exists():
        return raw
    if task['engine'] == 'hunyuan':
        url = cfg['hunyuan_url'].rstrip('/')
        job_file = folder / 'remote-job.json'
        with httpx.Client(timeout=60) as client:
            if job_file.exists():
                uid = json.loads(job_file.read_text())['uid']
            else:
                image = base64.b64encode(image_path.read_bytes()).decode()
                response = client.post(url + '/send', json={'image': image, 'texture': True, 'type': 'glb', 'seed': 42, 'num_inference_steps': 20})
                response.raise_for_status()
                uid = response.json()['uid']
                write_json(job_file, {'uid': uid})
            db.event(task['id'], 'generate', '已提交 Hunyuan 重建任务')
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                check_cancel(task['id'])
                response = client.get(url + '/status/' + uid)
                response.raise_for_status()
                body = response.json()
                if body.get('status') == 'completed':
                    content = base64.b64decode(body['model_base64'], validate=True)
                    if content[:4] != b'glTF':
                        raise RuntimeError('重建引擎没有返回有效 GLB。')
                    raw.write_bytes(content)
                    return raw
                if body.get('status') in ('failed', 'error'):
                    raise RuntimeError('Hunyuan 服务报告生成失败。')
                time.sleep(3)
            raise RuntimeError('Hunyuan 任务等待超时，作业编号已保存，重试将继续查询原任务。')
    else:
        repo = ROOT.parent / 'TripoSR'
        command = [cfg['triposr_python'], str(repo/'run.py'), str(image_path), '--output-dir', str(folder/'triposr'), '--model-save-format', 'glb', '--pretrained-model-name-or-path', cfg['triposr_model'], '--chunk-size', '4096']
        process_command(task['id'], command, folder/'reconstruction.log', timeout=900, cwd=repo)
        source = folder/'triposr/0/mesh.glb'
        if not source.exists():
            raise RuntimeError('TripoSR 未输出预期 mesh.glb。')
        raw.write_bytes(source.read_bytes())
        return raw


def generate(state):
    tid = state['task_id']
    phase(tid, 'generate', '正在生成几何并执行 Blender 处理', 50)
    task = db.get(tid)
    folder = ASSETS / tid
    if (folder/'report.json').exists() and (folder/'model.glb').exists():
        db.event(tid, 'generate', '复用已完成的模型处理结果')
        return {}
    executable = settings()['blender_path']
    if not executable or not Path(executable).is_file():
        raise RuntimeError('未找到 Blender，请在设置中填写 blender.exe 的完整路径。')
    args = [executable, '--background', '--factory-startup', '--threads', '4', '--python-exit-code', '1', '--python', str(ROOT/'backend/blender_worker.py'), '--', '--output', str(folder), '--quality', task['quality']]
    if task['engine'] == 'procedural':
        args += ['--scene', str(folder/'design.json')]
    else:
        raw = reconstruct(task, folder)
        args += ['--import-model', str(raw)]
    process_command(tid, args, folder/'blender.log')
    return {}


def validate(state):
    tid = state['task_id']
    phase(tid, 'validate', '正在验证导出文件、几何与材质', 90)
    folder = ASSETS / tid
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    scene = trimesh.load(folder/'model.glb', force='scene')
    if not scene.geometry or not all(len(mesh.vertices) and len(mesh.faces) for mesh in scene.geometry.values()):
        raise RuntimeError('GLB 回读验证未通过，请重试。')
    report['checks'] = [c for c in report['checks'] if c['label'] != '独立回读验证']
    report['checks'].append({'label': '独立回读验证', 'passed': True, 'detail': f'{len(scene.geometry)} 个几何对象可正常加载'})
    report['verified'] = True
    write_json(folder/'report.json', report)
    formats = [ext for ext in ['glb', 'obj', 'stl', 'fbx'] if (folder/f'model.{ext}').exists()]
    if (folder/'scene.blend').exists():
        formats.append('blend')
    db.update(tid, report=report, formats=formats, has_preview=(folder/'preview.png').exists())
    return {'report': report}


def finish(state):
    tid = state['task_id']
    check_cancel(tid)
    task = db.get(tid)
    elapsed = time.time() - task.get('started_at', time.time())
    db.update(tid, status='completed', stage='complete', progress=100, elapsed_seconds=round(elapsed, 1), error=None)
    db.event(tid, 'complete', '资产已完成并通过文件回读验证，可预览与下载')
    return {}


class Runner:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='3d-agent')
        self.pending = set()
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(DATA/'checkpoints.db', check_same_thread=False)
        graph = StateGraph(State)
        for name, node in [('analyze', analyze), ('design', design), ('generate', generate), ('validate', validate), ('finish', finish)]:
            graph.add_node(name, node)
        graph.add_edge(START, 'analyze')
        for source, target in [('analyze','design'), ('design','generate'), ('generate','validate'), ('validate','finish'), ('finish',END)]:
            graph.add_edge(source, target)
        self.graph = graph.compile(checkpointer=SqliteSaver(self.connection))
        from .neural_pipeline import build_graph
        self.neural_graph = build_graph(SqliteSaver(self.connection))

    def submit(self, task_id):
        with self.lock:
            if task_id in self.pending:
                return False
            self.pending.add(task_id)
        self.pool.submit(self._run, task_id)
        return True

    def _run(self, tid):
        try:
            if db.get(tid).get('status') in ('completed', 'needs_review'):
                return
            check_cancel(tid)
            db.update(tid, status='running', started_at=time.time(), error=None)
            cfg = {'configurable': {'thread_id': tid}}
            workflow = self.neural_graph if db.get(tid).get('engine') == 'neural' else self.graph
            snapshot = workflow.get_state(cfg)
            workflow.invoke(None if snapshot.next else {'task_id': tid}, cfg)
        except Cancelled:
            db.update(tid, status='cancelled', stage='cancelled')
            db.event(tid, 'cancelled', '任务已停止，已完成的中间结果保留')
        except Exception as exc:
            if isinstance(exc, (ModelError, RuntimeError)):
                message = str(exc)[:500]
            elif isinstance(exc, httpx.HTTPError):
                message = '重建服务请求失败，请检查服务地址和运行状态。'
            else:
                message = '任务执行异常：' + type(exc).__name__ + '。可重试或查看运行日志。'
            db.update(tid, status='failed', error=message)
            db.event(tid, 'failed', message)
        finally:
            with self.lock:
                self.pending.discard(tid)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
