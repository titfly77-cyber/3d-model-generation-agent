"""Runs only inside an isolated Blender background process; consumes validated JSON."""
import argparse
import json
import math
import sys
from pathlib import Path
import bpy
import bmesh
from mathutils import Vector


def material(name, spec):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    value = spec.get('color', '#aaaaaa').lstrip('#')
    rgb = [int(value[i:i+2], 16) / 255 for i in (0, 2, 4)]
    # Hex colors are sRGB; Blender material inputs are linear.
    linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb]
    mat.diffuse_color = (*linear, 1)
    shader = mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*linear, 1)
    shader.inputs['Metallic'].default_value = spec.get('metallic', 0)
    shader.inputs['Roughness'].default_value = spec.get('roughness', .4)
    shader.inputs['Emission Color'].default_value = (*linear, 1)
    shader.inputs['Emission Strength'].default_value = spec.get('emission', 0)
    return mat


def create_part(p, materials):
    kind = p['type']
    segments = p.get('segments', 32)
    if kind == 'box':
        bpy.ops.mesh.primitive_cube_add(size=1)
    elif kind == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=max(8, segments//2), radius=1)
    elif kind in ('cylinder', 'cone', 'beam'):
        radius = p.get('radius', .2)
        depth = p.get('depth', 1)
        if kind == 'beam':
            depth = (Vector(p['end']) - Vector(p['start'])).length
        bpy.ops.mesh.primitive_cone_add(vertices=segments, radius1=radius, radius2=p.get('radius_top', 0) if kind == 'cone' else radius, depth=depth)
    elif kind == 'torus':
        bpy.ops.mesh.primitive_torus_add(major_segments=segments, minor_segments=12, major_radius=p.get('radius', .5), minor_radius=p.get('tube', .06))
    elif kind == 'curve':
        curve = bpy.data.curves.new(p['name'], 'CURVE')
        curve.dimensions = '3D'
        curve.resolution_u = 8
        curve.bevel_depth = p.get('tube', .03)
        curve.bevel_resolution = 3
        curve.use_fill_caps = True
        spline = curve.splines.new('BEZIER')
        spline.bezier_points.add(len(p['points']) - 1)
        for bp, co in zip(spline.bezier_points, p['points']):
            bp.co = co
            bp.handle_left_type = bp.handle_right_type = 'AUTO'
        obj = bpy.data.objects.new(p['name'], curve)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.convert(target='MESH')
    elif kind in ('lathe', 'mesh'):
        if kind == 'lathe':
            vertices = []
            faces = []
            for radius, height in p['points']:
                for i in range(segments):
                    angle = 2 * math.pi * i / segments
                    vertices.append((radius * math.cos(angle), radius * math.sin(angle), height))
            for j in range(len(p['points']) - 1):
                for i in range(segments):
                    n = (i+1) % segments
                    faces.append((j*segments+i, j*segments+n, (j+1)*segments+n, (j+1)*segments+i))
            faces += [tuple(reversed(range(segments))), tuple((len(p['points'])-1)*segments+i for i in range(segments))]
        else:
            vertices, faces = p['vertices'], p['faces']
        mesh = bpy.data.meshes.new(p['name'])
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(p['name'], mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    obj = bpy.context.object
    obj.name = p['name']
    obj.location = p.get('position', [0, 0, 0])
    obj.rotation_euler = [math.radians(v) for v in p.get('rotation', [0, 0, 0])]
    obj.scale = p.get('scale', [1, 1, 1])
    if kind == 'beam':
        start, end = Vector(p['start']), Vector(p['end'])
        obj.location = (start + end) / 2
        obj.rotation_euler = (end - start).to_track_quat('Z', 'Y').to_euler()
    obj.data.materials.append(materials[p['material']])
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if p.get('bevel', .03) > 0 and kind in ('box', 'cylinder', 'cone', 'mesh'):
        mod = obj.modifiers.new('Edge finish', 'BEVEL')
        mod.width = min(p.get('bevel', .03), min(obj.dimensions) / 4)
        mod.segments = 3
        bpy.ops.object.modifier_apply(modifier=mod.name)
    if kind != 'box':
        for poly in obj.data.polygons:
            poly.use_smooth = True
    obj.select_set(False)
    return obj


def clean(objects, target, preserve_surface=False):
    total = sum(max(1, len(p.vertices)-2) for o in objects for p in o.data.polygons)
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        if preserve_surface:
            # Imported reconstruction UV seams and split normals carry appearance.
            # Report topology problems instead of blindly welding or decimating it.
            obj.data.validate(clean_customdata=False)
            obj.data.update()
            continue
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=0.00001)
        bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=0.000001)
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.to_mesh(obj.data)
        bm.free()
        if total > target:
            mod = obj.modifiers.new('Web optimization', 'DECIMATE')
            mod.ratio = target / total
            bpy.ops.object.modifier_apply(modifier=mod.name)


def bounds(objects):
    corners = [o.matrix_world @ Vector(c) for o in objects for c in o.bound_box]
    lo = Vector([min(v[i] for v in corners) for i in range(3)])
    hi = Vector([max(v[i] for v in corners) for i in range(3)])
    return lo, hi


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def save_textures(objects, out, report):
    images = {}
    for obj in objects:
        for mat in obj.data.materials:
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        images[node.image.name] = node.image
    for index, image in enumerate(images.values()):
        try:
            textures = out/'textures'
            textures.mkdir(exist_ok=True)
            image.filepath_raw = str(textures/f'texture-{index:03d}.png')
            image.file_format = 'PNG'
            image.save()
            # glTF imports embed images with a packed source path. Clear that
            # stale reference after saving, so OBJ resolves our actual E: file.
            if image.packed_file:
                image.unpack(method='REMOVE')
            image.filepath_raw = str(textures/f'texture-{index:03d}.png')
        except Exception:
            report['warnings'].append('部分贴图无法单独导出，OBJ 外观可能不完整；请优先使用 GLB。')
    report['texture_images'] = len(images)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene')
    parser.add_argument('--import-model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--quality', default='standard')
    parser.add_argument('--review-views', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.render.threads_mode = 'FIXED'
    bpy.context.scene.render.threads = 4
    if args.import_model:
        bpy.ops.import_scene.gltf(filepath=args.import_model)
        objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        # Bake imported transforms before detaching hierarchy.
        for obj in objects:
            world = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world
        for obj in list(bpy.context.scene.objects):
            if obj.type != 'MESH':
                bpy.data.objects.remove(obj, do_unlink=True)
    else:
        spec = json.loads(Path(args.scene).read_text(encoding='utf-8'))
        mats = {name: material(name, value) for name, value in spec['materials'].items()}
        objects = [create_part(part, mats) for part in spec['parts']]
    if not objects:
        raise ValueError('No mesh objects were generated')
    target_faces = {'draft': 20000, 'standard': 60000, 'high': 120000}[args.quality]
    clean(objects, target_faces, preserve_surface=bool(args.import_model))
    bpy.context.view_layer.update()
    lo, hi = bounds(objects)
    offset = Vector(((lo.x+hi.x)/2, (lo.y+hi.y)/2, lo.z))
    for obj in objects:
        obj.location -= offset
    bpy.context.view_layer.update()
    lo, hi = bounds(objects)
    size = hi - lo
    report = {'objects': len(objects), 'vertices': 0, 'triangles': 0, 'non_manifold_edges': 0,
              'materials': len({m.name for o in objects for m in o.data.materials if m}),
              'dimensions': [round(v, 3) for v in size], 'unit': 'm', 'checks': [], 'warnings': []}
    for obj in objects:
        obj.data.calc_loop_triangles()
        report['vertices'] += len(obj.data.vertices)
        report['triangles'] += len(obj.data.loop_triangles)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        report['non_manifold_edges'] += sum(not e.is_manifold for e in bm.edges)
        bm.free()
    if not report['triangles'] or max(size) <= 0:
        raise ValueError('Empty geometry')
    if args.import_model:
        report['warnings'].append('保留重建网格的 UV、材质和表面细节；未自动填补孔洞或重新拓扑。尺寸为场景单位，未经实物标定。')
        if report['triangles'] > target_faces:
            report['warnings'].append('面数超过当前预览目标；为保留还原度未强制减面，低性能设备可能加载较慢。')
    save_textures(objects, out, report)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.export_scene.gltf(filepath=str(out/'model.glb'), export_format='GLB', use_selection=True, export_yup=True)
    bpy.ops.wm.obj_export(filepath=str(out/'model.obj'), export_selected_objects=True,
                          export_uv=True, export_materials=True, export_pbr_extensions=True, path_mode='RELATIVE')
    bpy.ops.wm.stl_export(filepath=str(out/'model.stl'), export_selected_objects=True)
    try:
        bpy.ops.export_scene.fbx(filepath=str(out/'model.fbx'), use_selection=True, object_types={'MESH'},
                                 add_leaf_bones=False, path_mode='COPY', embed_textures=True)
    except Exception:
        report['warnings'].append('当前 Blender 未能导出 FBX；其他格式不受影响。')
    report['file_bytes'] = (out/'model.glb').stat().st_size
    report['checks'] = [
        {'label': '有效几何', 'passed': True, 'detail': f"{report['triangles']:,} 个三角面"},
        {'label': 'GLB 导出', 'passed': True, 'detail': '包含几何与材质'},
        {'label': '坐标与朝向', 'passed': True, 'detail': '底部归零 · glTF Y-up'},
        {'label': '部件流形检查', 'passed': report['non_manifold_edges'] == 0, 'detail': f"{report['non_manifold_edges']} 条非流形边"},
    ]
    report['warnings'].append('这是展示资产。部件可能相交或彼此分离；未验证壁厚、装配精度及整体可打印性。')
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = {'draft': 12, 'standard': 24, 'high': 48}[args.quality]
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = {'draft': 512, 'standard': 720, 'high': 1024}[args.quality]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True
    scene.world.color = (.2, .2, .2)
    scale = max(size)
    target = Vector((0, 0, size.z * .48))
    bpy.ops.object.camera_add(location=(scale*1.6, -scale*2.1, scale*1.35))
    camera = bpy.context.object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = scale * 1.65
    look_at(camera, target)
    scene.camera = camera
    for name, location, power, light_size in [
        ('Key', (scale*2, -scale*3, scale*4), 250*scale*scale, scale*3),
        ('Fill', (-scale*3, -scale, scale*2), 170*scale*scale, scale*3),
        ('Rim', (scale, scale*3, scale*3), 300*scale*scale, scale*2),
    ]:
        bpy.ops.object.light_add(type='AREA', location=location)
        lamp = bpy.context.object
        lamp.name = name
        lamp.data.energy = power
        lamp.data.shape = 'DISK'
        lamp.data.size = light_size
        look_at(lamp, target)
    scene.view_settings.view_transform = 'AgX'
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'))
    try:
        scene.render.filepath = str(out/'preview.png')
        bpy.ops.render.render(write_still=True)
    except Exception:
        report['warnings'].append('静态预览渲染失败，仍可在网页查看三维模型。')
    if args.review_views:
        scene.render.resolution_x = scene.render.resolution_y = 384
        scene.cycles.samples = 12
        center = Vector((0, 0, size.z / 2))
        camera.data.ortho_scale = scale * 1.18
        report['review_renders'] = []
        for view, direction in [('front', (0, -1, 0)), ('left', (-1, 0, 0)), ('back', (0, 1, 0))]:
            try:
                camera.location = center + Vector(direction) * scale * 3
                look_at(camera, center)
                scene.render.filepath = str(out/f'render_{view}.png')
                bpy.ops.render.render(write_still=True)
                report['review_renders'].append(view)
            except Exception:
                report['warnings'].append(f'{view} 检查视图渲染失败；外观需人工复核。')
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('STUDIO_COMPLETE', json.dumps({'triangles': report['triangles'], 'objects': report['objects']}))


if __name__ == '__main__':
    run()
