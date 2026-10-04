import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import {
  RotateCcw,
  Grid2x2,
  Box,
  Rotate3D,
  Maximize2,
  LoaderCircle,
  AlertTriangle,
} from "lucide-react";

export default function Viewer({ url, wireframe, setWireframe, onCapture }) {
  const holder = useRef(null),
    api = useRef(null);
  const [loading, setLoading] = useState(false),
    [error, setError] = useState(""),
    [grid, setGrid] = useState(true),
    [rotate, setRotate] = useState(false);
  useEffect(() => {
    const element = holder.current;
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true,
      });
    } catch {
      setError("当前浏览器无法启动 WebGL，请启用硬件加速后刷新。");
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    element.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 500);
    camera.position.set(5, 3.7, 5.5);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.minDistance = 0.2;
    controls.maxDistance = 60;
    controls.maxPolarAngle = Math.PI * 0.54;
    const pmrem = new THREE.PMREMGenerator(renderer);
    const room = new RoomEnvironment();
    const env = pmrem.fromScene(room, 0.04);
    scene.environment = env.texture;
    scene.environmentIntensity = 0.75;
    room.dispose();
    const ambient = new THREE.HemisphereLight(0xe4eedc, 0x414036, 0.7);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xfff2dc, 2);
    key.position.set(4, 7, 5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -5;
    key.shadow.camera.right = 5;
    key.shadow.camera.top = 5;
    key.shadow.camera.bottom = -5;
    key.shadow.normalBias = 0.03;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xe6efff, 1.2);
    rim.position.set(-4, 3, -4);
    scene.add(rim);
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(200, 200),
      new THREE.ShadowMaterial({ color: 0x000000, opacity: 0.32 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    floor.position.y = -0.006;
    scene.add(floor);
    const lines = new THREE.GridHelper(18, 36, 0x45473d, 0x30332c);
    lines.material.transparent = true;
    lines.material.opacity = 0.48;
    scene.add(lines);
    const axes = new THREE.Group();
    const fit = (object) => {
      const bounds = new THREE.Box3().setFromObject(object);
      const size = bounds.getSize(new THREE.Vector3());
      const center = bounds.getCenter(new THREE.Vector3());
      const max = Math.max(size.x, size.y, size.z);
      controls.target.copy(center);
      camera.position
        .copy(center)
        .add(new THREE.Vector3(1.5, 1.02, 1.85).multiplyScalar(max * 1.1));
      camera.near = max / 1000;
      camera.far = max * 150;
      camera.updateProjectionMatrix();
      controls.minDistance = max * 0.5;
      controls.maxDistance = max * 7;
      controls.update();
    };
    const resize = new ResizeObserver(() => {
      const w = element.clientWidth,
        h = element.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    resize.observe(element);
    let frame;
    const draw = () => {
      frame = requestAnimationFrame(draw);
      controls.update();
      renderer.render(scene, camera);
    };
    draw();
    api.current = {
      scene,
      renderer,
      camera,
      controls,
      lines,
      fit,
      object: null,
    };
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        o.geometry?.dispose();
        if (o.material) {
          (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) =>
            m.dispose(),
          );
        }
      });
      env.dispose();
      pmrem.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      api.current = null;
    };
  }, []);
  useEffect(() => {
    const context = api.current;
    if (!context) return;
    if (context.object) {
      context.scene.remove(context.object);
      context.object.traverse((o) => {
        o.geometry?.dispose();
        if (o.material)
          (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) =>
            m.dispose(),
          );
      });
      context.object = null;
    }
    if (!url) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setError("");
    let active = true;
    new GLTFLoader().load(
      url,
      (gltf) => {
        if (!active) {
          gltf.scene.traverse((o) => {
            o.geometry?.dispose();
            if (o.material)
              (Array.isArray(o.material) ? o.material : [o.material]).forEach(
                (m) => m.dispose(),
              );
          });
          return;
        }
        const object = gltf.scene;
        object.traverse((o) => {
          if (o.isMesh) {
            o.castShadow = true;
            o.receiveShadow = true;
            (Array.isArray(o.material) ? o.material : [o.material]).forEach(
              (m) => {
                m.wireframe = wireframe;
              },
            );
          }
        });
        context.scene.add(object);
        context.object = object;
        context.fit(object);
        setLoading(false);
      },
      undefined,
      () => {
        if (active) {
          setError("模型加载失败，请重新选择资产或刷新页面。");
          setLoading(false);
        }
      },
    );
    return () => {
      active = false;
    };
  }, [url]);
  useEffect(() => {
    const a = api.current;
    if (a) {
      a.lines.visible = grid;
      a.controls.autoRotate = rotate;
      a.controls.autoRotateSpeed = 0.7;
      a.object?.traverse((o) => {
        if (o.isMesh)
          (Array.isArray(o.material) ? o.material : [o.material]).forEach(
            (m) => {
              m.wireframe = wireframe;
            },
          );
      });
    }
  }, [wireframe, grid, rotate]);
  function screenshot() {
    const a = api.current;
    if (!a) return;
    a.renderer.render(a.scene, a.camera);
    const link = document.createElement("a");
    link.download = "forma-view.png";
    link.href = a.renderer.domElement.toDataURL("image/png");
    link.click();
  }
  return (
    <div className="viewer-wrap">
      <div className="canvas-host" ref={holder} data-testid="viewport" />
      {loading && (
        <div className="viewer-notice">
          <LoaderCircle className="spin" size={18} />
          加载三维资产
        </div>
      )}
      {error && (
        <div className="viewer-notice error">
          <AlertTriangle size={18} />
          {error}
        </div>
      )}
      <div className="view-tag">
        <span className="live-dot" />
        PERSPECTIVE <span className="muted">/</span> STUDIO LIGHT
      </div>
      <div className="view-axis">
        <span className="axis-y">Y</span>
        <span className="axis-z">Z</span>
        <span className="axis-x">X</span>
        <i />
      </div>
      <div className="viewer-tools">
        <button
          title="重置视角"
          aria-label="重置视角"
          onClick={() =>
            api.current?.object && api.current.fit(api.current.object)
          }
        >
          <RotateCcw size={17} />
        </button>
        <button
          title="网格"
          aria-label="网格"
          className={grid ? "active" : ""}
          onClick={() => setGrid(!grid)}
        >
          <Grid2x2 size={17} />
        </button>
        <button
          title="线框"
          aria-label="线框"
          className={wireframe ? "active" : ""}
          onClick={() => setWireframe(!wireframe)}
        >
          <Box size={17} />
        </button>
        <button
          title="自动旋转"
          aria-label="自动旋转"
          className={rotate ? "active" : ""}
          onClick={() => setRotate(!rotate)}
        >
          <Rotate3D size={18} />
        </button>
        <span />
        <button title="保存视图" aria-label="保存视图" onClick={screenshot}>
          <Maximize2 size={17} />
        </button>
      </div>
      <div className="view-instruction">
        拖动旋转 <i /> 滚轮缩放 <i /> 右键平移
      </div>
    </div>
  );
}
