import math
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class Material(BaseModel):
    color: str = '#91a575'
    metallic: float = Field(default=0.0, ge=0, le=1)
    roughness: float = Field(default=0.4, ge=0, le=1)
    emission: float = Field(default=0, ge=0, le=5)

    @field_validator('color')
    @classmethod
    def valid_color(cls, value):
        if len(value) != 7 or value[0] != '#':
            raise ValueError('Color must be #RRGGBB')
        int(value[1:], 16)
        return value


class Part(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    type: Literal['box', 'sphere', 'cylinder', 'cone', 'torus', 'beam', 'curve', 'lathe', 'mesh']
    material: str
    position: list[float] = Field(default_factory=lambda: [0, 0, 0], min_length=3, max_length=3)
    rotation: list[float] = Field(default_factory=lambda: [0, 0, 0], min_length=3, max_length=3)
    scale: list[float] = Field(default_factory=lambda: [1, 1, 1], min_length=3, max_length=3)
    radius: float = Field(default=0.2, gt=0, le=20)
    radius_top: float = Field(default=0.0, ge=0, le=20)
    depth: float = Field(default=1, gt=0, le=40)
    tube: float = Field(default=0.04, gt=0, le=5)
    bevel: float = Field(default=0.03, ge=0, le=0.5)
    segments: int = Field(default=32, ge=8, le=64)
    start: list[float] = Field(default_factory=lambda: [0, 0, 0], min_length=3, max_length=3)
    end: list[float] = Field(default_factory=lambda: [0, 0, 1], min_length=3, max_length=3)
    points: list[list[float]] = Field(default_factory=list, max_length=100)
    vertices: list[list[float]] = Field(default_factory=list, max_length=400)
    faces: list[list[int]] = Field(default_factory=list, max_length=600)

    @model_validator(mode='after')
    def bounds(self):
        for vector in [self.position, self.rotation, self.scale, self.start, self.end]:
            if not all(math.isfinite(x) and abs(x) <= 360 for x in vector):
                raise ValueError('Invalid coordinate')
        if any(x <= 0 or x > 40 for x in self.scale):
            raise ValueError('Scale must be positive and <=40')
        for point in self.points + self.vertices:
            if not all(math.isfinite(x) and abs(x) <= 40 for x in point):
                raise ValueError('Invalid geometry coordinate')
        if self.type == 'beam' and sum((a-b)**2 for a,b in zip(self.start,self.end)) < 1e-8:
            raise ValueError('Beam endpoints must differ')
        if self.type == 'lathe' and (len(self.points) < 2 or any(len(p) != 2 or p[0] < 0 for p in self.points)):
            raise ValueError('Lathe requires radius,height pairs')
        if self.type == 'curve' and (len(self.points) < 2 or any(len(p) != 3 for p in self.points)):
            raise ValueError('Curve requires 3D points')
        if self.type == 'mesh':
            if not self.vertices or not self.faces or any(len(v) != 3 for v in self.vertices):
                raise ValueError('Mesh needs vertices and faces')
            if any(len(f) < 3 or any(i < 0 or i >= len(self.vertices) for i in f) for f in self.faces):
                raise ValueError('Invalid mesh face index')
        return self


class SceneSpec(BaseModel):
    title: str = Field(max_length=80)
    description: str = Field(max_length=1500)
    materials: dict[str, Material] = Field(min_length=1, max_length=24)
    parts: list[Part] = Field(min_length=1, max_length=180)

    @model_validator(mode='after')
    def references(self):
        if any(p.material not in self.materials for p in self.parts):
            raise ValueError('Every part material must exist in materials')
        if len({p.name for p in self.parts}) != len(self.parts):
            raise ValueError('Part names must be unique')
        return self


class TaskRequest(BaseModel):
    prompt: str = Field(default='', max_length=5000)
    parent_id: str | None = None
    image_id: str | None = None
    engine: Literal['neural', 'procedural', 'hunyuan', 'triposr'] = 'neural'
    quality: Literal['draft', 'standard', 'high'] = 'standard'
    style: Literal['industrial', 'minimal', 'playful', 'realistic'] = 'industrial'

    @model_validator(mode='after')
    def input_required(self):
        self.prompt = self.prompt.strip()
        if not self.prompt and self.image_id:
            self.prompt = '根据参考图片生成 3D 模型，保留主体外观、比例和材质。'
        if len(self.prompt) < 2:
            raise ValueError('请描述要生成的物体，或上传参考图片')
        return self

    @field_validator('parent_id', 'image_id')
    @classmethod
    def valid_id(cls, value):
        if value is not None and (len(value) != 32 or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Invalid ID')
        return value
