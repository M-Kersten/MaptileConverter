/* ---------------------------------------------------------------------------
   TerreinLab: the 3D view of a built model.

   WebGL2 and nothing else, like the rest of the page. It reads the one file
   blender/web_export.py writes, a binary glTF of the scene as built, and the
   textures beside the model, and draws them lit by a low sun with shadows,
   under a sky that fades into haze past the edge of the area.

   It reads of glTF what that file holds: meshes with positions, byte normals
   (KHR_mesh_quantization), texture coordinates and an _OBJECT number per
   vertex; one node per layer; trees as GPU instances
   (EXT_mesh_gpu_instancing); materials with a colour or a texture, a
   roughness, and cut-out alpha.

   The axes are glTF's: X east, Y up, Z south. Clicking reads the object and
   the point under the cursor from an off-screen pass that draws object
   numbers and distances instead of colours, one pixel's worth.
--------------------------------------------------------------------------- */

const GLB_MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

// Attribute locations, the same in every program.
const LOC = { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2, _OBJECT: 3, MATRIX: 4 };

// The sun: from the south-east, 40 degrees up. The aerial photo had its own
// shadows taken out, so any sun agrees with it. This one stands to the side
// of the opening view, from the south-west, so that view sees the shadows
// fall beside the buildings rather than hidden behind them, and the south
// and west walls apart.
const SUN_AZIMUTH = (140 * Math.PI) / 180;
const SUN_ELEVATION = (40 * Math.PI) / 180;

// Linear light. Flat ground in full sun comes out at about 1, so the aerial
// photo keeps its own brightness; a wall turned away gets the sky alone.
const SUN_COLOR = [0.9, 0.86, 0.79];
const SKY_LIGHT = [0.37, 0.41, 0.47];
const GROUND_LIGHT = [0.19, 0.18, 0.16];
const ZENITH = [0.12, 0.27, 0.55];
const HORIZON = [0.62, 0.71, 0.8];
const BASE_COLOR = [0.15, 0.16, 0.14, 1];
// Below the horizon the sky is the land around the area, so a glimpse of it
// past an edge reads as more of that land.
const BELOW = [0.14, 0.15, 0.13];
const PLACEHOLDER = [150, 150, 140, 255];
const SELECTED = [1.0, 0.36, 0.02];

const MIN_PITCH = (4 * Math.PI) / 180;
const MAX_PITCH = Math.PI / 2;
const MIN_DISTANCE = 2;
const FOV = (45 * Math.PI) / 180;
// Larger textures are scaled down to this: an aerial at full resolution over
// two kilometres would not fit most graphics cards.
const TEXTURE_LIMIT = 8192;
const SHADOW_SIZE = 4096;

/* ---------- vectors and matrices, column-major as WebGL takes them ---------- */

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const mix = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function identity() {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f;
  m[10] = (far + near) / (near - far); m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

function orthographic(l, r, b, t, n, f) {
  const m = new Float32Array(16);
  m[0] = 2 / (r - l); m[5] = 2 / (t - b); m[10] = -2 / (f - n);
  m[12] = -(r + l) / (r - l); m[13] = -(t + b) / (t - b); m[14] = -(f + n) / (f - n); m[15] = 1;
  return m;
}

/* A view matrix from a camera's own axes, so straight down is no special case. */
function viewMatrix(eye, right, up, forward) {
  const m = new Float32Array(16);
  m[0] = right[0]; m[4] = right[1]; m[8] = right[2]; m[12] = -dot(right, eye);
  m[1] = up[0]; m[5] = up[1]; m[9] = up[2]; m[13] = -dot(up, eye);
  m[2] = -forward[0]; m[6] = -forward[1]; m[10] = -forward[2]; m[14] = dot(forward, eye);
  m[15] = 1;
  return m;
}

function lookAt(eye, target, up) {
  const forward = norm(sub(target, eye));
  const right = norm(cross(forward, up));
  return viewMatrix(eye, right, cross(right, forward), forward);
}

function fromTRS(t, q, s) {
  const [x, y, z, w] = q, m = new Float32Array(16);
  const xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
  m[0] = (1 - 2 * (yy + zz)) * s[0]; m[1] = 2 * (xy + wz) * s[0]; m[2] = 2 * (xz - wy) * s[0];
  m[4] = 2 * (xy - wz) * s[1]; m[5] = (1 - 2 * (xx + zz)) * s[1]; m[6] = 2 * (yz + wx) * s[1];
  m[8] = 2 * (xz + wy) * s[2]; m[9] = 2 * (yz - wx) * s[2]; m[10] = (1 - 2 * (xx + yy)) * s[2];
  m[12] = t[0]; m[13] = t[1]; m[14] = t[2]; m[15] = 1;
  return m;
}

function nodeMatrix(node) {
  if (node.matrix) return new Float32Array(node.matrix);
  return fromTRS(node.translation || [0, 0, 0], node.rotation || [0, 0, 0, 1], node.scale || [1, 1, 1]);
}

function transformPoint(m, p) {
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15] || 1;
  return [
    (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w,
    (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w,
    (m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]) / w
  ];
}

function invert(m) {
  const a = Array.from(m), inv = new Float32Array(16);
  inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
  inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
  inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
  inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
  inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
  inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
  inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
  inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
  inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
  inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
  inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
  inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
  inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
  inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
  inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
  inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
  const det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
  if (!det) return identity();
  for (let i = 0; i < 16; i++) inv[i] /= det;
  return inv;
}

/* ---------- shaders ---------- */

const VERTEX = `#version 300 es
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in float a_object;
layout(location = 4) in vec4 a_m0;
layout(location = 5) in vec4 a_m1;
layout(location = 6) in vec4 a_m2;
layout(location = 7) in vec4 a_m3;
uniform mat4 u_viewProj;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
flat out float v_object;
void main() {
  mat4 model = mat4(a_m0, a_m1, a_m2, a_m3);
  vec4 world = model * vec4(a_position, 1.0);
  v_world = world.xyz;
  // The inverse transpose keeps a normal square to its face when an
  // instance is stretched more one way than another, as every tree is.
  v_normal = transpose(inverse(mat3(model))) * a_normal;
  v_uv = a_uv;
  v_object = a_object;
  gl_Position = u_viewProj * world;
}`;

const COMMON = `
vec3 srgb(vec3 c) {
  c = max(c, vec3(0.0));
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}
vec3 skyColor(vec3 d, vec3 horizon, vec3 zenith, vec3 below) {
  if (d.y < 0.0) return mix(horizon, below, clamp(-d.y * 4.0, 0.0, 1.0));
  return mix(horizon, zenith, pow(clamp(d.y, 0.0, 1.0), 0.55));
}`;

const SHADE = `#version 300 es
precision highp float;
precision highp sampler2DShadow;
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
flat in float v_object;
uniform sampler2D u_texture;
uniform sampler2DShadow u_shadowMap;
uniform float u_hasTexture;
uniform vec4 u_color;
uniform float u_cutoff;
uniform float u_roughness;
uniform vec3 u_eye;
uniform vec3 u_sunDirection;
uniform vec3 u_sunColor;
uniform vec3 u_skyLight;
uniform vec3 u_groundLight;
uniform vec3 u_horizon;
uniform vec3 u_zenith;
uniform vec3 u_below;
uniform float u_fogStart;
uniform float u_fogRange;
uniform mat4 u_lightViewProj;
uniform float u_shadowTexel;
uniform float u_shadowOffset;
uniform float u_shadowBias;
uniform float u_shadows;
uniform float u_selected;
uniform vec3 u_highlight;
uniform float u_plain;
out vec4 o_color;
${COMMON}
float sunlight(vec3 n) {
  if (u_shadows < 0.5) return 1.0;
  vec4 p = u_lightViewProj * vec4(v_world + n * u_shadowOffset, 1.0);
  vec3 q = p.xyz / p.w * 0.5 + 0.5;
  if (any(lessThan(q, vec3(0.0))) || any(greaterThan(q, vec3(1.0)))) return 1.0;
  float lit = 0.0;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      lit += texture(u_shadowMap, vec3(q.xy + vec2(x, y) * u_shadowTexel, q.z - u_shadowBias));
    }
  }
  return lit / 9.0;
}
void main() {
  vec4 base = u_color;
  if (u_hasTexture > 0.5) base *= texture(u_texture, v_uv);
  if (base.a < u_cutoff) discard;
  // Wireframe mode: plain faces, so only the edges catch the eye.
  if (u_plain > 0.5) base.rgb = vec3(0.6);
  vec3 n = normalize(v_normal);
  if (!gl_FrontFacing) n = -n;
  vec3 toEye = normalize(u_eye - v_world);
  float facing = dot(n, u_sunDirection);
  float lit = facing > 0.0 ? sunlight(n) : 0.0;
  vec3 ambient = mix(u_groundLight, u_skyLight, n.y * 0.5 + 0.5);
  vec3 color = base.rgb * (ambient + u_sunColor * max(facing, 0.0) * lit);
  // Water and anything else this smooth mirrors the sky at a low angle.
  float gloss = u_plain > 0.5 ? 0.0 : clamp(1.0 - u_roughness / 0.3, 0.0, 1.0);
  if (gloss > 0.0) {
    float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, toEye), 0.0), 5.0);
    color = mix(color, skyColor(reflect(-toEye, n), u_horizon, u_zenith, u_below), fresnel * gloss);
    vec3 h = normalize(toEye + u_sunDirection);
    color += u_sunColor * pow(max(dot(n, h), 0.0), 300.0) * 3.0 * gloss * lit;
  }
  if (abs(v_object - u_selected) < 0.5) color = mix(color, u_highlight, 0.45);
  float fog = clamp((length(u_eye - v_world) - u_fogStart) / u_fogRange, 0.0, 1.0);
  color = mix(color, u_horizon, fog * fog * 0.9);
  o_color = vec4(srgb(color), 1.0);
}`;

// Object number and distance, each as three bytes: exact to sixteen million
// objects, and to a fraction of a millimetre over the view's depth.
const PICK = `#version 300 es
precision highp float;
in vec3 v_world;
in vec2 v_uv;
flat in float v_object;
uniform sampler2D u_texture;
uniform float u_hasTexture;
uniform vec4 u_color;
uniform float u_cutoff;
uniform vec3 u_eye;
uniform float u_far;
layout(location = 0) out vec4 o_object;
layout(location = 1) out vec4 o_distance;
vec4 bytes(float v) {
  float high = floor(v / 65536.0);
  float middle = floor((v - high * 65536.0) / 256.0);
  float low = v - high * 65536.0 - middle * 256.0;
  return vec4(high, middle, low, 255.0) / 255.0;
}
void main() {
  if (u_cutoff > 0.0) {
    float alpha = u_color.a * (u_hasTexture > 0.5 ? texture(u_texture, v_uv).a : 1.0);
    if (alpha < u_cutoff) discard;
  }
  o_object = bytes(v_object + 1.0);
  o_distance = bytes(floor(clamp(length(v_world - u_eye) / u_far, 0.0, 1.0) * 16777215.0));
}`;

const SHADOW = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_texture;
uniform float u_hasTexture;
uniform vec4 u_color;
uniform float u_cutoff;
void main() {
  if (u_cutoff > 0.0) {
    float alpha = u_color.a * (u_hasTexture > 0.5 ? texture(u_texture, v_uv).a : 1.0);
    if (alpha < u_cutoff) discard;
  }
}`;

// The wireframe's lines, over the model. Dark where the surface under them
// is light and light where it is dark, so they read on the aerial, on black
// asphalt and on pale walls alike; the selected object's in its own colour.
const WIRE = `#version 300 es
precision highp float;
in vec3 v_world;
in vec2 v_uv;
flat in float v_object;
uniform sampler2D u_texture;
uniform float u_hasTexture;
uniform vec4 u_color;
uniform vec3 u_eye;
uniform float u_fogStart;
uniform float u_fogRange;
uniform float u_selected;
uniform vec3 u_highlight;
uniform float u_focal;
uniform float u_plain;
out vec4 o_color;
${COMMON}
void main() {
  vec4 base = u_color;
  if (u_hasTexture > 0.5) base *= texture(u_texture, v_uv);
  float light = dot(base.rgb, vec3(0.2126, 0.7152, 0.0722));
  if (u_plain > 0.5) light = 1.0;
  vec3 color = light > 0.12 ? vec3(0.0) : vec3(1.0);
  float alpha = light > 0.12 ? 0.55 : 0.45;
  // Full where a metre is a few pixels or more, faint where edges a few
  // metres long crowd into a pixel or two and would only blacken the model.
  float perMetre = u_focal / max(length(u_eye - v_world), 0.001);
  alpha *= mix(0.12, 1.0, smoothstep(0.6, 3.0, perMetre));
  if (abs(v_object - u_selected) < 0.5) { color = u_highlight; alpha = 1.0; }
  float fog = clamp((length(u_eye - v_world) - u_fogStart) / u_fogRange, 0.0, 1.0);
  o_color = vec4(srgb(color), alpha * (1.0 - fog));
}`;

const SKY_VERTEX = `#version 300 es
out vec2 v_ndc;
void main() {
  vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  v_ndc = p;
  gl_Position = vec4(p, 0.0, 1.0);
}`;

const SKY = `#version 300 es
precision highp float;
in vec2 v_ndc;
uniform mat4 u_inverseViewProj;
uniform vec3 u_eye;
uniform vec3 u_horizon;
uniform vec3 u_zenith;
uniform vec3 u_below;
uniform vec3 u_sunDirection;
out vec4 o_color;
${COMMON}
void main() {
  vec4 far = u_inverseViewProj * vec4(v_ndc, 1.0, 1.0);
  vec3 d = normalize(far.xyz / far.w - u_eye);
  vec3 c = skyColor(d, u_horizon, u_zenith, u_below);
  c += vec3(1.0, 0.9, 0.72) * pow(max(dot(d, u_sunDirection), 0.0), 48.0) * 0.35;
  o_color = vec4(srgb(c), 1.0);
}`;

/* ---------- glTF ---------- */

const COMPONENT = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
const WIDTH = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const NORMALIZE = { 5120: 127, 5121: 255, 5122: 32767, 5123: 65535 };

function parseGlb(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 20 || view.getUint32(0, true) !== GLB_MAGIC) {
    throw new Error("dit is geen glTF-bestand");
  }
  const length = Math.min(view.getUint32(8, true), buffer.byteLength);
  let offset = 12, json = null, bin = null;
  while (offset + 8 <= length) {
    const size = view.getUint32(offset, true), type = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (type === CHUNK_JSON) json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, start, size)));
    else if (type === CHUNK_BIN && !bin) bin = new Uint8Array(buffer, start, size);
    offset = start + size;
  }
  if (!json) throw new Error("het bestand heeft geen inhoudsopgave");
  return { json, bin: bin || new Uint8Array(0) };
}

/* An accessor's values as floats, for what is worked out here rather than on
   the graphics card: instance transforms and bounds. */
function readAccessor(gltf, bin, index) {
  const accessor = gltf.accessors[index];
  const width = WIDTH[accessor.type], Type = COMPONENT[accessor.componentType];
  const out = new Float32Array(accessor.count * width);
  if (accessor.bufferView === undefined) return out;
  const view = gltf.bufferViews[accessor.bufferView];
  const element = Type.BYTES_PER_ELEMENT * width;
  const stride = view.byteStride || element;
  const start = bin.byteOffset + (view.byteOffset || 0) + (accessor.byteOffset || 0);
  if (stride === element && start % Type.BYTES_PER_ELEMENT === 0) {
    out.set(new Type(bin.buffer, start, accessor.count * width));
  } else {
    const data = new DataView(bin.buffer);
    const read = {
      5120: o => data.getInt8(o), 5121: o => data.getUint8(o), 5122: o => data.getInt16(o, true),
      5123: o => data.getUint16(o, true), 5125: o => data.getUint32(o, true), 5126: o => data.getFloat32(o, true)
    }[accessor.componentType];
    for (let i = 0; i < accessor.count; i++) {
      for (let k = 0; k < width; k++) out[i * width + k] = read(start + i * stride + k * Type.BYTES_PER_ELEMENT);
    }
  }
  const range = accessor.normalized && NORMALIZE[accessor.componentType];
  if (range) for (let i = 0; i < out.length; i++) out[i] = Math.max(out[i] / range, -1);
  return out;
}

/* An accessor of whole numbers, indices or a mask, as a copy in its own
   type: as floats, vertex numbers past sixteen million would round. */
function readIntegers(gltf, bin, index) {
  const accessor = gltf.accessors[index];
  const view = gltf.bufferViews[accessor.bufferView], Type = COMPONENT[accessor.componentType];
  const width = WIDTH[accessor.type], start = bin.byteOffset + (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const packed = !view.byteStride || view.byteStride === Type.BYTES_PER_ELEMENT * width;
  if (Type !== Float32Array && packed && start % Type.BYTES_PER_ELEMENT === 0) {
    return new Type(bin.buffer, start, accessor.count * width).slice();
  }
  return Uint32Array.from(readAccessor(gltf, bin, index));
}

async function fetchBytes(url, signal, onProgress) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`het model kwam niet binnen (${response.status})`);
  const total = Number(response.headers.get("Content-Length")) || 0;
  if (!response.body || !total) return response.arrayBuffer();
  const reader = response.body.getReader();
  let bytes = new Uint8Array(total), loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > bytes.length) {
      const grown = new Uint8Array(Math.max(bytes.length * 2, loaded + value.length));
      grown.set(bytes.subarray(0, loaded));
      bytes = grown;
    }
    bytes.set(value, loaded);
    loaded += value.length;
    onProgress(loaded, total);
  }
  return loaded === bytes.length ? bytes.buffer : bytes.slice(0, loaded).buffer;
}

/* ---------- the viewer ---------- */

export function webgl2Supported() {
  try {
    return !!document.createElement("canvas").getContext("webgl2");
  } catch {
    return false;
  }
}

export class Viewer {
  /* `on` takes progress(stage, done, total), hover(info | null),
     select(info | null), view({ yaw, pitch, distance }) and lost(). */
  constructor(canvas, on = {}) {
    this.canvas = canvas;
    this.on = on;
    const gl = canvas.getContext("webgl2", { antialias: true, alpha: false, powerPreference: "high-performance" });
    if (!gl) throw new Error("deze browser kan geen WebGL2 tekenen");
    this.gl = gl;
    this.abort = new AbortController();
    this.draws = [];
    this.layers = [];
    this.objects = [];
    this.textures = [];
    this.buffers = [];
    this.vaos = [];
    this.selected = -1;
    this.origin = null;
    this.frame = 0;
    this.shadowDirty = true;
    this.animation = null;
    this.camera = { target: [0, 0, 0], distance: 500, yaw: 0, pitch: 0.6 };
    this.bounds = { min: [-250, 0, -250], max: [250, 50, 250] };
    this.anisotropy = gl.getExtension("EXT_texture_filter_anisotropic");

    this.programs = {
      shade: this.program(VERTEX, SHADE),
      pick: this.program(VERTEX, PICK),
      shadow: this.program(VERTEX, SHADOW),
      sky: this.program(SKY_VERTEX, SKY),
      wire: this.program(VERTEX, WIRE)
    };
    this.emptyVao = gl.createVertexArray();
    // Texture unit 0 holds pictures, unit 1 the shadow map, and nothing
    // else ever: WebGL2 refuses to draw with a depth texture where a picture
    // is sampled, or with the map it is drawing into bound at all.
    this.placeholder = this.texture1x1(PLACEHOLDER);
    this.noShadow = this.depthTexture(1);

    this.lost = e => { e.preventDefault(); this.dead = true; (this.on.lost || (() => {}))(); };
    canvas.addEventListener("webglcontextlost", this.lost);
    this.resizer = new ResizeObserver(() => this.redraw());
    this.resizer.observe(canvas);
    this.listen();
  }

  /* ---------- setup ---------- */

  program(vertexSource, fragmentSource) {
    const gl = this.gl;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS) && !gl.isContextLost()) {
        throw new Error("shader: " + gl.getShaderInfoLog(shader));
      }
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS) && !gl.isContextLost()) {
      throw new Error("shader: " + gl.getProgramInfoLog(program));
    }
    const uniforms = {};
    const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) || 0;
    for (let i = 0; i < count; i++) {
      const info = gl.getActiveUniform(program, i);
      uniforms[info.name] = gl.getUniformLocation(program, info.name);
    }
    return { program, uniforms };
  }

  texture1x1(rgba) {
    const gl = this.gl, texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.SRGB8_ALPHA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(rgba));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    return texture;
  }

  /* A depth texture read the way a shadow map is: compared, not looked up. */
  depthTexture(size) {
    const gl = this.gl, texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.DEPTH_COMPONENT24, size, size);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL);
    gl.activeTexture(gl.TEXTURE0);
    return texture;
  }

  /* ---------- loading ---------- */

  /* Load a model: `url` is the .glb, `textureBase` where its textures are,
     `origin` the RD and NAP of its local zero ({ x, y, z }), and `extent`
     the area its ground covers, in the model's own X and Z
     ({ minX, maxX, minZ, maxZ }). Without it the ground layer's bounds
     stand in, which reach past the edge wherever a building does. */
  async load(url, { textureBase = "", origin = null, extent = null } = {}) {
    const progress = this.on.progress || (() => {});
    this.origin = origin;
    this.extent = extent;
    const buffer = await fetchBytes(url, this.abort.signal, (done, total) => progress("download", done, total));
    if (this.dead) return null;
    progress("build", 0, 1);
    const { json, bin } = parseGlb(buffer);
    this.build(json, bin);
    this.overview(false);
    progress("build", 1, 1);
    this.redraw();
    // Geometry first, so the model is there to look at while the photos
    // come in; they replace a grey placeholder one by one.
    this.loadTextures(json, textureBase, progress);
    return this.summary;
  }

  build(gltf, bin) {
    const gl = this.gl;
    const extras = (gltf.extras && gltf.extras.terreinlab) || {};
    this.objects = extras.objects || [];
    const scene = gltf.scenes[gltf.scene || 0] || { nodes: [] };
    const roots = scene.nodes || [];
    this.layers = roots.map((nodeIndex, i) => {
      const described = (extras.layers || [])[i] || {};
      return {
        index: i,
        name: described.name || gltf.nodes[nodeIndex].name || `Laag ${i + 1}`,
        objects: described.objects || 0,
        triangles: described.triangles || 0,
        visible: true
      };
    });

    this.materials = (gltf.materials || []).map(m => {
      const pbr = m.pbrMetallicRoughness || {};
      const texture = pbr.baseColorTexture ? pbr.baseColorTexture.index : null;
      return {
        name: m.name || "",
        color: pbr.baseColorFactor || [1, 1, 1, 1],
        texture,
        roughness: pbr.roughnessFactor !== undefined ? pbr.roughnessFactor : 1,
        cutoff: m.alphaMode === "MASK" ? (m.alphaCutoff !== undefined ? m.alphaCutoff : 0.5) : 0
      };
    });
    if (!this.materials.length) this.materials.push({ name: "", color: [0.8, 0.8, 0.8, 1], texture: null, roughness: 1, cutoff: 0 });
    this.textures = (gltf.textures || []).map(t => ({ image: t.source, gl: null }));

    const viewBuffers = new Map();
    const bufferFor = (viewIndex, target) => {
      const key = `${viewIndex}/${target}`;
      if (!viewBuffers.has(key)) {
        const view = gltf.bufferViews[viewIndex];
        const buffer = gl.createBuffer();
        gl.bindBuffer(target, buffer);
        gl.bufferData(target, new Uint8Array(bin.buffer, bin.byteOffset + (view.byteOffset || 0), view.byteLength), gl.STATIC_DRAW);
        viewBuffers.set(key, buffer);
        this.buffers.push(buffer);
      }
      return viewBuffers.get(key);
    };

    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    const ground = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
    let growingGround = false;
    const grow = p => {
      for (let k = 0; k < 3; k++) {
        min[k] = Math.min(min[k], p[k]); max[k] = Math.max(max[k], p[k]);
        if (growingGround) { ground.min[k] = Math.min(ground.min[k], p[k]); ground.max[k] = Math.max(ground.max[k], p[k]); }
      }
    };
    const corners = (lo, hi) => [0, 1, 2, 3, 4, 5, 6, 7].map(i => [i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]]);
    let triangles = 0;
    this.groundSample = [];

    const visit = (nodeIndex, parentMatrix, layer) => {
      const node = gltf.nodes[nodeIndex];
      const matrix = multiply(parentMatrix, nodeMatrix(node));
      if (node.mesh !== undefined) {
        const instancing = node.extensions && node.extensions.EXT_mesh_gpu_instancing;
        let instances = null, instanceObjects = null;
        if (instancing) {
          const a = instancing.attributes;
          const count = gltf.accessors[a.TRANSLATION ?? a.ROTATION ?? a.SCALE].count;
          const t = a.TRANSLATION !== undefined ? readAccessor(gltf, bin, a.TRANSLATION) : null;
          const r = a.ROTATION !== undefined ? readAccessor(gltf, bin, a.ROTATION) : null;
          const s = a.SCALE !== undefined ? readAccessor(gltf, bin, a.SCALE) : null;
          instances = new Float32Array(count * 16);
          for (let i = 0; i < count; i++) {
            const local = fromTRS(
              t ? [t[i * 3], t[i * 3 + 1], t[i * 3 + 2]] : [0, 0, 0],
              r ? [r[i * 4], r[i * 4 + 1], r[i * 4 + 2], r[i * 4 + 3]] : [0, 0, 0, 1],
              s ? [s[i * 3], s[i * 3 + 1], s[i * 3 + 2]] : [1, 1, 1]
            );
            instances.set(multiply(matrix, local), i * 16);
          }
          if (a._OBJECT !== undefined) instanceObjects = readAccessor(gltf, bin, a._OBJECT);
        }
        for (const primitive of gltf.meshes[node.mesh].primitives) {
          const draw = this.primitive(gltf, bin, primitive, bufferFor, matrix, instances, instanceObjects);
          if (!draw) continue;
          draw.layer = layer;
          this.draws.push(draw);
          triangles += (draw.count / 3) * Math.max(1, draw.instances);
          const position = gltf.accessors[primitive.attributes.POSITION];
          growingGround = layer === 0;
          if (position.min && position.max) {
            const box = corners(position.min, position.max);
            if (instances) {
              for (let i = 0; i < draw.instances; i++) {
                const m = instances.subarray(i * 16, i * 16 + 16);
                box.forEach(c => grow(transformPoint(m, c)));
              }
            } else {
              box.forEach(c => grow(transformPoint(matrix, c)));
            }
          }
          if (layer === 0 && !instances && this.groundSample.length < 4000) {
            // A sample of the first layer's heights, the ground in every
            // model built here, for the plane that carries on past its edge.
            const points = readAccessor(gltf, bin, primitive.attributes.POSITION);
            const step = Math.max(1, Math.floor(points.length / 3 / 1500));
            for (let i = 0; i < points.length / 3; i += step) this.groundSample.push(transformPoint(matrix, [points[i * 3], points[i * 3 + 1], points[i * 3 + 2]])[1]);
          }
        }
      }
      for (const child of node.children || []) visit(child, matrix, layer);
    };
    roots.forEach((nodeIndex, layer) => visit(nodeIndex, identity(), layer));

    if (!isFinite(min[0])) { min.splice(0, 3, -50, 0, -50); max.splice(0, 3, 50, 10, 50); }
    this.bounds = { min, max };
    this.groundLevel = this.estimateGround();
    const e = this.extent;
    this.groundExtent = e && [e.minX, e.maxX, e.minZ, e.maxZ].every(Number.isFinite)
      ? e
      : isFinite(ground.min[0])
        ? { minX: ground.min[0], maxX: ground.max[0], minZ: ground.min[2], maxZ: ground.max[2] }
        : { minX: min[0], maxX: max[0], minZ: min[2], maxZ: max[2] };
    this.buildBase();
    // Opaque first, then the cut-outs; within each, by texture, so one stays
    // bound for as long as it can.
    this.draws.sort((a, b) =>
      (this.materials[a.material].cutoff > 0) - (this.materials[b.material].cutoff > 0) ||
      (this.materials[a.material].texture ?? -1) - (this.materials[b.material].texture ?? -1));
    this.summary = {
      triangles: Math.round(triangles),
      objects: this.objects.length,
      layers: this.layers.map(l => ({ ...l })),
      size: [max[0] - min[0], max[2] - min[2], max[1] - min[1]]
    };
    this.shadowDirty = true;
  }

  primitive(gltf, bin, primitive, bufferFor, matrix, instances, instanceObjects) {
    const gl = this.gl;
    if ((primitive.mode ?? 4) !== 4 || primitive.attributes.POSITION === undefined) return null;
    // What each attribute reads, kept so the wireframe's lines can read the
    // same buffers through a vertex array of their own.
    const bindings = [], has = {};
    for (const name of ["POSITION", "NORMAL", "TEXCOORD_0", "_OBJECT"]) {
      const index = primitive.attributes[name];
      if (index === undefined) continue;
      const accessor = gltf.accessors[index], view = gltf.bufferViews[accessor.bufferView];
      bindings.push({
        loc: LOC[name], buffer: bufferFor(accessor.bufferView, gl.ARRAY_BUFFER), size: WIDTH[accessor.type],
        type: accessor.componentType, normalized: !!accessor.normalized,
        stride: view.byteStride || 0, offset: accessor.byteOffset || 0, divisor: 0
      });
      has[name] = true;
    }
    let count = gltf.accessors[primitive.attributes.POSITION].count, indexType = null, offset = 0;
    let elements = null, wire = null;
    if (primitive.indices !== undefined) {
      const accessor = gltf.accessors[primitive.indices];
      elements = bufferFor(accessor.bufferView, gl.ELEMENT_ARRAY_BUFFER);
      count = accessor.count;
      indexType = accessor.componentType;
      offset = accessor.byteOffset || 0;
      // The triangles and which of their sides are edges, copied out of
      // the file so it can go; the lines are made from them when asked for.
      const edges = primitive.extras && primitive.extras.edges;
      wire = {
        indices: readIntegers(gltf, bin, primitive.indices),
        mask: edges !== undefined ? readIntegers(gltf, bin, edges) : null
      };
    }
    let n = 0;
    if (instances) {
      n = instances.length / 16;
      const buffer = gl.createBuffer();
      this.buffers.push(buffer);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, instances, gl.STATIC_DRAW);
      for (let c = 0; c < 4; c++) {
        bindings.push({ loc: LOC.MATRIX + c, buffer, size: 4, type: gl.FLOAT, normalized: false,
          stride: 64, offset: c * 16, divisor: 1 });
      }
      if (instanceObjects) {
        const ids = gl.createBuffer();
        this.buffers.push(ids);
        gl.bindBuffer(gl.ARRAY_BUFFER, ids);
        gl.bufferData(gl.ARRAY_BUFFER, instanceObjects, gl.STATIC_DRAW);
        bindings.push({ loc: LOC._OBJECT, buffer: ids, size: 1, type: gl.FLOAT, normalized: false,
          stride: 0, offset: 0, divisor: 1 });
        has._OBJECT = true;
      }
    }
    return {
      vao: this.vertexArray(bindings, elements), count, indexType, offset, instances: n, has, matrix,
      bindings, wire, material: primitive.material !== undefined ? primitive.material : 0
    };
  }

  vertexArray(bindings, elements) {
    const gl = this.gl, vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    for (const b of bindings) {
      gl.bindBuffer(gl.ARRAY_BUFFER, b.buffer);
      gl.enableVertexAttribArray(b.loc);
      gl.vertexAttribPointer(b.loc, b.size, b.type, b.normalized, b.stride, b.offset);
      gl.vertexAttribDivisor(b.loc, b.divisor);
    }
    if (elements) gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, elements);
    gl.bindVertexArray(null);
    this.vaos.push(vao);
    return vao;
  }

  /* The wireframe of a draw: a line for every side of a triangle that is an
     edge of the mesh, so a quad shows as a quad, as in Blender. A file with
     no record of that shows every side. Made once, the first time. */
  wireOf(d) {
    if (d.wireLines !== undefined) return d.wireLines;
    d.wireLines = null;
    if (!d.wire) return null;
    const { indices, mask } = d.wire, triangles = Math.floor(indices.length / 3);
    const lines = new Uint32Array(triangles * 6);
    let n = 0;
    for (let t = 0; t < triangles; t++) {
      const sides = mask ? mask[t] : 7, i = t * 3;
      if (sides & 1) { lines[n++] = indices[i]; lines[n++] = indices[i + 1]; }
      if (sides & 2) { lines[n++] = indices[i + 1]; lines[n++] = indices[i + 2]; }
      if (sides & 4) { lines[n++] = indices[i + 2]; lines[n++] = indices[i]; }
    }
    d.wire = null;
    if (!n) return null;
    const gl = this.gl, elements = gl.createBuffer();
    this.buffers.push(elements);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, elements);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, lines.subarray(0, n), gl.STATIC_DRAW);
    d.wireLines = { vao: this.vertexArray(d.bindings, elements), count: n };
    return d.wireLines;
  }

  /* The ground's usual height: low, but not the bed of a canal. */
  estimateGround() {
    const sample = this.groundSample.slice().sort((a, b) => a - b);
    if (!sample.length) return this.bounds.min[1];
    return sample[Math.floor(sample.length * 0.1)];
  }

  /* A plain ring around the area, a little under its ground, so a low view
     has land to the horizon rather than an edge over nothing. It reaches
     two metres in under the ground's edge, so no gap opens between the two,
     and no further: under the model it would show through every tunnel. */
  buildBase() {
    const gl = this.gl, g = this.groundExtent;
    const y = this.groundLevel - 0.6;
    const reach = Math.max(g.maxX - g.minX, g.maxZ - g.minZ) * 25;
    const x0 = g.minX + 2, x1 = g.maxX - 2, z0 = g.minZ + 2, z1 = g.maxZ - 2;
    const X0 = x0 - reach, X1 = x1 + reach, Z0 = z0 - reach, Z1 = z1 + reach;
    const quads = [
      [X0, Z0, X1, z0], [X0, z1, X1, Z1], [X0, z0, x0, z1], [x1, z0, X1, z1]
    ];
    const positions = [], normals = [];
    for (const [a, b, c, d] of quads) {
      // Counter-clockwise seen from above, so the faces look up: north-west,
      // south-west, south-east, then north-west, south-east, north-east.
      positions.push(a, y, b, a, y, d, c, y, d, a, y, b, c, y, d, c, y, b);
      for (let i = 0; i < 6; i++) normals.push(0, 127, 0);
    }
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const p = gl.createBuffer(), n = gl.createBuffer();
    this.buffers.push(p, n);
    gl.bindBuffer(gl.ARRAY_BUFFER, p);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(LOC.POSITION);
    gl.vertexAttribPointer(LOC.POSITION, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, n);
    gl.bufferData(gl.ARRAY_BUFFER, new Int8Array(normals), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(LOC.NORMAL);
    gl.vertexAttribPointer(LOC.NORMAL, 3, gl.BYTE, true, 0, 0);
    gl.bindVertexArray(null);
    this.vaos.push(vao);
    this.base = { vao, count: positions.length / 3 };
  }

  async loadTextures(gltf, base, progress) {
    const gl = this.gl;
    const wanted = new Set(this.materials.map(m => m.texture).filter(t => t !== null && t !== undefined));
    const images = gltf.images || [];
    const jobs = [...wanted].map(index => ({ index, uri: images[this.textures[index].image]?.uri }))
      .filter(job => job.uri);
    // The aerial first: it is most of what is on screen.
    jobs.sort((a, b) => (/aerial/i.test(b.uri) ? 1 : 0) - (/aerial/i.test(a.uri) ? 1 : 0));
    let done = 0;
    const total = jobs.length;
    progress("textures", 0, total);
    const limit = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE), TEXTURE_LIMIT);
    const worker = async () => {
      while (jobs.length && !this.dead) {
        const job = jobs.shift();
        try {
          const response = await fetch(base + job.uri, { signal: this.abort.signal });
          if (!response.ok) throw new Error(`${response.status}`);
          const image = await decodeImage(await response.blob(), limit);
          if (this.dead) { image.close?.(); return; }
          this.textures[job.index].gl = this.upload(image);
          image.close?.();
          this.redraw();
        } catch (error) {
          if (error.name === "AbortError") return;
          console.warn(`texture ${job.uri}:`, error);
        }
        progress("textures", ++done, total);
      }
    };
    await Promise.all([worker(), worker(), worker(), worker()]);
    progress("done", 1, 1);
  }

  upload(image) {
    const gl = this.gl, texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.SRGB8_ALPHA8, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    if (this.anisotropy) {
      // The ground is mostly seen at a glancing angle, where this is the
      // difference between a photo and a smear.
      const most = gl.getParameter(this.anisotropy.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
      gl.texParameterf(gl.TEXTURE_2D, this.anisotropy.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, most));
    }
    return texture;
  }

  /* ---------- the camera ---------- */

  basis(camera = this.camera) {
    const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
    const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
    const back = [cp * sy, sp, cp * cy];
    const forward = [-back[0], -back[1], -back[2]];
    const right = [cy, 0, -sy];
    return { eye: add(camera.target, scale(back, camera.distance)), forward, right, up: cross(right, forward) };
  }

  get radius() {
    const { min, max } = this.bounds;
    return Math.max(10, Math.hypot(max[0] - min[0], max[2] - min[2]) / 2);
  }

  get center() {
    const { min, max } = this.bounds;
    return [(min[0] + max[0]) / 2, this.groundLevel ?? min[1], (min[2] + max[2]) / 2];
  }

  matrices() {
    const { width, height } = this.canvas;
    const b = this.basis();
    const near = clamp(this.camera.distance * 0.02, 0.05, 20);
    const far = Math.max(this.camera.distance * 4, this.radius * 8);
    const projection = perspective(FOV, width / Math.max(1, height), near, far);
    const view = viewMatrix(b.eye, b.right, b.up, b.forward);
    return { ...b, near, far, viewProj: multiply(projection, view), aspect: width / Math.max(1, height) };
  }

  /* The ray from the eye through a point of the canvas, in CSS pixels. */
  ray(x, y, m = this.matrices()) {
    const rect = this.canvas.getBoundingClientRect();
    const nx = (x / rect.width) * 2 - 1, ny = 1 - (y / rect.height) * 2;
    const t = Math.tan(FOV / 2);
    return {
      origin: m.eye,
      direction: norm(add(m.forward, add(scale(m.right, nx * t * m.aspect), scale(m.up, ny * t))))
    };
  }

  /* Where a ray through the canvas meets the level `height`, if in front. */
  onLevel(x, y, height, m) {
    const { origin, direction } = this.ray(x, y, m);
    if (Math.abs(direction[1]) < 1e-6) return null;
    const t = (height - origin[1]) / direction[1];
    if (t <= 0 || t > this.radius * 20) return null;
    return add(origin, scale(direction, t));
  }

  clampCamera() {
    const c = this.camera, { min, max } = this.bounds, margin = this.radius * 0.25;
    c.target[0] = clamp(c.target[0], min[0] - margin, max[0] + margin);
    c.target[2] = clamp(c.target[2], min[2] - margin, max[2] + margin);
    c.target[1] = clamp(c.target[1], min[1] - 10, max[1] + 10);
    c.distance = clamp(c.distance, MIN_DISTANCE, this.radius * 6);
    c.pitch = clamp(c.pitch, MIN_PITCH, MAX_PITCH);
  }

  fitDistance(pitch) {
    // The whole area in view, with a little room around it, across the
    // narrower way of the screen: a phone held upright is narrow.
    const width = this.canvas.clientWidth || this.canvas.width;
    const height = this.canvas.clientHeight || this.canvas.height;
    const half = Math.atan(Math.tan(FOV / 2) * Math.min(1, width / Math.max(1, height)));
    const along = Math.max(0.35, Math.sin(pitch));
    return (this.radius / Math.tan(half)) * (0.75 + 0.35 * (1 - along));
  }

  overview(animate = true) {
    const pitch = (36 * Math.PI) / 180;
    this.flyTo({ target: this.center, distance: this.fitDistance(pitch), yaw: (-32 * Math.PI) / 180, pitch }, animate);
  }

  topDown(animate = true) {
    this.flyTo({ target: this.center, distance: this.fitDistance(MAX_PITCH) * 0.95, yaw: 0, pitch: MAX_PITCH }, animate);
  }

  north(animate = true) {
    this.flyTo({ ...this.camera, yaw: 0 }, animate);
  }

  focus(point, animate = true) {
    const distance = clamp(Math.min(this.camera.distance * 0.45, 160), 12, this.radius * 6);
    this.flyTo({ ...this.camera, target: point.slice(), distance, pitch: Math.max(this.camera.pitch, 0.35) }, animate);
  }

  flyTo(goal, animate = true) {
    const from = { ...this.camera, target: this.camera.target.slice() };
    // The short way round.
    let yaw = goal.yaw;
    while (yaw - from.yaw > Math.PI) yaw -= 2 * Math.PI;
    while (yaw - from.yaw < -Math.PI) yaw += 2 * Math.PI;
    const to = { target: goal.target.slice(), distance: goal.distance, yaw, pitch: goal.pitch };
    if (!animate || matchMedia("(prefers-reduced-motion: reduce)").matches) {
      this.animation = null;
      Object.assign(this.camera, to);
      this.clampCamera();
      this.redraw();
      return;
    }
    this.animation = { from, to, start: performance.now(), ms: 650 };
    this.redraw();
  }

  stepAnimation(now) {
    const a = this.animation;
    if (!a) return false;
    const t = clamp((now - a.start) / a.ms, 0, 1);
    const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    const c = this.camera;
    c.target = [0, 1, 2].map(k => mix(a.from.target[k], a.to.target[k], e));
    // Distance eases in proportion, so a long zoom does not rush its end.
    c.distance = Math.exp(mix(Math.log(a.from.distance), Math.log(a.to.distance), e));
    c.yaw = mix(a.from.yaw, a.to.yaw, e);
    c.pitch = mix(a.from.pitch, a.to.pitch, e);
    this.clampCamera();
    if (t >= 1) this.animation = null;
    return true;
  }

  /* ---------- drawing ---------- */

  redraw() {
    if (!this.frame && !this.dead) this.frame = requestAnimationFrame(now => this.draw(now));
  }

  resize() {
    const canvas = this.canvas, ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  draw(now) {
    this.frame = 0;
    if (this.dead || this.gl.isContextLost()) return;
    if (this.stepAnimation(now)) this.redraw();
    this.resize();
    const gl = this.gl;
    if (this.shadowDirty && this.draws.length) this.renderShadows();
    const m = this.matrices();
    this.lastMatrices = m;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    const sky = this.programs.sky;
    gl.useProgram(sky.program);
    gl.uniformMatrix4fv(sky.uniforms.u_inverseViewProj, false, invert(m.viewProj));
    gl.uniform3fv(sky.uniforms.u_eye, m.eye);
    gl.uniform3fv(sky.uniforms.u_horizon, HORIZON);
    gl.uniform3fv(sky.uniforms.u_zenith, ZENITH);
    gl.uniform3fv(sky.uniforms.u_below, BELOW);
    gl.uniform3fv(sky.uniforms.u_sunDirection, this.sunDirection);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.CULL_FACE);
    const shade = this.programs.shade, u = shade.uniforms;
    gl.useProgram(shade.program);
    gl.uniformMatrix4fv(u.u_viewProj, false, m.viewProj);
    gl.uniform3fv(u.u_eye, m.eye);
    gl.uniform3fv(u.u_sunDirection, this.sunDirection);
    gl.uniform3fv(u.u_sunColor, SUN_COLOR);
    gl.uniform3fv(u.u_skyLight, SKY_LIGHT);
    gl.uniform3fv(u.u_groundLight, GROUND_LIGHT);
    gl.uniform3fv(u.u_horizon, HORIZON);
    gl.uniform3fv(u.u_zenith, ZENITH);
    gl.uniform3fv(u.u_below, BELOW);
    gl.uniform3fv(u.u_highlight, SELECTED);
    gl.uniform1f(u.u_fogStart, this.camera.distance + this.radius * 0.25);
    gl.uniform1f(u.u_fogRange, this.radius * 2.2);
    // Far from any object number, -1 included: that is "no object".
    gl.uniform1f(u.u_selected, this.selected >= 0 ? this.selected : -1e9);
    gl.uniform1f(u.u_plain, this.wireframe ? 1 : 0);
    const shadow = this.shadow && this.shadow.viewProj ? this.shadow : null;
    gl.uniform1f(u.u_shadows, shadow ? 1 : 0);
    if (shadow) {
      gl.uniformMatrix4fv(u.u_lightViewProj, false, shadow.viewProj);
      gl.uniform1f(u.u_shadowTexel, 1 / shadow.size);
      gl.uniform1f(u.u_shadowOffset, shadow.texelWorld);
      gl.uniform1f(u.u_shadowBias, shadow.bias);
    }
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, shadow ? shadow.texture : this.noShadow);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.placeholder);
    gl.uniform1i(u.u_texture, 0);
    gl.uniform1i(u.u_shadowMap, 1);

    // The land past the edge.
    if (this.base) {
      this.setMaterial(u, { color: BASE_COLOR, texture: null, roughness: 1, cutoff: 0 });
      this.setConstants({ has: {}, matrix: identity() });
      gl.bindVertexArray(this.base.vao);
      gl.drawArrays(gl.TRIANGLES, 0, this.base.count);
    }
    if (this.wireframe) {
      // The faces stand back a hair, so the lines on them always win.
      gl.enable(gl.POLYGON_OFFSET_FILL);
      gl.polygonOffset(1, 1);
    }
    this.drawAll(u, true);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    if (this.wireframe) this.drawWires(m);
    gl.bindVertexArray(null);
    if (this.on.view) this.on.view({ ...this.camera });
  }

  /* The wireframe, after the faces. The lines write depth and pass only in
     front of what is there, so a side two triangles share, or two flat
     faces with their own copies of a corner, is drawn once, not twice as
     dark. */
  drawWires(m) {
    const gl = this.gl, wire = this.programs.wire, u = wire.uniforms;
    gl.useProgram(wire.program);
    gl.uniformMatrix4fv(u.u_viewProj, false, m.viewProj);
    gl.uniform3fv(u.u_eye, m.eye);
    gl.uniform1f(u.u_fogStart, this.camera.distance + this.radius * 0.25);
    gl.uniform1f(u.u_fogRange, this.radius * 2.2);
    gl.uniform1f(u.u_selected, this.selected >= 0 ? this.selected : -1e9);
    gl.uniform3fv(u.u_highlight, SELECTED);
    // Pixels per metre at a metre away, on the drawing buffer.
    gl.uniform1f(u.u_focal, this.canvas.height / (2 * Math.tan(FOV / 2)));
    gl.uniform1f(u.u_plain, 1);
    gl.uniform1i(u.u_texture, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthFunc(gl.LESS);
    gl.depthMask(true);
    gl.activeTexture(gl.TEXTURE0);
    let bound = null;
    for (const d of this.draws) {
      if (!this.layers[d.layer] || !this.layers[d.layer].visible) continue;
      const lines = this.wireOf(d);
      if (!lines) continue;
      const material = this.materials[d.material] || this.materials[0];
      const texture = this.textureFor(material);
      if (texture !== bound) {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        bound = texture;
      }
      this.setMaterial(u, material);
      this.setConstants(d);
      gl.bindVertexArray(lines.vao);
      if (d.instances) gl.drawElementsInstanced(gl.LINES, lines.count, gl.UNSIGNED_INT, 0, d.instances);
      else gl.drawElements(gl.LINES, lines.count, gl.UNSIGNED_INT, 0);
    }
    gl.disable(gl.BLEND);
  }

  setWireframe(on) {
    this.wireframe = !!on;
    this.redraw();
  }

  /* Every visible draw, with the program whose uniforms are `u` in use.
     Unit 0 holds the placeholder when this starts. */
  drawAll(u, withTextures) {
    const gl = this.gl;
    let bound = this.placeholder;
    for (const d of this.draws) {
      if (!this.layers[d.layer] || !this.layers[d.layer].visible) continue;
      const material = this.materials[d.material] || this.materials[0];
      if (withTextures || material.cutoff > 0) {
        const texture = this.textureFor(material);
        if (texture !== bound) {
          gl.activeTexture(gl.TEXTURE0);
          gl.bindTexture(gl.TEXTURE_2D, texture);
          bound = texture;
        }
      }
      this.setMaterial(u, material);
      this.setConstants(d);
      gl.bindVertexArray(d.vao);
      if (d.indexType) {
        if (d.instances) gl.drawElementsInstanced(gl.TRIANGLES, d.count, d.indexType, d.offset, d.instances);
        else gl.drawElements(gl.TRIANGLES, d.count, d.indexType, d.offset);
      } else if (d.instances) {
        gl.drawArraysInstanced(gl.TRIANGLES, 0, d.count, d.instances);
      } else {
        gl.drawArrays(gl.TRIANGLES, 0, d.count);
      }
    }
  }

  textureFor(material) {
    if (material.texture === null || material.texture === undefined) return this.placeholder;
    const entry = this.textures[material.texture];
    return (entry && entry.gl) || this.placeholder;
  }

  setMaterial(u, material) {
    const gl = this.gl;
    const textured = material.texture !== null && material.texture !== undefined;
    if (u.u_color) gl.uniform4fv(u.u_color, material.color);
    if (u.u_hasTexture) gl.uniform1f(u.u_hasTexture, textured ? 1 : 0);
    if (u.u_cutoff) gl.uniform1f(u.u_cutoff, material.cutoff || 0);
    if (u.u_roughness) gl.uniform1f(u.u_roughness, material.roughness);
  }

  /* What a draw does not carry per vertex: its matrix, unless instanced,
     and a "no object" number where it has none. */
  setConstants(d) {
    const gl = this.gl;
    if (!d.instances) {
      const m = d.matrix;
      for (let c = 0; c < 4; c++) gl.vertexAttrib4f(LOC.MATRIX + c, m[c * 4], m[c * 4 + 1], m[c * 4 + 2], m[c * 4 + 3]);
    }
    if (!d.has._OBJECT) gl.vertexAttrib1f(LOC._OBJECT, -1);
    if (!d.has.TEXCOORD_0) gl.vertexAttrib2f(LOC.TEXCOORD_0, 0, 0);
    if (!d.has.NORMAL) gl.vertexAttrib3f(LOC.NORMAL, 0, 1, 0);
  }

  get sunDirection() {
    // Towards the sun. North is -Z, east +X.
    return [
      Math.sin(SUN_AZIMUTH) * Math.cos(SUN_ELEVATION),
      Math.sin(SUN_ELEVATION),
      -Math.cos(SUN_AZIMUTH) * Math.cos(SUN_ELEVATION)
    ];
  }

  /* One depth map from the sun, fitted to the model. The sun does not move,
     so it is drawn again only when a layer is switched on or off. */
  renderShadows() {
    const gl = this.gl;
    this.shadowDirty = false;
    if (!this.shadow) {
      const size = Math.min(SHADOW_SIZE, gl.getParameter(gl.MAX_TEXTURE_SIZE));
      const texture = this.depthTexture(size);
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, texture, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.deleteFramebuffer(framebuffer);
        gl.deleteTexture(texture);
        return;
      }
      this.shadow = { size, texture, framebuffer };
    }
    const { min, max } = this.bounds;
    const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    const sun = this.sunDirection;
    const reach = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    const view = lookAt(add(center, scale(sun, reach)), center, [0, 1, 0]);
    // The box around the model, as the sun sees it.
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < 8; i++) {
      const p = transformPoint(view, [i & 1 ? max[0] : min[0], i & 2 ? max[1] : min[1], i & 4 ? max[2] : min[2]]);
      for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
    }
    const projection = orthographic(lo[0], hi[0], lo[1], hi[1], -hi[2] - 1, -lo[2] + 1);
    const s = this.shadow;
    s.viewProj = multiply(projection, view);
    s.texelWorld = Math.max(hi[0] - lo[0], hi[1] - lo[1]) / s.size;
    // A few centimetres of depth, said in the map's 0-to-1 depth.
    s.bias = 0.04 / (hi[2] - lo[2] + 2);

    gl.bindFramebuffer(gl.FRAMEBUFFER, s.framebuffer);
    gl.viewport(0, 0, s.size, s.size);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.POLYGON_OFFSET_FILL);
    gl.polygonOffset(1.5, 3);
    // The map being drawn into may not sit on any unit the program reads.
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.noShadow);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.placeholder);
    const program = this.programs.shadow;
    gl.useProgram(program.program);
    gl.uniformMatrix4fv(program.uniforms.u_viewProj, false, s.viewProj);
    gl.uniform1i(program.uniforms.u_texture, 0);
    this.drawAll(program.uniforms, false);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /* ---------- picking ---------- */

  pickTarget() {
    const gl = this.gl, { width, height } = this.canvas;
    if (this.pickBuffer && this.pickBuffer.width === width && this.pickBuffer.height === height) return this.pickBuffer;
    if (this.pickBuffer) {
      gl.deleteFramebuffer(this.pickBuffer.framebuffer);
      this.pickBuffer.parts.forEach(p => gl.deleteRenderbuffer(p));
    }
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const parts = [gl.RGBA8, gl.RGBA8, gl.DEPTH_COMPONENT24].map((format, i) => {
      const rb = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
      gl.renderbufferStorage(gl.RENDERBUFFER, format, width, height);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, i === 2 ? gl.DEPTH_ATTACHMENT : gl.COLOR_ATTACHMENT0 + i, gl.RENDERBUFFER, rb);
      return rb;
    });
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.pickBuffer = { framebuffer, parts, width, height };
    return this.pickBuffer;
  }

  /* What is under a point of the canvas (CSS pixels): the object's number
     and the point on it, or null over sky. Draws one pixel's worth. */
  pick(x, y) {
    if (this.dead || !this.draws.length) return null;
    this.resize();
    const gl = this.gl, target = this.pickTarget();
    const rect = this.canvas.getBoundingClientRect();
    const px = Math.floor((x / rect.width) * target.width);
    const py = target.height - 1 - Math.floor((y / rect.height) * target.height);
    if (px < 0 || py < 0 || px >= target.width || py >= target.height) return null;
    const m = this.matrices();

    gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
    gl.viewport(0, 0, target.width, target.height);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(px, py, 1, 1);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
    gl.clearBufferfi(gl.DEPTH_STENCIL, 0, 1, 0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    const program = this.programs.pick, u = program.uniforms;
    gl.useProgram(program.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.placeholder);
    gl.uniformMatrix4fv(u.u_viewProj, false, m.viewProj);
    gl.uniform3fv(u.u_eye, m.eye);
    gl.uniform1f(u.u_far, m.far);
    gl.uniform1i(u.u_texture, 0);
    this.drawAll(u, false);
    gl.bindVertexArray(null);

    const object = new Uint8Array(4), distance = new Uint8Array(4);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, object);
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, distance);
    gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const id = object[0] * 65536 + object[1] * 256 + object[2] - 1;
    if (id < 0) return null;
    const along = ((distance[0] * 65536 + distance[1] * 256 + distance[2]) / 16777215) * m.far;
    const { origin, direction } = this.ray(x, y, m);
    return { id, point: add(origin, scale(direction, along)) };
  }

  /* The page's account of a picked point and object. */
  describe(hit) {
    if (!hit) return null;
    const entry = this.objects[hit.id] || [`#${hit.id}`, -1, {}];
    const [name, layer, properties, counts] = entry;
    const p = hit.point, o = this.origin;
    return {
      id: hit.id,
      name,
      layer,
      layerName: this.layers[layer] ? this.layers[layer].name : "",
      properties: properties || {},
      // As Blender counts them, before the export cuts the quads in two.
      mesh: Array.isArray(counts) && counts.length === 4
        ? { vertices: counts[0], triangles: counts[1], quads: counts[2], ngons: counts[3] }
        : null,
      point: p,
      // X east, -Z north, Y up, from the model's origin.
      rd: o ? [o.x + p[0], o.y - p[2]] : null,
      nap: o ? o.z + p[1] : null
    };
  }

  select(id) {
    this.selected = id === null || id === undefined ? -1 : id;
    this.redraw();
  }

  setLayerVisible(index, visible) {
    if (!this.layers[index]) return;
    this.layers[index].visible = visible;
    if (this.selected >= 0 && this.objects[this.selected] && this.objects[this.selected][1] === index && !visible) {
      this.selected = -1;
      (this.on.select || (() => {}))(null);
    }
    this.shadowDirty = true;
    this.redraw();
  }

  /* ---------- input ---------- */

  listen() {
    const c = this.canvas;
    const pointers = new Map();
    let gesture = null, hoverTimer = 0, lastHover = null, wheelPoint = null;
    const local = e => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const view = () => (this.on.view || (() => {}))({ ...this.camera });
    const changed = () => { this.clampCamera(); this.redraw(); view(); };

    const grabAt = (x, y) => {
      const hit = this.pick(x, y);
      const height = hit ? hit.point[1] : this.camera.target[1];
      return { height, point: hit ? hit.point : this.onLevel(x, y, height) };
    };

    const startGesture = e => {
      const [x, y] = local(e);
      if (pointers.size >= 2) {
        const [a, b] = [...pointers.values()];
        const mid = [(a.x + b.x) / 2, (a.y + b.y) / 2];
        gesture = { kind: "touch", spread: Math.hypot(a.x - b.x, a.y - b.y), angle: Math.atan2(b.y - a.y, b.x - a.x), mid, ...grabAt(mid[0], mid[1]) };
        return;
      }
      const turn = e.button === 1 || e.button === 2 || e.shiftKey || e.ctrlKey || e.altKey || e.metaKey;
      gesture = turn ? { kind: "turn" } : { kind: "pan", ...grabAt(x, y) };
      gesture.start = [x, y];
      gesture.moved = 0;
      gesture.button = e.button;
    };

    // Keep the grabbed point under the cursor: move the camera by however
    // far the point has slid on its own level.
    const panTo = (x, y, g) => {
      if (!g.point) return;
      const now = this.onLevel(x, y, g.height);
      if (!now) return;
      this.camera.target[0] += g.point[0] - now[0];
      this.camera.target[2] += g.point[2] - now[2];
    };

    this.handlers = {
      pointerdown: e => {
        if (e.pointerType === "mouse" && e.button > 2) return;
        c.focus({ preventScroll: true });
        c.setPointerCapture(e.pointerId);
        const [x, y] = local(e);
        pointers.set(e.pointerId, { x, y });
        this.animation = null;
        wheelPoint = null;
        startGesture(e);
      },
      pointermove: e => {
        const [x, y] = local(e);
        if (!pointers.has(e.pointerId)) {
          // Hovering: say where the cursor is, a few times a second.
          lastHover = [x, y];
          wheelPoint = null;
          if (!hoverTimer) {
            hoverTimer = setTimeout(() => {
              hoverTimer = 0;
              if (lastHover && this.on.hover) this.on.hover(this.describe(this.pick(lastHover[0], lastHover[1])));
            }, 90);
          }
          return;
        }
        const before = pointers.get(e.pointerId);
        pointers.set(e.pointerId, { x, y });
        if (!gesture) return;
        if (gesture.kind === "touch" && pointers.size >= 2) {
          const [a, b] = [...pointers.values()];
          const mid = [(a.x + b.x) / 2, (a.y + b.y) / 2];
          const spread = Math.hypot(a.x - b.x, a.y - b.y), angle = Math.atan2(b.y - a.y, b.x - a.x);
          // Two fingers: pinch to zoom, twist to turn, both up or down to tilt.
          if (gesture.spread > 0 && spread > 0) this.camera.distance *= gesture.spread / spread;
          this.camera.yaw -= angle - gesture.angle;
          const dy = mid[1] - gesture.mid[1];
          if (Math.abs(spread - gesture.spread) < 6) this.camera.pitch += dy * 0.006;
          this.clampCamera();
          panTo(mid[0], mid[1], gesture);
          Object.assign(gesture, { spread, angle, mid });
          changed();
          return;
        }
        gesture.moved += Math.abs(x - before.x) + Math.abs(y - before.y);
        if (gesture.kind === "pan") {
          panTo(x, y, gesture);
        } else if (gesture.kind === "turn") {
          this.camera.yaw -= (x - before.x) * 0.0065;
          this.camera.pitch += (y - before.y) * 0.005;
        }
        changed();
      },
      pointerup: e => {
        const [x, y] = local(e);
        pointers.delete(e.pointerId);
        if (gesture && gesture.kind !== "touch" && gesture.moved < 5 && gesture.button === 0) {
          // A click, not a drag: what is this?
          const info = this.describe(this.pick(x, y));
          this.select(info ? info.id : null);
          (this.on.select || (() => {}))(info);
        }
        gesture = null;
        if (pointers.size) {
          // One finger left of two: carry on panning with it.
          const [[id, p]] = [...pointers.entries()];
          gesture = { kind: "pan", moved: 99, button: -1, ...grabAt(p.x, p.y) };
          void id;
        }
      },
      pointercancel: e => { pointers.delete(e.pointerId); gesture = null; },
      pointerleave: e => {
        if (e.pointerType === "mouse" && !pointers.size) {
          lastHover = null;
          if (this.on.hover) this.on.hover(null);
        }
      },
      dblclick: e => {
        const [x, y] = local(e);
        const hit = this.pick(x, y);
        if (hit) this.focus(hit.point);
      },
      wheel: e => {
        e.preventDefault();
        const [x, y] = local(e);
        const lines = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
        const factor = Math.exp(clamp(e.deltaY * lines, -400, 400) * 0.0012);
        // Toward the point under the cursor, which stays under it. Asked once
        // while the cursor rests: zooming toward it keeps it there.
        if (!wheelPoint || wheelPoint.x !== x || wheelPoint.y !== y) {
          const hit = this.pick(x, y);
          wheelPoint = { x, y, point: hit ? hit.point : this.onLevel(x, y, this.camera.target[1]) };
        }
        this.animation = null;
        const c0 = this.camera, distance = clamp(c0.distance * factor, MIN_DISTANCE, this.radius * 6);
        const f = distance / c0.distance;
        if (wheelPoint.point) {
          c0.target = [0, 1, 2].map(k => wheelPoint.point[k] + (c0.target[k] - wheelPoint.point[k]) * f);
        }
        c0.distance = distance;
        changed();
      },
      contextmenu: e => e.preventDefault(),
      keydown: e => {
        const c0 = this.camera, step = c0.distance * 0.08;
        const r = [Math.cos(c0.yaw), 0, -Math.sin(c0.yaw)], f = [-Math.sin(c0.yaw), 0, -Math.cos(c0.yaw)];
        const move = (v, s) => { c0.target = add(c0.target, scale(v, s)); };
        const turn = e.shiftKey;
        switch (e.key) {
          case "ArrowLeft": case "a": case "A": turn ? (c0.yaw += 0.08) : move(r, -step); break;
          case "ArrowRight": case "d": case "D": turn ? (c0.yaw -= 0.08) : move(r, step); break;
          case "ArrowUp": case "w": case "W": turn ? (c0.pitch += 0.06) : move(f, step); break;
          case "ArrowDown": case "s": case "S": turn ? (c0.pitch -= 0.06) : move(f, -step); break;
          case "+": case "=": case "PageUp": c0.distance /= 1.25; break;
          case "-": case "_": case "PageDown": c0.distance *= 1.25; break;
          case "q": case "Q": c0.yaw += 0.08; break;
          case "e": case "E": c0.yaw -= 0.08; break;
          case "Home": this.overview(); return;
          default: return;
        }
        e.preventDefault();
        this.animation = null;
        wheelPoint = null;
        changed();
      }
    };
    for (const [name, handler] of Object.entries(this.handlers)) {
      c.addEventListener(name, handler, name === "wheel" ? { passive: false } : undefined);
    }
  }

  /* ---------- the end ---------- */

  dispose() {
    this.dead = true;
    this.abort.abort();
    if (this.frame) cancelAnimationFrame(this.frame);
    this.resizer.disconnect();
    for (const [name, handler] of Object.entries(this.handlers || {})) this.canvas.removeEventListener(name, handler);
    this.canvas.removeEventListener("webglcontextlost", this.lost);
    const gl = this.gl;
    if (!gl.isContextLost()) {
      this.buffers.forEach(b => gl.deleteBuffer(b));
      this.vaos.forEach(v => gl.deleteVertexArray(v));
      this.textures.forEach(t => t.gl && gl.deleteTexture(t.gl));
      if (this.shadow) { gl.deleteTexture(this.shadow.texture); gl.deleteFramebuffer(this.shadow.framebuffer); }
      if (this.pickBuffer) {
        gl.deleteFramebuffer(this.pickBuffer.framebuffer);
        this.pickBuffer.parts.forEach(p => gl.deleteRenderbuffer(p));
      }
      // Hand the graphics memory back now rather than whenever the page goes.
      const lose = gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();
    }
  }
}

/* An image file as something WebGL can take, no larger than `limit`. */
async function decodeImage(blob, limit) {
  const options = { premultiplyAlpha: "none", colorSpaceConversion: "none" };
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob, options);
  } catch {
    bitmap = await createImageBitmap(blob);
  }
  const largest = Math.max(bitmap.width, bitmap.height);
  if (largest <= limit) return bitmap;
  const s = limit / largest;
  const resized = await createImageBitmap(bitmap, {
    ...options,
    resizeWidth: Math.max(1, Math.round(bitmap.width * s)),
    resizeHeight: Math.max(1, Math.round(bitmap.height * s)),
    resizeQuality: "high"
  });
  bitmap.close();
  return resized;
}
