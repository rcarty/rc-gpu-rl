//#include "common.wgsl"

// floor and ceiling, then every drone as six flat quads in its body plane, then the target

struct View {
  vp: mat4x4f,
  right: vec3f,
  pop: u32,
  up: vec3f,
  hl: u32,
  goal: vec3f,
  pop_alpha: f32,
}

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> drones: array<Drone>;

const CORNERS =
  array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, 1.0),
  );
const BLUE = vec3f(0.486, 0.769, 1.0);
const WARM = vec3f(1.0, 0.624, 0.353);
const FLOOR_Y: f32 = -1.5;

// anti-aliased edge of a distance field, w is the pixel footprint
fn edge(d: f32, w: f32) -> f32 {
  return 1.0 - smoothstep(-0.5 * w, 0.5 * w, d);
}

// floor (instance 0) and ceiling outline (instance 1)
struct FloorOut {
  @builtin(position) pos: vec4f,
  @location(0) p: vec2f,
  @location(1) @interpolate(flat) top: u32,
}

@vertex
fn vs_floor(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FloorOut {
  var o: FloorOut;
  o.p = CORNERS[vi] * (BOUND.xz + 0.3);
  o.top = ii;
  o.pos = V.vp * vec4f(o.p.x, select(FLOOR_Y, BOUND.y, ii == 1u), o.p.y, 1.0);
  return o;
}

@fragment
fn fs_floor(i: FloorOut) -> @location(0) vec4f {
  let w = max(fwidth(i.p.x), fwidth(i.p.y));
  let q = abs(i.p) - BOUND.xz;
  let box = max(q.x, q.y);
  let line = edge(abs(box) - 0.004, w);
  let g = abs(fract(i.p / 0.5 + 0.5) - 0.5) * 0.5; // metres to the nearest grid line
  let grid = edge(min(g.x, g.y) - 0.002, w) * step(box, 0.0);
  let a = select(max(line * 0.5, grid * 0.16), line * 0.18, i.top == 1u);
  return vec4f(vec3f(0.6, 0.63, 0.7) * a, a);
}

// drones. the first pop + 1 instances are shadows, the rest are the bodies
struct DroneOut {
  @builtin(position) pos: vec4f,
  @location(0) p: vec2f,
  @location(1) @interpolate(flat) kind: u32, // 0 arm, 1 rotor
  @location(2) @interpolate(flat) thrust: f32,
  @location(3) @interpolate(flat) tint: vec4f, // premultiplied colour, alpha
}

@vertex
fn vs_drone(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DroneOut {
  var o: DroneOut;
  let per = V.pop + 1u;
  let shadow = ii < per;
  let slot = ii % per;
  let hl = slot == V.pop;
  let e = select(slot, V.hl, hl);
  let d = drones[e];
  let part = vi / 6u;
  let c = CORNERS[vi % 6u];

  var body: vec3f;
  if (part < 2u) {
    // two arms across the diagonals
    let along = normalize(vec3f(1.0, 0.0, select(-1.0, 1.0, part == 0u)));
    let side = vec3f(-along.z, 0.0, along.x);
    o.p = c * vec2f(ARM, 0.014);
    o.kind = 0u;
    body = along * o.p.x + side * o.p.y;
  } else {
    let m = part - 2u;
    o.p = c * 0.075;
    o.kind = 1u;
    o.thrust = d.thrust[m];
    body = motor_pos(m) + vec3f(o.p.x, 0.0, o.p.y);
  }

  var world = d.pos + qrot(d.q, body);
  if (shadow) {
    world.y = FLOOR_Y + 0.003;
  }
  let col = select(BLUE, mix(WARM, vec3f(1.0), 0.25), hl);
  let alpha = select(V.pop_alpha, 1.0, hl);
  o.tint = select(vec4f(col * alpha, alpha), vec4f(0.0, 0.0, 0.0, alpha * 0.5), shadow);
  o.pos = V.vp * vec4f(world, 1.0);
  if (d.alive == 0.0 || (!hl && e == V.hl)) {
    o.pos = vec4f(-2.0, -2.0, 0.0, 1.0); // zero area, off screen
  }
  return o;
}

@fragment
fn fs_drone(i: DroneOut) -> @location(0) vec4f {
  let w = max(fwidth(i.p.x), fwidth(i.p.y));
  var a = 1.0;
  if (i.kind == 1u) {
    // rotor: a ring, filled in more as the motor works harder
    let r = length(i.p);
    let ring = edge(abs(r - 0.062) - 0.006, w);
    let disc = edge(r - 0.062, w) * (0.12 + 0.6 * i.thrust);
    a = max(ring, disc);
  }
  return i.tint * a;
}

// target: a ring on the floor under it (instance 0) and a ring facing the camera (instance 1)
struct TargetOut {
  @builtin(position) pos: vec4f,
  @location(0) p: vec2f,
  @location(1) @interpolate(flat) flat_on_floor: u32,
}

@vertex
fn vs_target(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> TargetOut {
  var o: TargetOut;
  o.p = CORNERS[vi] * 0.14;
  o.flat_on_floor = 1u - ii;
  let lifted = V.goal + V.right * o.p.x + V.up * o.p.y;
  let on_floor = vec3f(V.goal.x + o.p.x, FLOOR_Y + 0.004, V.goal.z + o.p.y);
  o.pos = V.vp * vec4f(select(lifted, on_floor, ii == 0u), 1.0);
  return o;
}

@fragment
fn fs_target(i: TargetOut) -> @location(0) vec4f {
  let w = max(fwidth(i.p.x), fwidth(i.p.y));
  let r = length(i.p);
  let a = max(edge(abs(r - 0.09) - 0.008, w), edge(r - 0.018, w));
  let col = select(vec3f(1.0), vec3f(0.0), i.flat_on_floor == 1u);
  let alpha = a * select(1.0, 0.6, i.flat_on_floor == 1u);
  return vec4f(col * alpha, alpha);
}
