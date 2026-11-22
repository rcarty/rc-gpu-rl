// 3d quadrotor and its policy, shared by every pass

// policy: 12 inputs, two hidden layers of 16 (tanh), 4 outputs (tanh)
const NI: u32 = 12u;
const HID: u32 = 16u;
const NO: u32 = 4u;
const NP: u32 = 548u; // NI*HID + HID + HID*HID + HID + HID*NO + NO

const GRAV: f32 = 9.81;
const DT: f32 = 1.0 / 60.0;
const ARM: f32 = 0.15;
const KYAW: f32 = 0.02; // yaw torque per newton of thrust
const BOUND: vec3f = vec3f(2.4, 1.5, 2.4);
const DEAD_COST: f32 = 3.0;

// 80 bytes, matches writeParams in main.js
struct Params {
  gen: u32,
  pairs: u32,
  steps: u32,
  t0: u32,
  ep_len: u32,
  watch: u32,
  seed: u32,
  n_envs: u32,
  target_a: vec3f,
  sigma: f32,
  target_b: vec3f,
  wind: f32,
  mass_rand: f32,
  lr: f32,
  fault: f32, // power left in motor 0 when watching, 1 is healthy
}

// 24 floats, matches STRIDE in main.js. y is up, the motors push along body y
struct Drone {
  pos: vec3f,
  fit: f32,
  vel: vec3f,
  alive: f32,
  q: vec4f, // body to world, xyzw
  om: vec3f, // spin in the body frame
  dist: f32,
  thrust: vec4f,
  mass: f32,
  wind_amp: f32,
  wind_phase: f32,
  wind_dir: f32,
}

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rand(a: u32, b: u32, c: u32) -> f32 {
  return f32(pcg(a ^ pcg(b ^ pcg(c))) >> 8u) / 16777216.0;
}

fn gauss(a: u32, b: u32, c: u32) -> f32 {
  let u1 = max(rand(a, b, c * 2u), 1e-7);
  let u2 = rand(a, b, c * 2u + 1u);
  return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}

fn gen_key(p: Params) -> u32 {
  return p.seed ^ (p.gen * 0x9e3779b9u);
}

// the noise for pair k, weight j. perturb and update must agree on this
fn noise(p: Params, k: u32, j: u32) -> f32 {
  return gauss(gen_key(p), k, j);
}

fn qmul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

fn qrot(q: vec4f, v: vec3f) -> vec3f {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

// world to body
fn qrot_inv(q: vec4f, v: vec3f) -> vec3f {
  return qrot(vec4f(-q.xyz, q.w), v);
}

// where motor i sits in the body frame
fn motor_pos(i: u32) -> vec3f {
  let l = ARM * 0.70710678;
  let sx = select(-l, l, i == 0u || i == 3u);
  let sz = select(-l, l, i < 2u);
  return vec3f(sx, 0.0, sz);
}

fn spawn(key: u32, id: u32, p: Params) -> Drone {
  var d: Drone;
  d.pos =
    (vec3f(rand(key, id, 0u), rand(key, id, 1u), rand(key, id, 2u)) * 2.0 - 1.0) * vec3f(
      1.5,
      0.8,
      1.5,
    );
  d.vel = (vec3f(rand(key, id, 3u), rand(key, id, 4u), rand(key, id, 5u)) * 2.0 - 1.0) * 0.5;
  // a small tilt about a mostly horizontal axis, then any heading
  let axis =
    normalize(
      (vec3f(rand(key, id, 6u), rand(key, id, 7u), rand(key, id, 8u)) * 2.0 - 1.0) * vec3f(
        1.0,
        0.3,
        1.0,
      ) + 1e-5,
    );
  let tilt = (rand(key, id, 9u) * 2.0 - 1.0) * 0.2;
  let yaw = (rand(key, id, 10u) * 2.0 - 1.0) * 1.5707963;
  d.q = qmul(vec4f(axis * sin(tilt), cos(tilt)), vec4f(0.0, sin(yaw), 0.0, cos(yaw)));
  d.om = vec3f(rand(key, id, 11u), rand(key, id, 12u), rand(key, id, 13u)) * 2.0 - 1.0;
  d.fit = 0.0;
  d.alive = 1.0;
  d.dist = 0.0;
  d.thrust = vec4f(0.0);
  d.mass = 1.0 + p.mass_rand * (rand(key, id, 14u) * 2.0 - 1.0);
  d.wind_amp = p.wind * (0.5 + 0.5 * rand(key, id, 15u));
  d.wind_phase = rand(key, id, 16u) * 6.2831853;
  d.wind_dir = rand(key, id, 17u) * 6.2831853;
  return d;
}

// everything in the body frame, so the policy doesn't care which way the drone faces
fn observe(d: Drone, tgt: vec3f) -> array<f32, 12> {
  let rel = qrot_inv(d.q, clamp(tgt - d.pos, vec3f(-1.5), vec3f(1.5)));
  let vel = qrot_inv(d.q, d.vel) * 0.5;
  let up = qrot_inv(d.q, vec3f(0.0, 1.0, 0.0));
  let om = d.om * 0.1;
  return
    array<f32, 12>(rel.x, rel.y, rel.z, vel.x, vel.y, vel.z, up.x, up.y, up.z, om.x, om.y, om.z);
}

// one semi-implicit Euler step. a is the policy output in -1..1 per motor
fn integrate(d: ptr<function, Drone>, a: vec4f, time: f32, fault: f32) {
  let umax = 0.5 * GRAV; // zero output hovers at the nominal mass
  let u = (a + 1.0) * 0.5 * umax * vec4f(fault, 1.0, 1.0, 1.0);
  let m = (*d).mass;
  let l = ARM * 0.70710678;
  let gust =
    (*d).wind_amp * (0.6 * sin(1.3 * time + (*d).wind_phase) + 0.4 * sin(
      3.7 * time + 2.0 * (*d).wind_phase,
    ));
  let push = qrot((*d).q, vec3f(0.0, u.x + u.y + u.z + u.w, 0.0));
  let wind = gust * vec3f(cos((*d).wind_dir), 0.0, sin((*d).wind_dir));
  let acc = (push + wind) / m - vec3f(0.0, GRAV, 0.0) - 0.3 * (*d).vel;
  let inertia = vec3f(0.01, 0.02, 0.01) * m;
  let torque =
    vec3f(
      -l * (u.x + u.y - u.z - u.w),
      KYAW * (u.x - u.y + u.z - u.w),
      l * (u.x - u.y - u.z + u.w),
    );
  let om = (*d).om;
  let spin = (torque - cross(om, inertia * om)) / inertia - 0.5 * om;
  (*d).vel += DT * acc;
  (*d).om += DT * spin;
  (*d).pos += DT * (*d).vel;
  (*d).q = normalize((*d).q + 0.5 * DT * qmul((*d).q, vec4f((*d).om, 0.0)));
  (*d).thrust = u / umax;
}
