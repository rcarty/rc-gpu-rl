//#include "common.wgsl"

// es gradient + adam, one workgroup per weight. noise is rebuilt from the hash

const WD: f32 = 0.005;
const B1: f32 = 0.9;
const B2: f32 = 0.999;

@group(0) @binding(0) var<storage, read_write> theta: array<f32>;
@group(0) @binding(1) var<storage, read_write> adam: array<vec2f>;
@group(0) @binding(2) var<storage, read> util: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> part: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let j = wg.x;
  var g = 0.0;
  for (var k = li; k < P.pairs; k += 64u) {
    g += (util[2u * k] - util[2u * k + 1u]) * noise(P, k, j);
  }
  part[li] = g;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) {
      part[li] += part[li + s];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let w = theta[j];
    let grad = part[0] / (f32(P.pairs) * P.sigma) - WD * w;
    var mv = adam[j];
    mv = vec2f(B1 * mv.x + (1.0 - B1) * grad, B2 * mv.y + (1.0 - B2) * grad * grad);
    adam[j] = mv;
    let t = f32(P.gen + 1u);
    let mh = mv.x / (1.0 - pow(B1, t));
    let vh = mv.y / (1.0 - pow(B2, t));
    theta[j] = w + P.lr * mh / (sqrt(vh) + 1e-8); // ascent, fitness goes up
  }
}
