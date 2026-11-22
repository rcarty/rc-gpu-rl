//#include "common.wgsl"

// theta +/- sigma * noise, opposite signs within a pair. the rest fly plain theta

@group(0) @binding(0) var<storage, read> theta: array<f32>;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= P.n_envs * NP) {
    return;
  }
  let e = i / NP;
  let j = i % NP;
  var w = theta[j];
  if (P.watch == 0u && e < 2u * P.pairs) {
    let s = select(-1.0, 1.0, (e & 1u) == 0u);
    w += s * P.sigma * noise(P, e / 2u, j);
  }
  weights[i] = w;
}
