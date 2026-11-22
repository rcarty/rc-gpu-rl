//#include "common.wgsl"

// both envs of a pair start from the same state, so the only difference is the noise sign

@group(0) @binding(0) var<storage, read_write> drones: array<Drone>;
@group(0) @binding(1) var<uniform> P: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let e = gid.x;
  if (e >= P.n_envs) {
    return;
  }
  let es = 2u * P.pairs;
  var id = select(P.pairs + e - es, e / 2u, e < es);
  if (P.watch != 0u) {
    id = e;
  }
  // its own stream. on gen_key alone, pair k's start state reuses the draws behind its noise
  drones[e] = spawn(gen_key(P) ^ 0x68bc21ebu, id, P);
}
