//#include "common.wgsl"

// fitness as a sortable u32 key, env index as the value

@group(0) @binding(0) var<storage, read> drones: array<Drone>;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> vals: array<u32>;
@group(0) @binding(3) var<uniform> P: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let e = gid.x;
  if (e >= 2u * P.pairs) {
    return;
  }
  // flip the sign bit, or every bit for negatives, and floats order like uints
  let b = bitcast<u32>(drones[e].fit);
  keys[e] = select(b ^ 0x80000000u, ~b, (b >> 31u) == 1u);
  vals[e] = e;
}
