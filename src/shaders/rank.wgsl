//#include "common.wgsl"

// centred ranks from the sorted order. the sort is stable, so ties go to the lower index

@group(0) @binding(0) var<storage, read> order: array<u32>;
@group(0) @binding(1) var<storage, read_write> util: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = 2u * P.pairs;
  let r = gid.x;
  if (r >= n) {
    return;
  }
  util[order[r]] = f32(r) / f32(n - 1u) - 0.5;
}
