//#include "common.wgsl"

// P.steps physics steps per env, policy in the loop

@group(0) @binding(0) var<storage, read_write> drones: array<Drone>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> P: Params;

fn policy(base: u32, o: array<f32, 12>) -> vec4f {
  var k = base;
  var h1: array<f32, 16>;
  for (var i = 0u; i < HID; i++) {
    var a = 0.0;
    for (var j = 0u; j < NI; j++) {
      a += weights[k] * o[j];
      k++;
    }
    h1[i] = a;
  }
  for (var i = 0u; i < HID; i++) {
    h1[i] = tanh(h1[i] + weights[k]);
    k++;
  }
  var h2: array<f32, 16>;
  for (var i = 0u; i < HID; i++) {
    var a = 0.0;
    for (var j = 0u; j < HID; j++) {
      a += weights[k] * h1[j];
      k++;
    }
    h2[i] = a;
  }
  for (var i = 0u; i < HID; i++) {
    h2[i] = tanh(h2[i] + weights[k]);
    k++;
  }
  var out = vec4f(0.0);
  for (var i = 0u; i < NO; i++) {
    var a = 0.0;
    for (var j = 0u; j < HID; j++) {
      a += weights[k] * h2[j];
      k++;
    }
    out[i] = a;
  }
  return tanh(out + vec4f(weights[k], weights[k + 1u], weights[k + 2u], weights[k + 3u]));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let e = gid.x;
  if (e >= P.n_envs) {
    return;
  }
  var d = drones[e];
  for (var s = 0u; s < P.steps; s++) {
    let t = P.t0 + s;
    // second half of the episode moves the target
    let tgt = select(P.target_a, P.target_b, P.watch == 0u && t >= P.ep_len / 2u);

    if (d.alive == 0.0) {
      if (P.watch != 0u) {
        d = spawn(P.seed ^ pcg(t), e, P); // respawn somewhere new
      } else {
        d.fit -= DEAD_COST;
        d.dist += DEAD_COST;
        continue;
      }
    }

    // the weak motor only shows up when watching. the policy never trained with it
    let fault = select(1.0, P.fault, P.watch != 0u);
    integrate(&d, policy(e * NP, observe(d, tgt)), f32(t) * DT, fault);
    // written this way round so a NaN position counts as out too
    if (!all(abs(d.pos) <= BOUND)) {
      d.alive = 0.0;
    }
    let dist = min(3.0, distance(d.pos, tgt));
    d.fit -= dist + 0.01 * dot(d.om, d.om);
    d.dist += dist;
  }
  drones[e] = d;
}
