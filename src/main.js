import {
  initGPU,
  buf,
  bind,
  computePipeline,
  dispatchSize,
  fitCanvas,
  frameLoop,
  loadWGSL,
  loadSort,
  mat4,
  pickDir,
  OrbitCamera,
} from '../kit/src/index.js';

const $ = id => document.getElementById(id);
const canvas = $('view');
const status = t => {
  $('status').textContent = t;
};

const small = Math.min(innerWidth, innerHeight) < 700;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const EVAL = 64; // extra envs that fly the unperturbed policy, for the score
const LAYERS = [
  [12, 16],
  [16, 16],
  [16, 4],
];
const NP = LAYERS.reduce((n, [nin, nout]) => n + nin * nout + nout, 0); // 548, as in common.wgsl
const EP = 360; // steps per episode, 6 s at 60 Hz
const STRIDE = 24; // floats per drone
const DIST = 15; // where the summed distance sits in a drone
const SPEEDS = { real: 1, fast: 8, fastest: 90 };
const BG = { r: 0.051, g: 0.055, b: 0.067, a: 1 };

let device;
let context;
let sort;
let pipes;
let shared; // buffers that don't depend on the population
let pop = null; // buffers, sorter and bind groups sized by it
let pairs = small ? 1024 : 2048;
let mode = 'train';
let paused = false;
let speed = 'fast';
let gen = 0;
let t = 0;
let watchT = 0;
let owed = 0; // real-time steps not yet run
let run = 0; // bumped by restart, so a late score read can tell it's stale
let seed = 1;
let targets = [
  [0, 0, 0],
  [0, 0, 0],
];
let cursor = [0, 0, 0];
let sigma = 0.05;
let lr = 0.03;
let wind = 1;
let massRand = 0.2;
let fault = 1;
let history = [];
let reading = false;
let lastCam = null;
let stepped = 0; // drone-steps since the rate was last shown
let rateAt = performance.now();

const cam = new OrbitCamera(canvas, {
  yaw: 0.6,
  pitch: 0.36,
  target: [0, -0.3, 0],
  radius: 2.7,
  depthPad: 9,
  minDist: 3,
  maxDist: 16,
  autoRotate: 0.05,
  onTap: (sx, sy) => aim(sx, sy),
});

async function init() {
  let format;
  ({ device, context, format } = await initGPU({ canvas, alphaMode: 'opaque' }));
  device.lost.then(i => status('GPU device lost: ' + i.message));

  const shader = name => loadWGSL(new URL(`./shaders/${name}.wgsl`, import.meta.url));
  const names = ['perturb', 'reset', 'step', 'keys', 'rank', 'update', 'draw'];
  const src = Object.fromEntries(await Promise.all(names.map(async n => [n, await shader(n)])));
  sort = await loadSort(device);
  const module = device.createShaderModule({ code: src.draw });
  const blend = {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
  };
  const render = name =>
    device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs_' + name },
      fragment: { module, entryPoint: 'fs_' + name, targets: [{ format, blend }] },
    });
  pipes = {
    perturb: computePipeline(device, src.perturb),
    reset: computePipeline(device, src.reset),
    step: computePipeline(device, src.step),
    keys: computePipeline(device, src.keys),
    rank: computePipeline(device, src.rank),
    update: computePipeline(device, src.update),
    floor: render('floor'),
    drone: render('drone'),
    target: render('target'),
  };
  const u = GPUBufferUsage;
  shared = {
    params: buf(device, 80, u.UNIFORM | u.COPY_DST),
    view: buf(device, 112, u.UNIFORM | u.COPY_DST),
    theta: buf(device, NP * 4, u.STORAGE | u.COPY_DST),
    adam: buf(device, NP * 8, u.STORAGE | u.COPY_DST),
  };
}

// everything sized by the population. rebuilt when the drone count changes
function allocate() {
  if (pop) pop.destroy();
  const u = GPUBufferUsage;
  const nEs = 2 * pairs; // the population, in antithetic pairs
  const n = nEs + EVAL;
  const b = {
    weights: buf(device, n * NP * 4, u.STORAGE),
    drones: buf(device, n * STRIDE * 4, u.STORAGE | u.COPY_SRC),
    util: buf(device, nEs * 4, u.STORAGE),
    keys: buf(device, nEs * 4, u.STORAGE),
    tmpKeys: buf(device, nEs * 4, u.STORAGE),
    order: buf(device, nEs * 4, u.STORAGE),
    tmpOrder: buf(device, nEs * 4, u.STORAGE),
    staging: buf(device, n * STRIDE * 4, u.COPY_DST | u.MAP_READ),
  };
  // the population is ranked with the kit's key-value radix sort
  const sorter = sort.radix({
    count: nEs,
    keys: b.keys,
    tmpKeys: b.tmpKeys,
    values: b.order,
    tmpValues: b.tmpOrder,
  });
  const B = (p, r) => bind(device, p, r);
  const { params, view, theta, adam } = shared;
  pop = {
    n,
    nEs,
    ...b,
    sorter,
    envGroups: Math.ceil(n / 64),
    rankGroups: Math.ceil(nEs / 256),
    perturbGroups: dispatchSize(Math.ceil((n * NP) / 256)),
    bg: {
      perturb: B(pipes.perturb, [theta, b.weights, params]),
      reset: B(pipes.reset, [b.drones, params]),
      step: B(pipes.step, [b.drones, b.weights, params]),
      keys: B(pipes.keys, [b.drones, b.keys, b.order, params]),
      rank: B(pipes.rank, [b.order, b.util, params]),
      update: B(pipes.update, [theta, adam, b.util, params]),
      floor: B(pipes.floor, [view]),
      drone: B(pipes.drone, [view, b.drones]),
      target: B(pipes.target, [view]),
    },
    destroy() {
      sorter.destroy();
      Object.values(b).forEach(x => x.destroy());
    },
  };
}

// params uniform, laid out like Params in common.wgsl
const pAB = new ArrayBuffer(80);
const pU = new Uint32Array(pAB);
const pF = new Float32Array(pAB);
function writeParams({ steps = 0, t0 = 0 } = {}) {
  const watch = mode === 'watch';
  const [a, b] = watch ? [cursor, cursor] : targets;
  pU.set([gen, pairs, steps, t0, EP, watch ? 1 : 0, seed, pop.n]);
  pF.set([...a, sigma, ...b, wind, massRand, lr, fault], 8);
  device.queue.writeBuffer(shared.params, 0, pAB);
}

const vAB = new ArrayBuffer(112); // view uniform
const vF = new Float32Array(vAB);
const vU = new Uint32Array(vAB);

function compute(fn) {
  const enc = device.createCommandEncoder();
  const cp = enc.beginComputePass();
  fn(cp);
  cp.end();
  return enc;
}
const dispatch = (cp, name, x, y = 1) => {
  cp.setPipeline(pipes[name]);
  cp.setBindGroup(0, pop.bg[name]);
  cp.dispatchWorkgroups(x, y);
};

// small random weights, last layer scaled down so the first policies output about zero
function randomTheta() {
  const g = () =>
    Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random());
  const w = new Float32Array(NP);
  let k = 0;
  LAYERS.forEach(([nin, nout], l) => {
    const s = (l === LAYERS.length - 1 ? 0.1 : 1) / Math.sqrt(nin);
    for (let i = 0; i < nin * nout; i++) w[k++] = g() * s;
    k += nout; // biases start at zero
  });
  return w;
}

const r11 = () => Math.random() * 2 - 1;
const randTarget = () => [r11() * 1.6, r11() * 0.9, r11() * 1.6];

// new weights + fresh envs, for a generation or for watching
function startEpisode() {
  writeParams();
  const enc = compute(cp => {
    dispatch(cp, 'perturb', ...pop.perturbGroups);
    dispatch(cp, 'reset', pop.envGroups);
  });
  device.queue.submit([enc.finish()]);
}

function newGeneration() {
  targets = [randTarget(), randTarget()];
  t = 0;
  startEpisode();
}

function restart() {
  seed = (Math.random() * 2 ** 32) >>> 0;
  run++;
  gen = 0;
  history = [];
  device.queue.writeBuffer(shared.theta, 0, randomTheta());
  device.queue.writeBuffer(shared.adam, 0, new Float32Array(NP * 2));
  drawChart();
  if (mode === 'watch') setMode('train');
  else newGeneration();
  status(`Generation 0 · ${pop.nEs.toLocaleString()} drones per generation`);
}

function endGeneration() {
  writeParams();
  const enc = compute(cp => {
    dispatch(cp, 'keys', pop.rankGroups);
    pop.sorter.encode(cp);
    dispatch(cp, 'rank', pop.rankGroups);
    dispatch(cp, 'update', NP);
  });
  const grab = !reading;
  if (grab) enc.copyBufferToBuffer(pop.drones, 0, pop.staging, 0, pop.n * STRIDE * 4);
  device.queue.submit([enc.finish()]);
  if (grab) readScores(gen, pop);
  gen++;
  newGeneration();
}

// average distance to the target over the episode, for the chart
async function readScores(g, p) {
  const mine = run;
  reading = true;
  try {
    await p.staging.mapAsync(GPUMapMode.READ);
  } catch {
    reading = false;
    return; // the population was resized while this was in flight
  }
  const f = new Float32Array(p.staging.getMappedRange());
  const dist = e => f[e * STRIDE + DIST] / EP;
  const all = Array.from({ length: p.nEs }, (_, e) => dist(e)).sort((a, b) => a - b);
  let policy = 0;
  for (let e = p.nEs; e < p.n; e++) policy += dist(e) / EVAL;
  p.staging.unmap();
  reading = false;
  if (mine !== run) return; // started over since, so this score belongs to the old run
  history.push({ gen: g, policy, median: all[p.nEs >> 1] });
  drawChart();
  if (mode === 'train')
    status(
      `Generation ${g + 1} · ${p.nEs.toLocaleString()} drones per generation · ` +
        `the policy stays ${policy.toFixed(2)} m from the target on average`,
    );
}

function setMode(m) {
  mode = m;
  $('watch').textContent = m === 'watch' ? 'Back to training' : 'Fly the policy';
  $('train').disabled = m === 'watch';
  document.querySelectorAll('[data-speed]').forEach(b => (b.disabled = m === 'watch'));
  if (m === 'watch') {
    watchT = 0;
    cursor = [0, 0, 0];
    startEpisode();
    status(
      `Flying the policy from generation ${gen}. ${pop.n.toLocaleString()} drones, each with its ` +
        'own mass and gusts. Move the cursor or tap to move the target. Drag to look around.',
    );
  } else {
    newGeneration();
  }
}

function step(steps, t0) {
  writeParams({ steps, t0 });
  device.queue.submit([compute(cp => dispatch(cp, 'step', pop.envGroups)).finish()]);
  stepped += steps * pop.n;
}

// drone-steps per second over the last second or so
function showRate(now) {
  if (now - rateAt < 1000) return;
  const rate = (stepped * 1000) / (now - rateAt);
  stepped = 0;
  rateAt = now;
  $('rate').textContent =
    rate >= 1e6
      ? `${(rate / 1e6).toFixed(1)} million drone-steps per second`
      : `${Math.round(rate / 1e3).toLocaleString()} thousand drone-steps per second`;
}

function frame(now, dt) {
  // real time is 60 steps a second whatever the refresh rate, and no catching up after a slow frame
  owed = Math.min(owed + dt * 60, 4);
  const due = Math.floor(owed);
  owed -= due;
  if (mode === 'watch') {
    if (due) step(due, watchT);
    watchT += due;
  } else if (!paused) {
    let left = speed === 'real' ? due : SPEEDS[speed];
    while (left > 0) {
      const n = Math.min(left, EP - t);
      step(n, t);
      t += n;
      left -= n;
      if (t === EP) endGeneration();
    }
  }
  showRate(now);

  fitCanvas(canvas, small ? 1.5 : 2);
  cam.update(dt, reduce);
  const c = cam.matrices(canvas.width / canvas.height);
  lastCam = c;
  const watching = mode === 'watch';
  const shown = watching ? pop.n : pop.nEs;
  vF.set(mat4.mul(c.proj, c.view), 0);
  vF.set(c.r, 16);
  vU[19] = shown;
  vF.set(c.up, 20);
  vU[23] = watching ? 0 : pop.nEs; // the highlighted drone
  vF.set(watching ? cursor : targets[t < EP / 2 ? 0 : 1], 24);
  vF[27] = Math.min(0.3, 250 / shown);
  device.queue.writeBuffer(shared.view, 0, vAB);

  const enc = device.createCommandEncoder();
  const rp = enc.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: BG,
      },
    ],
  });
  const draw = (name, verts, instances) => {
    rp.setPipeline(pipes[name]);
    rp.setBindGroup(0, pop.bg[name]);
    rp.draw(verts, instances);
  };
  draw('floor', 6, 2);
  draw('drone', 36, 2 * (shown + 1));
  draw('target', 6, 2);
  rp.end();
  device.queue.submit([enc.finish()]);
}

// chart: policy distance and population median per generation
function drawChart() {
  const c = $('chart');
  const dpr = Math.min(devicePixelRatio || 1, 2);
  c.width = Math.round(c.clientWidth * dpr);
  c.height = Math.round(c.clientHeight * dpr);
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = c.clientWidth;
  const h = c.clientHeight;
  const pad = { l: 40, r: 12, t: 10, b: 24 };
  const xMax = Math.max(50, history.length ? history.at(-1).gen + 1 : 0);
  const yMax = 2.5;
  const x = v => pad.l + (v / xMax) * (w - pad.l - pad.r);
  const y = v => pad.t + (1 - Math.min(v, yMax) / yMax) * (h - pad.t - pad.b);
  g.clearRect(0, 0, w, h);
  g.font = '11px "JetBrains Mono", monospace';
  g.fillStyle = '#8e929b';
  g.strokeStyle = '#22252c';
  g.lineWidth = 1;
  for (let v = 0; v <= yMax; v += 0.5) {
    g.beginPath();
    g.moveTo(pad.l, y(v));
    g.lineTo(w - pad.r, y(v));
    g.stroke();
    g.fillText(`${v.toFixed(1)} m`, 0, y(v) + 4);
  }
  g.fillText('generation', w - pad.r - 70, h - 6);
  g.fillText('0', pad.l, h - 6);
  g.fillText(String(xMax), w - pad.r - 110, h - 6);
  const line = (key, colour) => {
    g.strokeStyle = colour;
    g.lineWidth = 2;
    g.beginPath();
    history.forEach((p, i) => (i ? g.lineTo : g.moveTo).call(g, x(p.gen), y(p[key])));
    g.stroke();
  };
  line('median', '#3f6c94');
  line('policy', '#ff9f5a');
}

// put the target where the cursor's ray meets the level plane through the middle of the arena
function aim(sx, sy) {
  if (mode !== 'watch' || !lastCam) return;
  const dir = pickDir(lastCam, sx, sy, lastCam.fovy, lastCam.aspect);
  if (dir[1] > -0.02 || lastCam.eye[1] <= 0) return; // the ray has to come down onto the plane
  const k = -lastCam.eye[1] / dir[1];
  const clamp = (v, m) => Math.max(-m, Math.min(m, v));
  cursor = [clamp(lastCam.eye[0] + dir[0] * k, 2.1), 0, clamp(lastCam.eye[2] + dir[2] * k, 2.1)];
}
canvas.addEventListener('pointermove', e => {
  if (e.pointerType !== 'mouse' || e.buttons) return; // dragging orbits the camera
  const rc = canvas.getBoundingClientRect();
  aim(((e.clientX - rc.left) / rc.width) * 2 - 1, 1 - ((e.clientY - rc.top) / rc.height) * 2);
});

// wired up once the GPU side exists, because most of these touch it
function wireControls() {
  $('train').addEventListener('click', () => {
    paused = !paused;
    $('train').textContent = paused ? 'Train' : 'Pause';
  });
  $('watch').addEventListener('click', () => setMode(mode === 'watch' ? 'train' : 'watch'));
  $('restart').addEventListener('click', restart);
  document.querySelectorAll('[data-speed]').forEach(b =>
    b.addEventListener('click', () => {
      speed = b.dataset.speed;
      document
        .querySelectorAll('[data-speed]')
        .forEach(o => o.setAttribute('aria-pressed', o === b));
    }),
  );
  const slider = (id, fn) => {
    const el = $(id);
    const update = () => {
      $(id + '-out').textContent = fn(+el.value);
    };
    el.addEventListener('input', update);
    update();
  };
  slider('sigma', v => {
    sigma = v;
    return v.toFixed(2);
  });
  slider('lr', v => {
    lr = v;
    return v.toFixed(3);
  });
  slider('wind', v => {
    wind = v;
    return `${v.toFixed(1)} N`;
  });
  slider('mass', v => {
    massRand = v / 100;
    return `±${v}%`;
  });
  slider('fault', v => {
    fault = v / 100;
    return `${v}%`;
  });
  function setPairs(p) {
    pairs = p;
    allocate();
    restart();
  }
  $('drones').value = String(2 * pairs);
  $('drones').addEventListener('change', e => setPairs(+e.target.value / 2));
  addEventListener('resize', drawChart);
}

init()
  .then(() => {
    allocate();
    wireControls();
    restart();
    frameLoop(device, canvas, frame);
  })
  .catch(err => {
    console.warn(err);
    document.body.classList.add('no-gpu');
    // the note in the page says there's no WebGPU. anything else gets its own message
    if (navigator.gpu)
      document.querySelector('.nogpu').textContent = `Couldn't start: ${err.message}`;
    document
      .querySelectorAll('.controls button, .controls select, .sliders input')
      .forEach(b => (b.disabled = true));
  });
