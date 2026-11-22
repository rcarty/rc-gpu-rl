# Training an AI drone pilot in WebGPU

Thousands of simulated drones learning to fly to a target, trained in the browser on the GPU. Every generation, each drone flies a slightly different copy of one small neural network, and the network moves towards the copies that did best. When it can fly, switch to the policy view and steer the target with the cursor.

```sh
git clone --recurse-submodules https://github.com/rcarty/rc-gpu-rl
cd rc-gpu-rl
python3 -m http.server        # then open http://localhost:8000
```

It needs a browser with WebGPU (recent Chrome, Edge, Safari or Firefox). There's no build step and there are no third-party dependencies. Add `?pairs=256` to the URL for a smaller population on a slow GPU.

## Layout

| Path                        | Contents                                                                                                                                                                          |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `index.html`, `src/main.js` | The page, GPU setup, the training loop and the chart                                                                                                                              |
| `src/shaders/common.wgsl`   | The drone, its dynamics, the observation, and the hash noise                                                                                                                      |
| `src/shaders/`              | `perturb`, `reset`, `step`, `keys`, `rank`, `update` and `draw`                                                                                                                   |
| `kit/`                      | [rc-webgpu-kit](https://github.com/rcarty/rc-webgpu-kit) as a git submodule: device setup, buffer helpers, the shader loader, the frame loop, the orbit camera and the radix sort |

## Formatting and lint

```sh
npm install
npm run lint      # eslint, then prettier --check
npm run format    # prettier --write
```

The rules come from rc-webgpu-kit, so every repository that uses it is formatted the same way.

## How it works

**Environment.** A rigid quadrotor with four motors in an X layout, an arm of 0.15 m and a thrust-to-weight ratio of 2, so zero output hovers. Orientation is a quaternion, and the angular update includes the gyroscopic term. The policy sees the target's position relative to the drone (clamped to 1.5 m), its velocity, the world's up direction and its spin rate, all in the body frame, and outputs a thrust for each motor. Working in the body frame means the policy doesn't depend on which way the drone faces. Physics steps at 60 Hz with semi-implicit Euler and a little linear drag. An episode is 360 steps and the target moves at step 180. Each drone gets its own mass (±20% by default) and its own gusts (two sine waves with a random phase and heading). Leaving the arena ends the drone's episode with a fixed cost for every remaining step.

**Policy.** An MLP with 12 inputs, two hidden layers of 16 and 4 outputs, tanh throughout. That is 548 weights.

**Training.** Evolution strategies with antithetic sampling (Salimans et al., 2017). Each generation:

1. `perturb` writes the weights for every drone: θ + σε for one of each pair and θ − σε for the other, with σ = 0.05 by default. 64 extra drones fly θ unchanged, and their average is the score on the chart.
2. `reset` gives both drones of a pair the same start state, mass and gusts.
3. `step` runs the episode, one thread per drone, with the network evaluated in the loop. Reward is minus the distance to the target, minus a small spin penalty.
4. `keys` writes each score as a sortable integer key, the key-value radix sort from rc-webgpu-kit orders the population, and `rank` turns that order into centred ranks in −0.5..0.5.
5. `update` uses one workgroup per weight to sum (u⁺ − u⁻)ε over the pairs, adds a little weight decay, and takes an Adam step (learning rate 0.03 by default).

The noise ε is never stored. `perturb` and `update` both regenerate it from a PCG hash of the seed, the generation, the pair and the weight index, which keeps memory to one weight buffer per drone. The only readback is the scores for the chart, at most one generation at a time.

**Policy view.** Every drone flies θ with no noise, from its own start, mass and gusts. The target follows the cursor on a level plane, and drones that leave the arena respawn. A slider weakens one motor on every drone, which the policy never trained with.

**Controls.** The page has sliders for σ, the learning rate, gusts and mass variation, and a population size from 512 to 16,384 drones. A counter shows drone-steps per second.

## Limits

- The drone is a rigid body with instant motor response and no ground effect or rotor drag
- One fixed network size. Of the ES settings, only the noise and the learning rate can be changed
- The chart's score comes from 64 episodes per generation, so it's noisy
