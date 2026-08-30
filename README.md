# Neuromorphic Oscillator — A Clockless, 100-Neuron Silicon Pacemaker (MOOSE)

## 1. The Scenario

This repository implements and validates the circuit dynamics for an
ultra-low-power, implantable neonatal cardiac pacemaker built from a hardware
network of sub-threshold analog spiking neurons — before committing to an
expensive chip tape-out.

The design goal: two competing pools of silicon neuron circuits that behave
like opposing phases of a clock/cardiac cycle. When Pool A is highly active
it suppresses Pool B, and vice versa. This mutual, alternating "tug-of-war"
produces a stable, rhythmic, **clockless** macro-pulse — entirely from the
internal non-linear feedback of the circuit, never from an external clock
or oscillating drive.

All of this is built and simulated in [MOOSE](https://moose.ncbs.res.in/)
(via `pymoose`), using its native `LIF` (leaky integrate-and-fire) neuron
class as the "analog silicon neuron" surrogate.

---

## 2. Rules → Design Mapping

| Rule | How it's satisfied |
|---|---|
| **Tonic bias only** | Every neuron's `inject` field (its DC bias current) is set exactly once, at t = 0, and never touched again for the rest of the simulation. There is no `PulseGen`, `sine`, `StimulusTable`, or any other time-varying driving signal anywhere in the model. |
| **Dale's Law compliance** | Every neuron belongs to exactly one hard-wired population: **Excitatory (E)** or **Inhibitory (I)**. All outgoing synapses from an E population use a **positive** weight (`exc_syn`); all outgoing synapses from an I population use a **negative** weight (`inh_syn`). No neuron ever forms both types of output synapse. |
| **100-neuron silicon budget** | Exactly 100 `moose.LIF` neurons are instantiated — see the breakdown below. The script asserts this at import time. |
| **Fixed hardwired connectome** | All synaptic connections (which neuron connects to which, with what weight and delay) are decided once during network construction (`connect_synapses()`), using a single seeded random number generator, and never modified again. There is no online learning rule, no STDP, no weight update at runtime. |

### 100-neuron budget

| Pool | Excitatory (E) | Inhibitory (I) | Subtotal |
|---|---|---|---|
| Pool A | 30 | 20 | 50 |
| Pool B | 30 | 20 | 50 |
| **Total** | **60** | **40** | **100** |

---

## 3. Circuit Architecture

```
                     exc_syn                         inh_syn
        ┌───────┐  ─────────►  ┌───────┐  ─────────────────────────────►  ┌───────┐
        │ E_A   │              │ I_A   │                                  │ E_B   │
        │ (30)  │              │ (20)  │                                  │ (30)  │
        └───────┘              └───────┘                                  └───┬───┘
            ▲                                                                 │
            │  tonic bias (DC, fixed)                          exc_syn        │ exc_syn
            │                                                     │           ▼
        ┌───┴───┐                                             ┌───┴───┐  ┌───────┐
        │ bias  │                                             │ I_B   │◄─┤       │
        └───────┘                                             │ (20)  │  └───────┘
                                                                └───┬───┘
                     inh_syn                                       │
        ┌───────┐  ◄─────────────────────────────────────────────┘
        │ E_A   │
        └───────┘
```

* `E_A → I_A` and `E_B → I_B`: feed-forward excitation (`exc_syn`), so an
  active E pool recruits its own local inhibitory bank.
* `I_A → E_B` and `I_B → E_A`: **cross-pool inhibition only** (`inh_syn`).
  This is the only channel of communication between the two pools, and it
  is always suppressive — literally implementing "when Pool A is highly
  active, it suppresses Pool B."
* There is **no direct excitatory connection between the two pools** — the
  only way Pool A can affect Pool B is by silencing it.

### Why doesn't this just settle into one pool winning forever?

A pure mutual-inhibition network like the one above is *bistable*, not
*oscillatory*: whichever pool gets a tiny head start silences the other
forever, and nothing left in the circuit tells it to ever stop. Real half-
center oscillators (in CPG circuits in the spinal cord, in relaxation
oscillators, in astable multivibrators) solve this with a **slow, intrinsic
"fatigue" current**.

We add exactly that: **spike-frequency adaptation (SFA)**, implemented as
a private RC low-pass filter physically local to each Excitatory neuron
(`moose.RC`):

1. Every time an E neuron spikes, a small fixed charge is dumped onto its
   own personal RC integrator (`ADAPT_STEP`, delivered through a private,
   single-neuron `SimpleSynHandler` — this is *not* a synapse to another
   neuron, so it is outside the scope of Dale's Law).
2. That RC integrator's output decays back to zero with a slow time
   constant `ADAPT_TAU` (≈ 300 ms — think of it as a Ca²⁺-activated K⁺
   "leakage" current on real silicon).
3. The RC's (negative) output is fed straight back onto **the same
   neuron's own membrane** (`activation` field).

The more a pool fires, the more its own E population's excitability droops.
Eventually the dominant pool's own fatigue currents outweigh its tonic
bias, its firing rate drops, its inhibitory bank quiets down, and the
previously-silenced rival pool — which never stopped receiving its own
tonic bias — escapes and takes over. Then the process repeats in reverse.
The result is a robust, self-sustaining, **clockless anti-phase
oscillator**, exactly analogous to a two-transistor astable multivibrator,
but built entirely out of spiking neuron circuits and internal feedback.

A tiny, fixed (non-oscillating) bias mismatch between the two pools
(`POOL_BIAS_HEADSTART`, representing unavoidable transistor mismatch on
real silicon) is used only to break the initial left/right symmetry so the
network reliably falls into the alternating limit cycle rather than a
symmetric in-phase one.

---

## 4. Implementation Details

* **Neuron model**: `moose.LIF` (leaky integrate-and-fire), used as the
  "sub-threshold analog neuron" surrogate. Every neuron in a class (E or I)
  shares the same `Rm`, `Cm`, `thresh`, `vReset`, `refractoryPeriod`.
* **Synapse model**: `moose.SimpleSynHandler`, connected to each
  post-synaptic neuron's `activation` field. Each synapse simply adds its
  fixed `weight` directly to the post-synaptic `Vm` on spike arrival
  (a "delta" synapse), with a fixed axonal `delay`. This is the same
  synapse model used in MOOSE's own reference E-I network example
  (`ExcInhNet`, Higgins/Graupner/Brunel 2014 cookbook).
* **Connectivity**: sparse, random, Erdős–Rényi-style (fixed connection
  probabilities `P_EI`, `P_IE`), drawn once from a single seeded
  `numpy.random.default_rng` and then frozen for the rest of the run.
* **Adaptation**: `moose.RC` low-pass filter per E neuron, driven by that
  neuron's own `spikeOut` through a private single-synapse
  `SimpleSynHandler`, feeding back onto that neuron's own `activation`.
* **Scheduling**: everything runs on MOOSE's own discrete-event/fixed-
  timestep scheduler (`dt = 0.5 ms`), for 8 seconds of simulated time.
* **Reproducibility**: a single seed (`SEED = 42`) seeds `numpy`, `moose`,
  and the connectivity RNG, so re-running the script reproduces the exact
  same network and the exact same spike trains.

---

## 5. Results

### 5.1 Spike raster

![Raster plot](images/raster_plot.png)

Top: the full 8-second run, all 100 neurons. Bottom: a 3-second zoom-in.
Blue/cyan = Pool A (E/I), red/orange = Pool B (E/I). You can see the two
pools taking turns being the dominant, high-firing-rate population.

### 5.2 Population firing rate — the macro-pulse pattern

![Population rate](images/population_rate.png)

This is the key result: the smoothed population firing rate of the two
Excitatory sub-populations. Pool A (blue) and Pool B (red) clearly
alternate — whenever one is active (10–50 Hz/neuron), the other drops to
near 0 Hz, and they swap every few hundred milliseconds, continuously, for
the entire run. This is the emergent, clockless "macro-pulse" the pacemaker
needs.

Running `oscillator_model.py` prints a quantitative anti-phase quality
check at the end (computed after discarding the first 1 s transient):

```
Anti-phase quality check (t > 1.0s):
  Pearson correlation of Pool A vs Pool B firing rate: -0.581 (want strongly negative)
  Mean rate Pool A = 9.69 Hz/neuron, Pool B = 9.38 Hz/neuron, balance = 0.98 (1.0 = perfectly symmetric)
```

A strongly negative correlation between the two pools' firing rates, with a
balance close to 1.0 (i.e. neither pool structurally dominates the other),
is exactly the signature of a healthy, symmetric anti-phase oscillator.
This behavior is robust across random seeds (we swept several seeds during
development and consistently saw correlations in the −0.5 to −0.6 range
with balance > 0.85 — see `sweep.py`).

### 5.3 Membrane voltage traces

![Voltage traces](images/voltage_traces.png)

Example `Vm` traces for one E and one I neuron from each pool. Note the
characteristic "sawtooth" pattern of repeated spiking punctuated by slow
adaptation-driven dips — and that Pool A's and Pool B's E-neuron dips are
staggered in time, exactly matching the anti-phase rate pattern above.

---

## 6. How to Run

### Requirements

```bash
pip install pymoose numpy matplotlib --break-system-packages
```

(or `pip install -r requirements.txt`)

### Run the model

```bash
python3 oscillator_model.py
```

This will:
1. Build the 100-neuron, two-pool network described above.
2. Run 8 seconds of simulated time.
3. Save `output/raster_plot.png`, `output/population_rate.png`,
   `output/voltage_traces.png`, and the raw spike times
   (`output/spike_data.npz`).
4. Print the quantitative anti-phase quality metrics to the console.

### (Optional) Re-run the parameter sweep

`sweep.py` contains a lighter-weight, headless version of the network
builder (`run_trial(...)`) used during development to search the
`(W_EE, W_EI, W_IE, ADAPT_STEP, ADAPT_TAU, IBIAS_E, IBIAS_I, headstart)`
parameter space for a configuration that maximizes anti-phase quality
(strong negative correlation between the two pools' rates, with balanced
mean firing rates). Running it directly reproduces the grid search:

```bash
python3 sweep.py
```

---

## 7. Design Choices & Tuning Notes

* **Why is `W_EE = 0`?** An earlier version of this model also included
  recurrent excitatory self-connections within each E population
  (`E_A → E_A`, `E_B → E_B`) to help each pool "ignite" collectively. In
  practice this made the two pools' bursts lock into a **synchronized
  (in-phase)** limit cycle instead of the desired anti-phase one, because
  both pools — being statistically identical — would ramp up together
  through positive feedback before inhibition had a chance to separate
  them. Removing the E→E recurrence (relying on each E neuron's own tonic
  bias to be self-sufficient) and letting cross-pool inhibition + intrinsic
  adaptation do all the work reliably produces clean anti-phase switching
  instead. The E→E connection code path is still present (guarded by
  `if W_EE != 0.0:`) for further experimentation.
* **Why an RC filter for adaptation instead of a literal Ca²⁺/K⁺ channel
  model?** A full Hodgkin-Huxley-style Ca-activated-K⁺ channel would need
  gating-variable lookup tables and a `CaConc` pool, which is significantly
  more machinery for the same qualitative effect. The RC low-pass filter
  captures the essential dynamics (a slow, spike-driven, self-inhibitory
  feedback current) with two parameters (`ADAPT_STEP`, `ADAPT_TAU`) and is
  fully transparent about being a `moose.RC` object, not a synapse — so it
  is unambiguous that it doesn't interact with Dale's Law.
* **Units note**: because the adaptation feedback is delivered as a single
  MOOSE-timestep impulse into an RC integrator's `injectIn` field, its
  effective per-spike impulse (`ADAPT_STEP × dt / ADAPT_TAU`) is much
  smaller than `ADAPT_STEP` itself — this is why `ADAPT_STEP` in the code
  (`-0.015`) looks numerically large compared to the millivolt-scale
  synaptic weights. See the comments above `add_adaptation()` in
  `oscillator_model.py`.
* **Randomness is only ever used for two things**: (1) the fixed,
  once-only wiring diagram (which pre-neuron connects to which
  post-neuron), and (2) fixed, once-only per-neuron bias/initial-condition
  mismatch (representing manufacturing variation). Neither is regenerated
  or resampled during the run — the connectome and every bias current are
  frozen the instant the network is built.

---

## 8. Repository Structure

```
.
├── oscillator_model.py   # Main model: builds, runs, and plots the network
├── sweep.py               # Headless parameter-sweep harness used for tuning
├── requirements.txt
├── README.md              # This file
└── images/
    ├── raster_plot.png
    ├── population_rate.png
    └── voltage_traces.png
```

Running `oscillator_model.py` will also create an `output/` directory with
fresh copies of these plots plus the raw spike-time data
(`spike_data.npz`), so you can re-analyze or re-plot without re-running the
simulation.
