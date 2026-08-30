#!/usr/bin/env python3
"""
oscillator_model.py
====================
Neuromorphic half-center oscillator built in MOOSE.

Two competing hardware "pools" (A and B), each containing an Excitatory (E)
sub-population and an Inhibitory (I) sub-population, wired as a classic
"half-center oscillator" (HCO):

    E_A --(exc_syn)--> I_A   (feed-forward drive of local inhibitory bank)
    I_A --(inh_syn)--> E_B   (cross-pool suppression: A silences B)

    E_B --(exc_syn)--> I_B
    I_B --(inh_syn)--> E_A   (cross-pool suppression: B silences A)

Every E and I neuron is individually tonic: its own constant bias current
is enough, on its own, to make it spike periodically (no recurrent E->E
excitation is required to reach threshold - W_EE is wired in and left at 0
by default, see "Design choices" in the README for why).

Every neuron also receives a single, constant ("tonic") bias current
(the `inject` field) that is set once at t=0 and never touched again -
this is the only external drive in the whole circuit.

Because a purely static mutual-inhibition circuit would just settle into
one pool permanently winning (a bistable "flip-flop", not an oscillator),
each Excitatory neuron additionally carries an intrinsic, per-neuron
Spike-Frequency-Adaptation (SFA) current. This models a slow,
Ca-activated-K+ -like leakage capacitor physically built into every E
neuron circuit (NOT a synapse, so it does not touch Dale's Law): every time
the neuron fires, a small negative charge is dumped onto an RC integrator
local to that same neuron; the RC integrator's slowly-decaying output is
fed straight back into that same neuron's membrane. The more a neuron/pool
fires, the more it fatigues; eventually cross-inhibition + self-fatigue
let the silenced pool escape and take over -> a stable, clockless,
alternating (anti-phase) rhythm emerges purely from internal feedback.

Neuron model : moose.LIF (leaky integrate-and-fire, "analog sub-threshold"
               silicon neuron surrogate)
Synapse model: moose.SimpleSynHandler + delta-activation (weight added
               directly to Vm on spike arrival) - sign of weight encodes
               exc/inh, exactly as in the standard MOOSE E-I network
               cookbook example (ExcInhNet, Brunel-type network).

100-neuron silicon budget
--------------------------
Pool A: 30 Excitatory + 20 Inhibitory = 50
Pool B: 30 Excitatory + 20 Inhibitory = 50
TOTAL                                 = 100   (exactly matches the budget)

Author: (fill in your name)
"""

import numpy as np
import moose
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------------
# 0. Reproducibility
# --------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
moose.seed(SEED)
_GLOBAL_RNG = np.random.default_rng(SEED)  # single shared RNG for all connectivity draws

# --------------------------------------------------------------------------
# 1. The 100-neuron silicon budget
# --------------------------------------------------------------------------
N_E = 30           # excitatory neurons per pool
N_I = 20           # inhibitory neurons per pool
N_POOL = N_E + N_I  # 50 per pool
assert 2 * N_POOL == 100, "Silicon budget must be exactly 100 neurons"

# --------------------------------------------------------------------------
# 2. LIF ("analog silicon neuron") biophysical parameters
# --------------------------------------------------------------------------
EL         = -70e-3     # V   resting / leak potential
VT         = -50e-3     # V   firing threshold
VRESET     = -60e-3     # V   reset potential after a spike
RM         = 20e6       # Ohm membrane (leak) resistance
CM         = 1e-9       # F   membrane capacitance   -> tau_m = Rm*Cm = 20 ms
REFRACT    = 2e-3       # s   absolute refractory period

# Tonic (DC, non-oscillating) bias currents -----------------------------
# Deliberately kept sub-threshold on their own: (VT-EL)/RM ~ 1.0 nA is the
# current that would exactly balance the neuron at threshold with no
# noise. We inject slightly less than that into E cells (so a lone E cell
# is quiescent) and let recurrent E->E excitation push the *pool*, not
# individual neurons, over threshold once it gets going.
IBIAS_E = 1.20e-9        # A, constant bias into every Excitatory neuron
IBIAS_I = 0.75e-9        # A, constant bias into every Inhibitory neuron

# Small, FIXED (not time varying) manufacturing-mismatch-like jitter so the
# two pools are not perfectly symmetric (breaks the symmetry so one pool
# starts first - this is analogous to unavoidable transistor mismatch on
# real silicon, not an externally injected oscillating signal).
IBIAS_JITTER = 0.03e-9

# A deliberate, tiny, but *constant* (non-oscillating) bias mismatch between
# the two pools' Excitatory bias currents. On a real chip this is exactly
# the kind of static transistor-to-transistor mismatch you cannot avoid;
# here it is used to seed a decisive first mover so the network settles
# into the anti-phase limit cycle instead of the symmetric in-phase one.
# It is still 100% tonic/DC - it never changes after t=0.
POOL_BIAS_HEADSTART = 0.010e-9

# --------------------------------------------------------------------------
# 3. Synaptic weights (sign encodes Dale's law: E=+, I=-)
# --------------------------------------------------------------------------
W_EE = 0.0e-3      # V, recurrent E->E excitation within a pool (not needed -
                   # tonic bias alone drives each E cell; see README)
W_EI = 0.90e-3     # V, feed-forward E->I excitation within a pool
W_IE = 2.00e-3     # V, magnitude of cross-pool I->E inhibition (applied as -W_IE)
SYN_DELAY = 1e-3   # s

P_EE = 0.30        # connection probability, E->E (same pool) [unused, W_EE=0]
P_EI = 0.55        # connection probability, E->I (same pool)
P_IE = 0.70        # connection probability, I->E (cross pool)

# --------------------------------------------------------------------------
# 4. Intrinsic spike-frequency-adaptation (SFA) parameters (per E neuron)
# --------------------------------------------------------------------------
ADAPT_STEP  = -0.015     # impulse dumped onto the RC integrator per own spike
ADAPT_TAU   = 0.30       # s, RC decay time constant (~300 ms, slow K+-like)

# --------------------------------------------------------------------------
# 5. Simulation parameters
# --------------------------------------------------------------------------
DT       = 0.5e-3   # s
SIM_TIME = 8.0       # s of simulated time

# ==========================================================================
# BUILD THE NETWORK
# ==========================================================================
moose.Neutral('/model')


def make_pool(name, n_e, n_i, ibias_e, ibias_i):
    """Create one competing pool: an E sub-population and an I sub-population
    of moose.LIF neurons, with fixed intrinsic parameters and constant bias.
    Returns the two LIF vecs."""
    e = moose.LIF(f'/model/{name}_E', n_e)
    i = moose.LIF(f'/model/{name}_I', n_i)

    for pop, ibias, jitter_scale in ((e, ibias_e, 1.0), (i, ibias_i, 1.0)):
        pop.vec.Em = EL
        pop.vec.thresh = VT
        pop.vec.vReset = VRESET
        pop.vec.Rm = RM
        pop.vec.Cm = CM
        pop.vec.refractoryPeriod = REFRACT
        # tonic (constant) bias + fixed per-neuron mismatch, set ONCE at t=0
        jitter = np.random.uniform(-IBIAS_JITTER, IBIAS_JITTER, size=len(pop.vec))
        pop.vec.inject = ibias + jitter * jitter_scale
        # random initial membrane potential (breaks symmetry, purely an
        # initial condition - not a driving oscillation)
        pop.vec.initVm = np.random.uniform(EL, VT, size=len(pop.vec))
    return e, i


poolA_E, poolA_I = make_pool('A', N_E, N_I, IBIAS_E + POOL_BIAS_HEADSTART, IBIAS_I)
poolB_E, poolB_I = make_pool('B', N_E, N_I, IBIAS_E - POOL_BIAS_HEADSTART, IBIAS_I)


def connect_synapses(pre_vec, post_vec, weight, prob, delay, syn_owner_name):
    """Fixed, hard-wired sparse connectivity from pre_vec -> post_vec.
    `weight` sign encodes exc(+)/inh(-) -> Dale's law compliant, since the
    *entire* pre-population uses one sign only.
    A SimpleSynHandler is created per post-synaptic neuron collecting all
    its incoming synapses of this particular projection."""
    n_pre = len(pre_vec.vec)
    n_post = len(post_vec.vec)
    synh = moose.SimpleSynHandler(f'/model/{syn_owner_name}', n_post)
    moose.connect(synh, 'activationOut', post_vec, 'activation', 'OneToOne')

    rng = _GLOBAL_RNG
    for j in range(n_post):
        # Decide, ONCE, which pre-neurons connect to post neuron j.
        mask = rng.random(n_pre) < prob
        pre_idxs = np.nonzero(mask)[0]
        if len(pre_idxs) == 0:           # guarantee at least one input
            pre_idxs = [rng.integers(0, n_pre)]
        synh.vec[j].numSynapses = len(pre_idxs)
        for k, pidx in enumerate(pre_idxs):
            syn = synh.vec[j].synapse[k]
            moose.connect(pre_vec.vec[int(pidx)], 'spikeOut', syn, 'addSpike')
            syn.delay = delay
            syn.weight = weight            # fixed for the entire run
    return synh


# ---- Within-pool excitation (Dale's law: E population -> exc_syn only) ---
# (W_EE = 0 in the tuned operating point below -- tonic bias alone is
#  already enough to drive each E cell; recurrent E->E self-excitation is
#  wired in here and left available for experimentation, see README)
if W_EE != 0.0:
    synh_AA = connect_synapses(poolA_E, poolA_E, W_EE, P_EE, SYN_DELAY, 'synh_AA_EE')
    synh_BB = connect_synapses(poolB_E, poolB_E, W_EE, P_EE, SYN_DELAY, 'synh_BB_EE')
synh_AI = connect_synapses(poolA_E, poolA_I, W_EI, P_EI, SYN_DELAY, 'synh_AI_EI')
synh_BI = connect_synapses(poolB_E, poolB_I, W_EI, P_EI, SYN_DELAY, 'synh_BI_EI')

# ---- Cross-pool inhibition (Dale's law: I population -> inh_syn only) ----
synh_IA_to_EB = connect_synapses(poolA_I, poolB_E, -W_IE, P_IE, SYN_DELAY, 'synh_IA_to_EB')
synh_IB_to_EA = connect_synapses(poolB_I, poolA_E, -W_IE, P_IE, SYN_DELAY, 'synh_IB_to_EA')

# ==========================================================================
# Intrinsic spike-frequency adaptation for every E neuron (NOT a synapse -
# this is a per-neuron feedback capacitor, so it never touches Dale's Law
# or the fixed inter-pool connectome).
# ==========================================================================

def add_adaptation(e_vec, name):
    n = len(e_vec.vec)
    rc = moose.RC(f'/model/{name}_adaptRC', n)
    rc.vec.R = 1.0
    rc.vec.C = ADAPT_TAU        # tau = R*C
    # self spike -> its own RC integrator (one synapse each, a private
    # feedback loop of that single neuron, not a connection to any other
    # neuron in the network)
    adapt_synh = moose.SimpleSynHandler(f'/model/{name}_adaptSyn', n)
    moose.connect(adapt_synh, 'activationOut', rc, 'injectIn', 'OneToOne')
    for i in range(n):
        adapt_synh.vec[i].numSynapses = 1
        syn = adapt_synh.vec[i].synapse[0]
        moose.connect(e_vec.vec[i], 'spikeOut', syn, 'addSpike')
        syn.delay = 0.0
        syn.weight = ADAPT_STEP
    # feed the slow, decaying output straight back onto the SAME neuron
    moose.connect(rc, 'output', e_vec, 'activation', 'OneToOne')
    return rc


rcA = add_adaptation(poolA_E, 'A')
rcB = add_adaptation(poolB_E, 'B')

# ==========================================================================
# Recording
# ==========================================================================
def make_spike_tables(vec, name):
    tabs = moose.Table(f'/model/tab_{name}', len(vec.vec))
    moose.connect(vec, 'spikeOut', tabs, 'input', 'OneToOne')
    return tabs

tabA_E = make_spike_tables(poolA_E, 'A_E')
tabA_I = make_spike_tables(poolA_I, 'A_I')
tabB_E = make_spike_tables(poolB_E, 'B_E')
tabB_I = make_spike_tables(poolB_I, 'B_I')

# a few example Vm traces for illustration
vm_tabs = moose.Table('/model/vm_examples', 4)
moose.connect(poolA_E.vec[0], 'VmOut', vm_tabs.vec[0], 'input')
moose.connect(poolA_I.vec[0], 'VmOut', vm_tabs.vec[1], 'input')
moose.connect(poolB_E.vec[0], 'VmOut', vm_tabs.vec[2], 'input')
moose.connect(poolB_I.vec[0], 'VmOut', vm_tabs.vec[3], 'input')

# ==========================================================================
# Scheduling & Run
# ==========================================================================
for path in ('/model', ):
    moose.setClock(0, DT)
    moose.setClock(1, DT)
    moose.setClock(2, DT)

moose.useClock(0, '/model/A_E,/model/A_I,/model/B_E,/model/B_I', 'init')
moose.useClock(1, '/model/A_E,/model/A_I,/model/B_E,/model/B_I', 'process')
_synh_paths = ['/model/synh_AI_EI', '/model/synh_BI_EI',
               '/model/synh_IA_to_EB', '/model/synh_IB_to_EA',
               '/model/A_adaptSyn', '/model/B_adaptSyn']
if W_EE != 0.0:
    _synh_paths += ['/model/synh_AA_EE', '/model/synh_BB_EE']
moose.useClock(0, ','.join(_synh_paths), 'process')
moose.useClock(0, '/model/A_adaptRC,/model/B_adaptRC', 'process')
moose.useClock(2, '/model/tab_A_E,/model/tab_A_I,/model/tab_B_E,/model/tab_B_I,'
                   '/model/vm_examples', 'process')

print("Reinitializing MOOSE ...")
moose.reinit()
print(f"Running {SIM_TIME:.1f} s of simulated time ...")
moose.start(SIM_TIME)
print("Done.")

# ==========================================================================
# Save raw spike data (for reproducibility / re-analysis) + make plots
# ==========================================================================
os.makedirs('output', exist_ok=True)

def get_spikes(tabs):
    return [tabs.vec[i].vector for i in range(len(tabs.vec))]

spikesA_E = get_spikes(tabA_E)
spikesA_I = get_spikes(tabA_I)
spikesB_E = get_spikes(tabB_E)
spikesB_I = get_spikes(tabB_I)

np.savez('output/spike_data.npz',
         spikesA_E=np.array(spikesA_E, dtype=object),
         spikesA_I=np.array(spikesA_I, dtype=object),
         spikesB_E=np.array(spikesB_E, dtype=object),
         spikesB_I=np.array(spikesB_I, dtype=object))

# ---- 1) Raster plot -------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                          gridspec_kw={'height_ratios': [1.4, 1]})

ax = axes[0]
row = 0
for spikes in spikesA_E:
    ax.plot(spikes, [row]*len(spikes), '.', color='tab:blue', markersize=3)
    row += 1
for spikes in spikesA_I:
    ax.plot(spikes, [row]*len(spikes), '.', color='tab:cyan', markersize=3)
    row += 1
row += 2
for spikes in spikesB_E:
    ax.plot(spikes, [row]*len(spikes), '.', color='tab:red', markersize=3)
    row += 1
for spikes in spikesB_I:
    ax.plot(spikes, [row]*len(spikes), '.', color='tab:orange', markersize=3)
    row += 1
ax.axhline(N_E - 0.5, color='k', lw=0.5, ls=':')
ax.axhline(N_POOL + N_E + 1.5, color='k', lw=0.5, ls=':')
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='.', color='w', markerfacecolor='tab:blue', label='Pool A - Excitatory', markersize=10),
    Line2D([0], [0], marker='.', color='w', markerfacecolor='tab:cyan', label='Pool A - Inhibitory', markersize=10),
    Line2D([0], [0], marker='.', color='w', markerfacecolor='tab:red', label='Pool B - Excitatory', markersize=10),
    Line2D([0], [0], marker='.', color='w', markerfacecolor='tab:orange', label='Pool B - Inhibitory', markersize=10),
]
ax.legend(handles=legend_elems, loc='upper right', fontsize=8)
ax.set_ylabel('Neuron index')
ax.set_title('Neuromorphic Oscillator - Spike Raster (100-neuron silicon budget), full run')
ax.set_xlim(0, SIM_TIME)

# Zoomed-in panel so the anti-phase switching is visible without dot-overlap
ax2 = axes[1]
row = 0
ZOOM_T = min(3.0, SIM_TIME)
for spikes in spikesA_E:
    ax2.plot(spikes, [row]*len(spikes), '.', color='tab:blue', markersize=5)
    row += 1
for spikes in spikesA_I:
    ax2.plot(spikes, [row]*len(spikes), '.', color='tab:cyan', markersize=5)
    row += 1
row += 2
for spikes in spikesB_E:
    ax2.plot(spikes, [row]*len(spikes), '.', color='tab:red', markersize=5)
    row += 1
for spikes in spikesB_I:
    ax2.plot(spikes, [row]*len(spikes), '.', color='tab:orange', markersize=5)
    row += 1
ax2.axhline(N_E - 0.5, color='k', lw=0.5, ls=':')
ax2.axhline(N_POOL + N_E + 1.5, color='k', lw=0.5, ls=':')
ax2.set_xlim(0, ZOOM_T)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Neuron index')
ax2.set_title(f'Zoomed view (0-{ZOOM_T:.0f}s): clean anti-phase switching between Pool A and Pool B')

plt.tight_layout()
plt.savefig('output/raster_plot.png', dpi=150)
plt.close()

# ---- 2) Population firing-rate traces (anti-phase pulse pattern) --------
def pop_rate(spike_lists, sim_time, dt_bin=0.02):
    bins = np.arange(0, sim_time + dt_bin, dt_bin)
    all_spikes = np.concatenate([np.asarray(s) for s in spike_lists if len(s) > 0]) \
        if any(len(s) > 0 for s in spike_lists) else np.array([])
    counts, edges = np.histogram(all_spikes, bins=bins)
    rate = counts / (len(spike_lists) * dt_bin)   # Hz per neuron
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, rate

tA, rA = pop_rate(spikesA_E, SIM_TIME)
tB, rB = pop_rate(spikesB_E, SIM_TIME)

plt.figure(figsize=(11, 4))
plt.plot(tA, rA, color='tab:blue', label='Pool A (Excitatory) rate')
plt.plot(tB, rB, color='tab:red', label='Pool B (Excitatory) rate')
plt.xlabel('Time (s)')
plt.ylabel('Population firing rate (Hz/neuron)')
plt.title('Anti-phase pulse pattern emerging from tonic bias + mutual inhibition + adaptation')
plt.legend()
plt.xlim(0, SIM_TIME)
plt.tight_layout()
plt.savefig('output/population_rate.png', dpi=150)
plt.close()

# ---- 3) Example Vm traces -------------------------------------------------
t_vm = np.arange(len(vm_tabs.vec[0].vector)) * DT
plt.figure(figsize=(11, 6))
labels = ['Pool A, E neuron #0', 'Pool A, I neuron #0',
          'Pool B, E neuron #0', 'Pool B, I neuron #0']
colors = ['tab:blue', 'tab:cyan', 'tab:red', 'tab:orange']
for k in range(4):
    plt.subplot(4, 1, k+1)
    plt.plot(t_vm, np.array(vm_tabs.vec[k].vector) * 1e3, color=colors[k], lw=0.7)
    plt.ylabel('Vm (mV)')
    plt.title(labels[k], fontsize=9)
    plt.xlim(0, SIM_TIME)
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig('output/voltage_traces.png', dpi=150)
plt.close()

print("Saved output/raster_plot.png, output/population_rate.png, "
      "output/voltage_traces.png and output/spike_data.npz")

# ---- Quantitative anti-phase quality check --------------------------------
skip_s = 1.0  # ignore initial transient
mask = tA >= skip_s
if rA[mask].std() > 1e-9 and rB[mask].std() > 1e-9:
    anti_corr = np.corrcoef(rA[mask], rB[mask])[0, 1]
else:
    anti_corr = float('nan')
meanA, meanB = rA[mask].mean(), rB[mask].mean()
balance = 1 - abs(meanA - meanB) / (meanA + meanB + 1e-9)
print(f"\nAnti-phase quality check (t > {skip_s:.1f}s):")
print(f"  Pearson correlation of Pool A vs Pool B firing rate: {anti_corr:+.3f} "
      f"(want strongly negative)")
print(f"  Mean rate Pool A = {meanA:.2f} Hz/neuron, Pool B = {meanB:.2f} Hz/neuron, "
      f"balance = {balance:.2f} (1.0 = perfectly symmetric)")
