#!/usr/bin/env python3
import numpy as np
import moose
import itertools

EL, VT, VRESET = -70e-3, -50e-3, -60e-3
RM, CM, REFRACT = 20e6, 1e-9, 2e-3
N_E, N_I = 30, 20
DT = 0.5e-3
SIM_TIME = 4.0

def run_trial(W_EE, W_EI, W_IE, ADAPT_STEP, ADAPT_TAU, IBIAS_E, IBIAS_I,
              P_EE=0.30, P_EI=0.55, P_IE=0.70, headstart=0.01e-9, seed=1):
    moose.Neutral('/model') if not moose.exists('/model') else None
    if moose.exists('/model'):
        moose.delete('/model')
    moose.Neutral('/model')
    np.random.seed(seed)
    moose.seed(seed)

    def make_pool(name, ibias_e, ibias_i):
        e = moose.LIF(f'/model/{name}_E', N_E)
        i = moose.LIF(f'/model/{name}_I', N_I)
        for pop, ib in ((e, ibias_e), (i, ibias_i)):
            pop.vec.Em = EL; pop.vec.thresh = VT; pop.vec.vReset = VRESET
            pop.vec.Rm = RM; pop.vec.Cm = CM; pop.vec.refractoryPeriod = REFRACT
            jitter = np.random.uniform(-0.03e-9, 0.03e-9, size=len(pop.vec))
            pop.vec.inject = ib + jitter
            pop.vec.initVm = np.random.uniform(EL, VT, size=len(pop.vec))
        return e, i

    eA, iA = make_pool('A', IBIAS_E + headstart, IBIAS_I)
    eB, iB = make_pool('B', IBIAS_E - headstart, IBIAS_I)

    rng = np.random.default_rng(seed)
    def connect(pre, post, weight, prob, name):
        n_pre, n_post = len(pre.vec), len(post.vec)
        synh = moose.SimpleSynHandler(f'/model/{name}', n_post)
        moose.connect(synh, 'activationOut', post, 'activation', 'OneToOne')
        for j in range(n_post):
            mask = rng.random(n_pre) < prob
            idxs = np.nonzero(mask)[0]
            if len(idxs) == 0: idxs = [rng.integers(0, n_pre)]
            synh.vec[j].numSynapses = len(idxs)
            for k, p in enumerate(idxs):
                syn = synh.vec[j].synapse[k]
                moose.connect(pre.vec[int(p)], 'spikeOut', syn, 'addSpike')
                syn.delay = 1e-3
                syn.weight = weight
        return synh

    if W_EE != 0:
        connect(eA, eA, W_EE, P_EE, 'sAA')
        connect(eB, eB, W_EE, P_EE, 'sBB')
    connect(eA, iA, W_EI, P_EI, 'sAI')
    connect(eB, iB, W_EI, P_EI, 'sBI')
    connect(iA, eB, -W_IE, P_IE, 'sIAeB')
    connect(iB, eA, -W_IE, P_IE, 'sIBeA')

    def add_adapt(e_vec, name):
        n = len(e_vec.vec)
        rc = moose.RC(f'/model/{name}_rc', n)
        rc.vec.R = 1.0; rc.vec.C = ADAPT_TAU
        synh = moose.SimpleSynHandler(f'/model/{name}_asyn', n)
        moose.connect(synh, 'activationOut', rc, 'injectIn', 'OneToOne')
        for k in range(n):
            synh.vec[k].numSynapses = 1
            syn = synh.vec[k].synapse[0]
            moose.connect(e_vec.vec[k], 'spikeOut', syn, 'addSpike')
            syn.delay = 0.0; syn.weight = ADAPT_STEP
        moose.connect(rc, 'output', e_vec, 'activation', 'OneToOne')

    add_adapt(eA, 'A')
    add_adapt(eB, 'B')

    tabA = moose.Table('/model/tabA', N_E)
    moose.connect(eA, 'spikeOut', tabA, 'input', 'OneToOne')
    tabB = moose.Table('/model/tabB', N_E)
    moose.connect(eB, 'spikeOut', tabB, 'input', 'OneToOne')

    moose.setClock(0, DT); moose.setClock(1, DT); moose.setClock(2, DT)
    moose.useClock(0, '/model/A_E,/model/A_I,/model/B_E,/model/B_I', 'init')
    moose.useClock(1, '/model/A_E,/model/A_I,/model/B_E,/model/B_I', 'process')
    kids = [k.name for k in moose.wildcardFind('/model/#[TYPE=SimpleSynHandler]')]
    path = ','.join(f'/model/{k}' for k in kids)
    moose.useClock(0, path, 'process')
    moose.useClock(0, '/model/A_rc,/model/B_rc', 'process')
    moose.useClock(2, '/model/tabA,/model/tabB', 'process')

    moose.reinit()
    moose.start(SIM_TIME)

    spikesA = [tabA.vec[i].vector for i in range(N_E)]
    spikesB = [tabB.vec[i].vector for i in range(N_E)]

    def rate(spikelists, dt_bin=0.05):
        bins = np.arange(0, SIM_TIME + dt_bin, dt_bin)
        allsp = np.concatenate([np.asarray(s) for s in spikelists if len(s) > 0]) \
            if any(len(s) > 0 for s in spikelists) else np.array([])
        counts, _ = np.histogram(allsp, bins=bins)
        return counts / (N_E * dt_bin)

    rA = rate(spikesA)
    rB = rate(spikesB)
    # skip initial transient
    skip = int(0.5 / 0.05)
    rA_, rB_ = rA[skip:], rB[skip:]
    if rA_.std() < 1e-6 or rB_.std() < 1e-6:
        corr = 0.0
    else:
        corr = np.corrcoef(rA_, rB_)[0, 1]
    meanA, meanB = rA_.mean(), rB_.mean()
    balance = 1 - abs(meanA - meanB) / (meanA + meanB + 1e-9)
    return corr, balance, meanA, meanB


if __name__ == '__main__':
    results = []
    W_EE_list = [0.0, 0.10e-3, 0.20e-3]
    W_IE_list = [1.5e-3, 2.5e-3, 3.5e-3]
    ADAPT_list = [(-0.015, 0.25), (-0.030, 0.30), (-0.05, 0.4)]
    IBIAS_E_list = [1.0e-9, 1.1e-9]

    for W_EE, W_IE, (astep, atau), ibias in itertools.product(
            W_EE_list, W_IE_list, ADAPT_list, IBIAS_E_list):
        corr, bal, mA, mB = run_trial(W_EE, 0.9e-3, W_IE, astep, atau, ibias, 0.75e-9)
        results.append((corr, bal, mA, mB, W_EE, W_IE, astep, atau, ibias))
        print(f"WEE={W_EE:.2e} WIE={W_IE:.2e} astep={astep:.3f} atau={atau:.2f} "
              f"ibiasE={ibias:.2e} -> corr={corr:+.2f} bal={bal:.2f} mA={mA:.1f} mB={mB:.1f}")

    results.sort(key=lambda r: r[0] + 0.3*r[1])  # prefer very negative corr, good balance
    print("\nBEST (most anti-phase, balanced):")
    for r in results[:8]:
        print(r)
