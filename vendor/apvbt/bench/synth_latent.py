import numpy as np
import jax, jax.numpy as jp, vbjax as vb, tqdm, torch
import aaicd
import apvbt.dynamics as ad, apvbt.inference as ai
import jax
import json

with open(f'k_per_parc.json', 'r') as f:
    k_per_parc = json.load(f)

def run(xc, mvn, parc, num_samp=8192, algo='maf'):

    # load xcoder & connectivity
    # xc = aaicd.XCode.from_pkl('both.pkl')
    # parc = '150-Destrieux'
    # arch = xc.arch[0]
    iparc = xc.parcs.index(parc)
    ws = aaicd.triu_to_mat(xc.means[iparc] + xc.conns[iparc])[:xc.tts]
    nn = ws.shape[-1]
    print(parc, ws.shape, end=' ')

    # dynamical regime
    opt = np.array(k_per_parc[parc])
    k = opt[np.argmax(opt[:,1]), 0]
    print(f'k={k:0.2f}', end=' ')
    D = 0.2
    nwin = 100
    key = jax.random.PRNGKey(42)

    # per-validation-subject ground truth w/ own connectome
    model = ad.DynaModel('mpr', ad.mpr.mpr_dfun, lambda x: x[:, 1].mean(axis=0), dt=0.01)
    xfs_iid = jax.vmap(lambda w: model.run_w(w, k=k, D=D, nwin=nwin, key=key))(ws)
    keys = jax.random.split(key, ws.shape[0])
    # xfs = jax.vmap(lambda w, key: model.run_w(w, k=k, D=D, nwin=nwin, key=key))(ws, keys)
    xfs_iid = xfs_iid[:, nwin//2:]
    # xfs = xfs[:, nwin//2:]
    print('.', end='')

    # latent prior, samples
    # mvn = xc.calc_mvn()
    new_us = mvn.sample(num_samp)
    new_ws = xc.decode_conn(parc, new_us)
    print('.', end='')

    # run model on synth conns
    model = ad.DynaModel('mpr', ad.mpr.mpr_dfun, lambda x: x[:, 1].mean(axis=0), dt=0.01)
    run = jax.jit(jax.vmap(lambda w: model.run_w(w, k=k, D=D, nwin=nwin, key=key)))
    xfs_syn = []
    for new_ws_batch in new_ws.reshape(-1, 512, nn, nn):
        xfs_syn.append(run(new_ws_batch))
    xfs_syn = jp.array(xfs_syn).reshape(-1, nwin, nn)
    xfs_syn = xfs_syn[:, nwin//2:]
    print('.', end='')

    # train sbi
    poco = ai.run_sbi(new_us, xfs_syn.mean(axis=1), prog=False, algo=algo)
    print('.', end='')

    # infer/eval
    tru_us = xc.encode_conn(16, parc)  # NB: only the 
    diags = []
    for tru_u, xfi in zip(tru_us, xfs_iid):
        po_us = poco.sample((200, ), np.array(xfi.mean(axis=0)), show_progress_bars=False)
        diags.append(ai.posterior_diags(new_us, po_us, tru_u))
    s, z, _ = np.array(diags).transpose(1, 0, 2)
    ok = ((s>0)*(z<1.96)).mean() * 100
    szmean = np.r_[np.percentile(s.ravel(), 50), np.percentile(z.ravel(), 50)]
    print(f' {ok:0.1f}% OK, median s={szmean[0]:0.2f}, z={szmean[1]:0.2f}')