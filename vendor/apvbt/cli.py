"""Command-line interface for APVBT.

This module provides the CLI for running the full pipeline:
data loading, training, simulation, SBI, and evaluation.
"""

import numpy as np
import pickle
import tqdm
import typed_argparse as tap

from .data import XCode
from .utils import load_pkl, triu_to_mat, all_conf_rates, apply
from .dynamics import DynaModel
from .dynamics.hopf import hopf_dfun
from .inference import to_torch, run_sbi, posterior_diags


# ============================================================================
# Argument Classes
# ============================================================================

class DataArgs(tap.TypedArgs):
    input: str = tap.arg(help="input fname")
    pkl: str = tap.arg(help="output pkl fname")
    hcp: bool = tap.arg(default=False, help='is hcp dataset?')
    combine: bool = tap.arg(default=False, help='combine datasets')
    skip_parc: str = tap.arg(default='', help='parcellations to skip')


class TrainArgs(tap.TypedArgs):
    data: str = tap.arg(help="data to use: hcp, 1kb or both")
    arch: int = tap.arg(default=8, help="Architecture: latent dimension")
    niter: int = tap.arg(default=100, help="Number of optimization iterations")
    out: str = tap.arg(help="Output filename")
    mb: int = tap.arg(default=64, help="mini-batch size")


class HopfTestArgs(tap.TypedArgs):
    pass


class HopfSampleArgs(tap.TypedArgs):
    data: str = tap.arg('-d', default='both.pkl')
    seed: int = tap.arg('-s', default=42)
    parc: str = tap.arg('-p', default='079-Shen2013')
    arch: int = tap.arg('-a', default=10)
    num_batch: int = tap.arg('-n', default=1)
    batch_size: int = tap.arg('-b', default=8)
    use_pmap: bool = tap.arg(default=False, help='use all cores')
    out_npz: str = tap.arg('-o')
    per_subj: bool = tap.arg(default=False, help='per subj sampling')


class SBIArgs(tap.TypedArgs):
    samples: str = tap.arg('-s', help='samples file')
    sbi_pkl: str = tap.arg('-o', help='pickle file to store sbi posterior')


class EvalSBIArgs(tap.TypedArgs):
    pass


class DownloadDataArgs(tap.TypedArgs):
    pass


class DebugArgs(tap.TypedArgs):
    show_crs: bool = tap.arg(default=False)


# ============================================================================
# CLI Functions
# ============================================================================

def run_data(args: DataArgs) -> None:
    """Load and format data."""
    print(args)
    if args.combine:
        pkl_hcp, pkl_1kb = args.input.split(',')
        xc_hcp = XCode.from_pkl(pkl_hcp)
        xc_1kb = XCode.from_pkl(pkl_1kb)
        XCode.combine_xc(xc_hcp, xc_1kb).to_pkl(args.pkl)
    elif args.input and args.pkl:
        XCode.from_kg(args.input, hcp=args.hcp).to_pkl(args.pkl)
    else:
        print('unhandled case', args)


def run_train(args: TrainArgs) -> None:
    """Train cross-coder with specified architecture."""
    # Import crosscoder to add training methods to XCode
    from . import crosscoder

    print(args)
    xc: XCode = XCode.from_pkl(args.data)
    trace, wbs, crs_test = xc.train(args.arch, niter=args.niter, mb=args.mb)
    print('trained confusion rate: ', crs_test)
    import pylab as pl
    pl.plot(trace)
    pl.savefig(f'{args.out}.pdf')
    pl.close()
    xc.to_pkl(args.out)


def run_hopf_test(args: HopfTestArgs) -> None:
    """Test that Hopf simulation is identifying for connectomes in a particular regime."""
    # Import crosscoder to add methods to XCode
    from . import crosscoder

    import pylab as pl
    import jax, jax.numpy as jp, vbjax as vb

    xc = XCode.from_pkl('both.pkl')
    parc = '079-Shen2013'
    arch, *_ = xc.arch

    # setup ground truth
    w = xc.get_conn(parc)  # (n_test, n_parc, n_parc)
    w = w[:8]
    assert w.shape == (8, 79, 79)
    u = xc.encode_conn(arch, parc)
    u = u[:8]
    k = 0.2 + vb.rand(len(w))*0.2
    D = 0.2 + vb.rand(len(w))*0.2
    theta = jp.concat([u, jp.c_[k, D]], axis=1)

    key = vb.keys[0]
    k = 0.01
    D = 0.1
    dt = 0.02

    # run sim w/ fc as feature
    ti, tj = jp.triu_indices(79, k=1)
    def features(x):
        return jp.corrcoef(x[500:, 0].T)[ti, tj]

    model = DynaModel('hopf', hopf_dfun, features, dt=dt)
    fcs = []
    for i in range(8):
        fc1 = model.run_w(w[i], k, D, nwin=10, key=vb.keys[0]).mean(axis=0)
        fc2 = model.run_w(w[i], k, D, nwin=10, key=vb.keys[1]).mean(axis=0)
        assert fc1.shape == ti.shape
        fcs.append((fc1, fc2))

    sim = np.zeros((8,8))
    for i in range(8):
        fi = fcs[i][0].reshape(-1)
        for j in range(8):
            fj = fcs[j][1].reshape(-1)
            sim[i, j] = np.sum(np.square(fi - fj))

    np.testing.assert_array_equal(
        np.argmin(sim, axis=1), np.r_[:len(w)])
    # pl.imshow(-sim)
    # pl.show()


def run_hopf_sample(args: HopfSampleArgs) -> None:
    """Sample Hopf dynamics simulations for SBI training."""
    # Import crosscoder to add methods to XCode
    from . import crosscoder

    print(args)
    import pylab as pl
    import jax, jax.numpy as jp, vbjax as vb

    # load data
    xc = XCode.from_pkl(args.data)
    nreg = int(args.parc.split('-')[0])
    iparc = xc.parcs.index(args.parc)
    test_ws = triu_to_mat(xc.conns[iparc][xc.tts:] + xc.means[iparc])

    # simulation & features
    ti, tj = jp.triu_indices(nreg, k=1)
    def features(x):
        return jp.corrcoef(x[500:, 0].T)[ti, tj]
    model = DynaModel('hopf', hopf_dfun, features, dt=0.02)
    def f(w, k, D, key):
        return model.run_w(w, k, D, nwin=10, key=key).mean(axis=0)
    f = jax.vmap(f)
    if args.use_pmap:
        f = jax.pmap(f)
    else:
        f = jax.jit(f)

    # data, parameters
    B, A = args.batch_size, args.arch
    if A not in xc.arch:
        A, *_ = xc.arch
    mvn = xc.calc_mvn(A)
    keys = jax.random.split(
        jax.random.PRNGKey(args.seed), (args.num_batch, 3 + B))
    if args.per_subj:
        theta = np.zeros((args.num_batch, B, 2), 'f')
    else:
        theta = np.zeros((args.num_batch, B, A + 2), 'f')
    feats = np.zeros((args.num_batch, B, ti.size), 'f')
    i_test_ws = np.zeros((args.num_batch, B), dtype=np.uint16)
    for i_batch in tqdm.trange(args.num_batch):
        key = keys[i_batch]
        if args.per_subj: # sample test ws
            i_test_ws[i_batch] = jax.random.randint(
                key[0], (B, ), minval=0, maxval=test_ws.shape[0])
            w = test_ws[i_test_ws[i_batch]]
        else: # sampling cohort prior over u
            u = mvn.sample(B)
            w = xc.decode_conn(args.parc, u)
        # k = 0.01 + vb.rand(B, key=key[1]) * 0.01
        # D = 0.1 + vb.rand(B, key=key[2]) * 0.1
        lk = jax.random.normal(key[1], (B, )) - 6.0
        lD = jax.random.normal(key[2], (B, )) - 1.5
        k = jp.exp(lk)
        D = jp.exp(lD)
        if args.use_pmap:
            # reshape leading
            rl = lambda a: a.reshape((vb.cores, -1) + a.shape[1:])
            fc = f(*(rl(_) for _ in (w, k, D, key[3:])))
            fc = fc.reshape((B,) + fc.shape[2:])
        else:
            fc = f(w, k, D, key[3:])
        fc.block_until_ready()
        # save results
        if args.per_subj:
            theta[i_batch, :, 0] = lk
            theta[i_batch, :, 1] = lD
        else:
            theta[i_batch, :, :A] = u
            theta[i_batch, :, A] = lk
            theta[i_batch, :, A+1] = lD
        feats[i_batch] = fc

    print(f'saving to {args.out_npz}')
    outputs = dict(theta=theta, feats=feats, arch=A, parc=args.parc)
    if args.per_subj:
        outputs['i_test_ws'] = i_test_ws
        outputs['test_ws'] = test_ws
    np.savez(args.out_npz, **outputs)


def run_sbi_args(args: SBIArgs):
    """Train SBI posterior from sampled simulations."""
    npz = np.load(args.samples, allow_pickle=True)
    theta = npz['theta']  # u, k, D
    feats = npz['feats']  # fc
    theta = theta.reshape(-1, theta.shape[-1])
    feats = feats.reshape(-1, feats.shape[-1])
    assert theta.shape[0] == feats.shape[0]
    per_subj = 'i_test_ws' in npz
    if per_subj:
        print(f'per subject SBI w/ theta {theta.shape}, features {feats.shape}; saving to {args.sbi_pkl}')
        # run_sbi(theta, feats, fname=args.sbi_pkl)
        iw = npz['i_test_ws'].reshape(-1)
        tw = npz['test_ws']
        uiw = np.unique(iw)
        print(iw.shape, tw.shape, feats.shape, theta.shape)
        # (32768,) (231, 79, 79) (32768, 3081) (32768, 2)
        subj_posts = []
        for subj in tqdm.tqdm(uiw):
            subj_mask = iw == subj
            subj_post = run_sbi(theta[subj_mask], feats[subj_mask], prog=False)
            subj_posts.append(subj_post)
        with open(args.sbi_pkl, 'wb') as fd:
            pickle.dump(subj_posts, fd)
    else:
        print(f'cohort SBI w/ theta {theta.shape}, features {feats.shape}; saving to {args.sbi_pkl}')
        run_sbi(theta, feats, fname=args.sbi_pkl)


def run_eval(args: EvalSBIArgs):
    """Evaluate SBI results comparing cohort vs subject-level."""
    # Import crosscoder to add methods to XCode
    from . import crosscoder

    import sbi
    from sbi.inference.posteriors import DirectPosterior
    from typing import List
    # read files, names in run.sh TODO args
    cohort_samp = np.load('cohort-samples.npz', allow_pickle=True)
    cohort_post: DirectPosterior = load_pkl('cohort-posterior.pkl')
    subj_samp = np.load('subj-samples.npz', allow_pickle=True)
    subj_post: List[DirectPosterior] = load_pkl('subj-posterior.pkl')
    # extract relevant arrays
    iw = subj_samp['i_test_ws'].reshape(-1)
    tw = subj_samp['test_ws']
    uf = lambda a: a.reshape((-1,) + a.shape[2:])
    s_theta = uf(subj_samp['theta'])
    s_feats = uf(subj_samp['feats'])
    c_theta = uf(cohort_samp['theta'])
    c_feats = uf(cohort_samp['feats'])
    arch = cohort_samp['arch']
    # go over each subject, apply cohort posterior to the subject
    # and make two comparisons:
    # 1. is the subject connectome recovered?
    # 2. how well the per-subject parameters recovered?
    for subj in np.unique(iw): #tqdm.tqdm(np.unique(iw)):
        mask = iw == subj
        _sample = lambda post: post.sample_batched(
            (200,), to_torch(s_feats[mask]), show_progress_bars=False)
        try:
            cp_theta = _sample(cohort_post)
            sp_theta = _sample(subj_post[subj])  # (200, 75, arch + 2)
        except RuntimeError:
            print(subj, 'fail')
            continue
        import pylab as pl
        # subject sbi
        s, sz, sci90 = posterior_diags(p_us=s_theta, po_us=sp_theta, true_us=s_theta[mask])
        s, cz, cci90 = posterior_diags(
            p_us=s_theta, po_us=cp_theta[..., arch:], true_us=s_theta[mask])
        print(subj, mask.sum(), 'z', sz.mean(), np.quantile(cz, 0.5),
              sci90.mean()*100, cci90.mean()*100, "% ok")

        pl.figure()
        for i in range(25):
            pl.subplot(5, 5, i + 1)
            pl.hist(s_theta[:, 0], alpha=0.2, density=True, label='prior', log=True)
            pl.hist(cp_theta[:, i, arch], alpha=0.5, density=True, label='cohort', log=True)
            pl.hist(sp_theta[:, i, 0], alpha=0.5, density=True, label='subject', log=True)
            pl.axvline(s_theta[mask][i, 0], color='r', label='true')
            # pl.legend()
        pl.tight_layout()
        pl.show()
        1/0


def run_dl(args: DownloadDataArgs):
    """Download HCP and 1000 Brains datasets from Knowledge Graph."""
    XCode._download_kg_zip('hcp.zip', hcp=True)
    XCode._download_kg_zip('1kb.zip', hcp=False)


def run_debug(args: DebugArgs):
    """Debug mode for testing regime selection and SBI."""
    # Import crosscoder to add methods to XCode
    from . import crosscoder

    print('\n\n' + 'DEBUG ' * 10)
    print(args)
    import pylab as pl
    import jax, jax.numpy as jp, vbjax as vb
    import sbi

    # load data
    xc = XCode.from_pkl('both.pkl')

    if args.show_crs:
        c = [_[xc.tts:] for _ in xc.conns]
        crs = all_conf_rates(xc.wbs[0], c)
        pl.imshow(crs, vmin=0, vmax=1.0); pl.colorbar()
        pl.show()

    parc = '079-Shen2013'
    iparc = xc.parcs.index(parc)
    arch = A = 16
    assert arch in xc.arch
    nreg = int(parc.split('-')[0])

    ti, tj = jp.triu_indices(nreg, k=1)
    def features(x):
        return jp.corrcoef(x[500:, 0].T)[ti, tj]
    model = DynaModel('hopf', hopf_dfun, features, dt=0.02)
    def f(w, k, D, key):
        return model.run_w(w, k, D, nwin=10, key=key).mean(axis=0)

    mvn = xc.calc_mvn(A)
    ng = 64
    k, D = jp.exp(jp.mgrid[-8:-4:1j*ng, -2:1:1j*ng])
    k, D = k.reshape(-1), D.reshape(-1)
    print('k', k.min(), k.max())
    print('D', D.min(), D.max())
    u = mvn.sample(k.size)
    w = xc.decode_conn(parc, u)
    keys = jax.random.split(vb.key, k.size)
    x = apply(f, w, k, D, keys, B=ng)
    print(x.shape)

    # post = load_pkl('cohort-posterior.pkl')
    lk, lD = np.log(k), np.log(D)
    samp = np.c_[u, lk, lD]
    post = run_sbi(samp, x)
    theta_hat = post.sample_batched((200, ), to_torch(x))
    print(theta_hat.shape)  # (200, 64, 18)

    s, z, ok90 = posterior_diags(samp, theta_hat, samp)
    print(s.shape)

    pl.figure()
    for i in range(s.shape[-1]):
        pl.subplot(5, 4, i + 1)
        s, z, ok90 = posterior_diags(samp[:, i], theta_hat[:, :, i], samp[:, i])
        pl.plot(s, z, 'x')
        print(i, ok90.mean())

    pl.figure()
    # i_par = 0
    for i_par in range(18):
        pl.subplot(5, 4, i_par + 1)
        i_samp = 0
        pl.hist(samp[:, i_par], alpha=0.5)
        pl.hist(theta_hat[:, i_samp, i_par], alpha=0.5)
        pl.axvline(samp[i_samp, i_par], color='r')

    pl.show()

    print('\n'*3, 'DONE '*10)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main CLI entry point."""
    tap.Parser(
        tap.SubParserGroup(
            tap.SubParser('data', DataArgs, help="Load and format data"),
            tap.SubParser('train', TrainArgs, help="Train cross-coder"),
            tap.SubParser('hopf_test', HopfTestArgs, help="Test Hopf regime identification"),
            tap.SubParser('hopf_sample', HopfSampleArgs, help="Sample Hopf model simulations"),
            tap.SubParser('sbi', SBIArgs, help="Run SBI training"),
            tap.SubParser('eval', EvalSBIArgs, help="Evaluate SBI results"),
            tap.SubParser('download', DownloadDataArgs, help="Download datasets from Knowledge Graph"),
            tap.SubParser('dbg', DebugArgs, help="Debug mode"),
        )
    ).bind(
        run_data,
        run_train,
        run_hopf_test,
        run_hopf_sample,
        run_sbi_args,
        run_eval,
        run_dl,
        run_debug,
    ).run()
