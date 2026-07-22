# Extracted from apvbt (https://github.com/ins-amu/apvbt) — see vendor/README.md
"""Cross-coder training and encoding/decoding operations.

This module extends the XCode class with training, encoding, and decoding functionality.
"""

import numpy as np
import tqdm
from .data import XCode
from .utils import small, MvNorm, triu_to_mat, all_conf_rates


# Extend XCode class with training and encoding methods
def make_wbs(self, nlat):
    """Initialize encoder-decoder weights for given latent dimension.

    Args:
        nlat: Latent space dimension

    Returns:
        List of ((enc_w, enc_b), (dec_w, dec_b)) tuples per parcellation
    """
    wbs = []
    for c in self.conns:
        n = c.shape[1]
        w1, w2t = small(2, n, nlat)
        b1 = small(nlat)
        b2 = small(n)
        wb = (w1, b1), (w2t.T, b2)  # enc, dec
        wbs.append(wb)
    return wbs


def make_loss(self):
    """Create loss and gradient functions for cross-coder training.

    Loss is mean squared error across all encode-decode pairs between parcellations.

    Returns:
        Tuple of (loss_fn, grad_fn) that are JIT-compiled
    """
    import jax, jax.numpy as jp
    def loss(wbs, conns):
        ll = 0
        for ((ew, eb), _), c_e in zip(wbs, conns):
            u = c_e @ ew + eb  # encode from parc i
            for (_, (dw, db)), c_d in zip(wbs, conns):
                v = u @ dw + db  # decode to parc j
                ll = ll + jp.mean((v - c_d)**2)
        return ll
    loss = jax.jit(loss)
    grad = jax.jit(jax.grad(loss))
    return loss, grad


def train(self, nlat, lr=3e-4, niter=500, nlog=None, tts=None, mb=64,
          all_conf_rates_fn=all_conf_rates):
    """Train cross-coder with specified latent dimension.

    Args:
        nlat: Latent space dimension
        lr: Learning rate for Adam optimizer
        niter: Number of training iterations
        nlog: Logging frequency (default: niter)
        tts: Train/test split (default: self.tts)
        mb: Mini-batch size
        all_conf_rates_fn: Function to compute confusion rates

    Returns:
        Tuple of (trace, wbs, confusion_rate)
        - trace: List of (train_loss, test_loss) per iteration
        - wbs: Trained weights
        - confusion_rate: Mean confusion rate on test set
    """
    import jax, jax.numpy as jp
    from jax.example_libraries import optimizers
    tts = tts or self.tts
    mbkey = jax.random.PRNGKey(mb)
    train_conns = [_[:tts] for _ in self.conns]
    test_conns = [_[tts:] for _ in self.conns]
    trace = []
    opt_init, opt_update, get_params = optimizers.adam(lr)
    wbs = self.make_wbs(nlat)
    opt_state = opt_init(wbs)
    nlog = nlog or niter
    loss, grad = self.make_loss()
    for i in (pbar := tqdm.trange(niter+1)):
        mbkey, _key = jax.random.split(mbkey)
        imb = jax.random.randint(mbkey, (mb,), 0, tts)
        mb_conns = [_[imb] for _ in train_conns]
        ll_train = np.log(loss(wbs, mb_conns))
        ll_test = np.log(loss(wbs, test_conns))
        trace.append((ll_train, ll_test))
        pbar.set_description(f'-ll {trace[0][1] - ll_test:0.3f}')
        wbs = get_params(opt_state)
        opt_state = opt_update(i, grad(wbs, mb_conns), opt_state)
    crs_test = all_conf_rates_fn(wbs, test_conns).mean()
    self.wbs.append(wbs)
    return trace, wbs, crs_test


@property
def arch(self):
    """Get list of trained latent dimensions (architectures).

    Returns:
        List of latent dimensions for each trained architecture
    """
    # all_wb1[arch][conn][e,d][w,b]
    arch = [b.size for (((_, b), _), *_) in self.wbs]
    return arch


def calc_mvn(self, arch=None, tts=None):
    """Calculate multivariate normal distribution in latent space.

    Args:
        arch: Latent dimension to use (default: first architecture)
        tts: Train/test split (default: self.tts)

    Returns:
        MvNorm instance with latent codes, mean, and covariance
    """
    import jax.numpy as jp
    iarch = 0 if arch is None else self.arch.index(arch)
    tts = tts or self.tts
    us = []
    for ((ew, eb), _), c in zip(self.wbs[iarch], self.conns):
        us.append(c[tts:] @ ew + eb)
    us = jp.array(us)
    us_ = us.reshape(-1, us.shape[-1])
    u_mu = us_.mean(axis=0)
    u_cov = jp.cov(us_.T)
    return MvNorm(us, u_mu, u_cov)


def get_triu(self, parc, tts=None):
    """Get upper triangular connectome data for parcellation.

    Args:
        parc: Parcellation name
        tts: Train/test split (default: self.tts)

    Returns:
        Array of triu connectomes for test set
    """
    tts = tts or self.tts
    iparc = self.parcs.index(parc)
    return self.conns[iparc][self.tts:]


def get_conn(self, parc, tts=None):
    """Get full connectome matrices for parcellation.

    Args:
        parc: Parcellation name
        tts: Train/test split (default: self.tts)

    Returns:
        Array of full symmetric connectome matrices
    """
    iparc = self.parcs.index(parc)
    c_ = self.get_triu(parc, tts) + self.means[iparc]
    return triu_to_mat(c_)


def decode_conn(self, parc, us):
    """Decode latent codes to connectomes for specified parcellation.

    Args:
        parc: Parcellation name
        us: Latent codes (n_samples, n_latent)

    Returns:
        Decoded connectome matrices (n_samples, n_roi, n_roi)
    """
    iarch = self.arch.index(us.shape[1])
    iparc = self.parcs.index(parc)
    _, (w, b) = self.wbs[iarch][iparc]
    mean = self.means[iparc]
    c_ = us @ w + b + mean
    return triu_to_mat(c_)


def encode_conn(self, arch, parc, tts=None):
    """Encode connectomes to latent space for specified parcellation.

    Args:
        arch: Latent dimension
        parc: Parcellation name
        tts: Train/test split (default: self.tts)

    Returns:
        Encoded latent vectors (n_samples, n_latent)
    """
    iarch = self.arch.index(arch)
    iparc = self.parcs.index(parc)
    c_ = self.get_triu(parc, tts)
    (w, b), _ = self.wbs[iarch][iparc]
    return c_ @ w + b


# Add methods to XCode class
XCode.make_wbs = make_wbs
XCode.make_loss = make_loss
XCode.train = train
XCode.arch = arch
XCode.calc_mvn = calc_mvn
XCode.get_triu = get_triu
XCode.get_conn = get_conn
XCode.decode_conn = decode_conn
XCode.encode_conn = encode_conn
