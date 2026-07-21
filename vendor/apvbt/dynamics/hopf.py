
def hopf_dfun(ys, p):
    import vbjax as vb, jax.numpy as jp
    y0, y1 = ys
    cfun = vb.make_diff_cfun(jp.array(p[-1]))
    Ic = cfun(y0), # cfun(y1)
    # dy0 = y0 (eta - y0^2 - y1^2) - omega*y1
    return jp.array([y0 * (p[2]-y0**2-y1**2) - p[3]*y1 + 100.*p[0]*Ic[0],
                     y1 * (p[2]-y0**2-y1**2) + p[3]*y0  # + 100.*p[0]*Ic[1]
                 ])


# hopf without averaging and applies bold
# TODO : merge with hopf
def make_bold_hopf(features=None, key=None, with_bold=True):
    import jax, jax.numpy as jp, vbjax as vb
    key = key or jax.random.PRNGKey(42)

    features = features or (lambda x: x[:, 0].var(axis=0))
    hyper_key = key

    def hopf_dfun(ys, p):
        y0, y1 = ys
        cfun = vb.make_diff_cfun(jp.array(p[4]))
        Ic = cfun(y0), cfun(y1)
        return jp.array([y0 * (p[0]-y0**2-y1**2) - p[1]*y1 + p[2]*Ic[0],
                         y1 * (p[0]-y0**2-y1**2) + p[1]*y0 + p[2]*Ic[1]])

    def g(x, p): return p[3]
    dt = 1e-3  # needed for num stability
    _, loop = vb.make_sde(dt, hopf_dfun, g)

    def run_w(w, key, k=0.15, D=4e-1):
        w = w / w.max()
        # eta and omega are not inferred but have to be != for each node
        eta = -1. + vb.random.normal(hyper_key, shape=(w.shape[1],))
        omega = 2.*jp.pi*jax.random.uniform(hyper_key, shape=(w.shape[1],),
                                            minval=0.02, maxval=0.04)
        p = eta, omega, 100.*k, D, w
        n = 1000
        if with_bold:
            n *= 100
            z = vb.randn(n, 2, w.shape[0], key=key)
            x0 = jp.zeros((2, w.shape[0])) + jp.c_[0., 0.].T
            x = loop(x0, z, p)
            bold_buf, bold_step, bold_sample = vb.make_bold(shape=(w.shape[0],),
                                                            dt=dt,
                                                            p=vb.bold_default_theta)
            bold_sample = vb.make_offline(
                step_fn=bold_step, sample_fn=bold_sample)
            windowed_r = x[:, 0].reshape(
                (-1, 200, w.shape[0]))  # len(bold)=500
            bold_buf, bold = jax.lax.scan(bold_sample, bold_buf, windowed_r)
            xf = features(bold)
        else:
            z = vb.randn(n, 2, w.shape[0], key=key)
            x0 = jp.zeros((2, w.shape[0])) + jp.c_[0., 0.].T
            x = loop(x0, z, p)
            xf = features(x)

        return xf

    def run_ws(w, k, D, key=hyper_key, use_pmap=True):
        def f(w, key, k, D): return run_w(w, key, k, D)
        if use_pmap:
            w_ = w.reshape((vb.cores, -1,) + w.shape[1:])
            keys_ = jax.random.split(key, w_.shape[:2])
            k_ = k.reshape(w_.shape[:2])
            D_ = D.reshape(w_.shape[:2])
            xf = jax.pmap(jax.vmap(f))(w_, keys_, k_, D_)
            xf = xf.reshape((-1,) + xf.shape[2:])
        else:
            keys = jax.random.split(key, w.shape[0])
            xf = jax.jit(jax.vmap(f))(w, keys, k, D)
        return xf

    return run_ws


