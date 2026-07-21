

def mpr_dfun(ys, p):
    import vbjax as vb
    return vb.mpr_dfun(ys, (p[0]*p[-1]@ys[0], 0), vb.mpr_default_theta)

