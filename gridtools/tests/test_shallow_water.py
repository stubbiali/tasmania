import os

import numpy as np

import gridtools as gt


def test_shallow_water():
    def droplet(H, domain, val=1.0):
        """
        A two-dimensional falling drop into the water:

            H   the water height field.-
        """
        x, y = np.mgrid[:domain[0], :domain[1]]
        droplet_x, droplet_y = domain[0]/2, domain[1]/2
        rr = (x-droplet_x)**2 + (y-droplet_y)**2
        H[rr < (domain[0]/10.0)**2] = val

    domain = (64, 64, 1)

    dt = 0.05

    out_H  = np.zeros(domain, order='F')
    out_H += 0.000001
    out_U  = np.zeros(domain, order='F')
    out_U += 0.000001
    out_V  = np.zeros(domain, order='F')
    out_V += 0.000001

    droplet(out_H, domain)

    in_H = np.copy(out_H)
    in_U = np.copy(out_U)
    in_V = np.copy(out_V)

    stencil = gt.NGStencil(definitions_func=definitions_sw,
                           inputs={"in_U": in_U, "in_V": in_V, "in_H": in_H},
                           outputs={"out_H": out_H, "out_U": out_U, "out_V": out_V},
                           domain=gt.domain.Rectangle((1, 1), (62, 62)),
                           mode=gt.mode.ALPHA)
    stencil.compute()

    # Results comparison
    expected_data_file = os.path.dirname(os.path.abspath(__file__)) + "/sw_001.npy"
    expected = np.load(expected_data_file)
    assert np.allclose(out_H[1:62, 1:62], expected[1:62, 1:62], atol=1e-12)


def definitions_sw(in_U, in_V, in_H):
    i = gt.Index()
    j = gt.Index()

    # Constants
    bl = 0.2
    growth = 1.2
    dt = 0.05

    # Outputs
    out_U = gt.Equation()
    out_V = gt.Equation()
    out_H = gt.Equation()

    # Temporaries
    Hd = gt.Equation()
    Ud = gt.Equation()
    Vd = gt.Equation()
    Hx = gt.Equation()
    Ux = gt.Equation()
    Vx = gt.Equation()
    Hy = gt.Equation()
    Uy = gt.Equation()
    Vy = gt.Equation()

    Dh = gt.Equation()
    Du = gt.Equation()
    Dv = gt.Equation()

    # U Momentum
    Ux[i, j] = in_U[i+1, j] - in_U[i-1, j]
    Uy[i, j] = in_U[i, j+1] - in_U[i, j-1]
    Ud[i, j] = in_U[i, j] * (1.0 - bl) + bl * (0.25 * (in_U[i-1, j] + in_U[i+1, j] + in_U[i, j+1] + in_U[i, j-1]))

    # V Momentum
    Vx[i, j] = in_V[i+1, j] - in_V[i-1, j]
    Vy[i, j] = in_V[i, j+1] - in_V[i, j-1]
    Vd[i, j] = in_V[i,j] * (1.0 - bl) + bl * (0.25 * (in_V[i-1, j] + in_V[i+1, j] + in_V[i, j+1] + in_V[i, j-1]))

    # H Momentum
    Hx[i, j] = in_H[i+1, j] - in_H[i-1, j]
    Hy[i, j] = in_H[i, j+1] - in_H[i, j-1]
    Hd[i, j] = in_H[i, j] * (1.0 - bl) + bl * (0.25 * (in_H[i-1, j] + in_H[i+1, j] + in_H[i, j+1] + in_H[i, j-1]))

    # Dynamics
    Dh[i, j] = -1.0 * Ud[i, j] * Hx[i, j] - Vd[i, j] * Hy[i, j] - Hd[i, j] * (Ux[i, j] + Vy[i, j])
    Du[i, j] = -1.0 * Ud[i, j] * Ux[i, j] - Vd[i, j] * Uy[i, j] - growth * Hx[i, j]
    Dv[i, j] = -1.0 * Ud[i, j] * Vx[i, j] - Vd[i, j] * Vy[i, j] - growth * Hy[i, j]

    # Combine momentum and dynamics with a first-order Euler step
    out_H[i, j] = Hd[i, j] + dt * Dh[i, j]
    out_U[i, j] = Ud[i, j] + dt * Du[i, j]
    out_V[i, j] = Vd[i, j] + dt * Dv[i, j]

    return out_H, out_U, out_V
