import os

import numpy as np

import gridtools as gt


def test_horizontal_diffusion():
    domain = (64, 64, 32)
    diffusion = np.zeros(domain)
    weight = np.ones(domain)
    data = np.zeros(domain)

    for i in range(domain[0]):
        for j in range(domain[1]):
            for k in range(domain[2]):
                data[i, j, k] = i**5 + j

    stencil = gt.NGStencil(definitions_func=definitions_horizontal_diffusion,
                           inputs={"data": data, "weight": weight},
                           outputs={"diffusion": diffusion},
                           domain=gt.domain.Rectangle((2, 2), (61, 61)),
                           mode=gt.mode.SUPERHACK)
    stencil.compute()

    # Results comparison
    expected_data_file = os.path.dirname(os.path.abspath(__file__)) + "/horizontaldiffusion_result.npy"
    expected = np.load(expected_data_file)
    assert np.allclose(diffusion[2:61, 2:61], expected[2:61, 2:61], atol=1e-12)


def definitions_horizontal_diffusion(data, weight):
    i = gt.Index()
    j = gt.Index()

    laplacian = gt.Equation()
    flux_i = gt.Equation()
    flux_j = gt.Equation()
    diffusion = gt.Equation()

    alpha = -4.0
    laplacian[i, j] = alpha * data[i, j] + (data[i-1, j] + data[i+1, j] + data[i, j-1] + data[i, j+1])
    flux_i[i, j] = laplacian[i+1, j] - laplacian[i, j]
    flux_j[i, j] = laplacian[i, j+1] - laplacian[i, j]

    diffusion[i, j] = weight[i, j] * (flux_i[i-1, j] - flux_i[i, j] + flux_j[i, j-1] - flux_j[i, j])

    return diffusion
