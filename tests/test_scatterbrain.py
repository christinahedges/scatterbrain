from scatterbrain import __version__
from scatterbrain.designmatrix import *
from scatterbrain import BackDrop
import os


def test_version():
    assert __version__ == "0.1.0"


def test_design_matrix():
    frame = xp.random.normal(size=(9, 10))
    for dm in [
        cartesian_design_matrix,
        radial_design_matrix,
        spline_design_matrix,
        strap_design_matrix,
    ]:
        A = dm(column=xp.arange(10), row=xp.arange(9))
        assert A.shape[0] == 90
        w = xp.random.normal(size=A.shape[1])
        A.dot(w)
        assert A.sigma_w_inv.shape == (A.shape[1], A.shape[1])
        assert len(A.sigma_f) == A.shape[0]
        assert len(A.prior_sigma) == A.shape[1]
        assert len(A.prior_mu) == A.shape[1]
        assert isinstance(A.join(A), dm)
        A = dm(column=xp.arange(10), row=xp.arange(9), prior_sigma=1e5)
        A.fit_frame(frame)
        A = dm(cutout_size=128)
        assert A.shape[0] == 128 ** 2


def test_backdrop_cutout():
    frames = xp.random.normal(size=(2, 128, 128)) + 100
    b = BackDrop(cutout_size=128)
    b.fit_model(frames)
    assert len(b.weights_full) == 2
    assert len(b.weights_basic) == 2
    model = b.model(0)
    assert model.shape == (128, 128)


def test_backdrop():
    frames = xp.random.normal(size=(2, 2048, 2048)) + 100
    b = BackDrop()
    b.fit_model(frames)
    assert len(b.weights_full) == 2
    assert len(b.weights_basic) == 2
    b.save("backdrop_weights.npz")
    b = BackDrop(column=xp.arange(10), row=xp.arange(9)).load("backdrop_weights.npz")
    model = b.model(0)
    assert model.shape == (9, 10)
    if os.path.exists("backdrop_weights.npz"):
        os.remove("backdrop_weights.npz")
