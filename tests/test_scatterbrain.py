import os

from scatterbrain import PACKAGEDIR, BackDrop, __version__
from scatterbrain.cupy_numpy_imports import fitsio, load_image, np, xp
from scatterbrain.designmatrix import (
    cartesian_design_matrix,
    radial_design_matrix,
    spline_design_matrix,
    strap_design_matrix,
)


def test_version():
    assert __version__ == "0.1.0"


def test_design_matrix():
    frame = xp.random.normal(size=(9, 10))
    cube = xp.asarray([xp.random.normal(size=(9, 10))])
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
        A.fit_batch(cube)
        A = dm(cutout_size=128)
        assert A.shape[0] == 128 ** 2


def test_backdrop_cutout():
    fname = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/data/tempffi.fits"
    f = fitsio.read(fname).astype(xp.float32)[:128, 45 : 128 + 45]
    frames = xp.asarray([f, f], dtype=xp.float32)
    b = BackDrop(1, 1, 1, cutout_size=128)
    b.fit_model(frames)
    b.fit_model_batched(frames, batch_size=2)
    assert len(b.weights_full) == 2
    assert len(b.weights_basic) == 2
    model = b.model(0)
    assert model.shape == (128, 128)
    assert np.isfinite(b.average_frame).all()
    assert b.average_frame.shape == (128, 128)

    BackDrop.from_tess_images([fname, fname], sector=1, batch_size=2, cutout_size=128)


def test_backdrop_save():
    fname = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/data/tempffi.fits"
    f = load_image(fname)
    frames = xp.asarray([f, f], dtype=xp.float32)
    b = BackDrop(1, 1, 1)
    b.fit_model(frames)
    b.fit_model_batched(frames, batch_size=2)
    assert len(b.weights_full) == 2
    assert len(b.weights_basic) == 2
    assert np.isfinite(np.asarray(b.weights_basic)).all()
    assert np.isfinite(np.asarray(b.weights_full)).all()
    b.save(output_dir="")
    b = BackDrop(1, 1, 1, column=xp.arange(10), row=xp.arange(9)).load(
        "tessbackdrop_sector1_camera1_ccd1.fits", dir=""
    )
    model = b.model(0)
    assert model.shape == (9, 10)
    assert np.isfinite(model).all()
    b = BackDrop.from_disk(1, 1, 1, column=xp.arange(10), row=xp.arange(9), dir="")
    model = b.model(0)
    assert np.isfinite(model).all()
    assert model.shape == (9, 10)
    if os.path.exists("tessbackdrop_sector1_camera1_ccd1.fits"):
        os.remove("tessbackdrop_sector1_camera1_ccd1.fits")
