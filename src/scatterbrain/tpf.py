"""Tools to work with TPFs"""
import lightkurve as lk
import numpy as np
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy import sparse

from .background import ScatteredLightBackground


def tpf_to_lightcurve(
    tpf, aperture_mask=None, dir=None, timescale=1, npca_components=10
):
    """
    Converts a TPF to a light curve, removing TESS scattered light and detrending against
    the top PCA components on the channel.

    Will load a `ScatteredLightBackground` object to create the model.

    Parameters
    ----------
    tpf : lk.TargetPixelFile
        The TPF to convert to a light curve
    aperture_mask : Optional
        An optional aperture mask to use. You can pass `all`, `threshold`, or a
        np.ndarray of bools with shape tpf.shape[1:]
    dir : str
        Optional directory where the `ScatteredLightBackground` solutions are stored
    timescale: float
        The timescale to use for splines when detrending. By default, will use 1 day.
    npca_components: int
        The number of PCA components to detrend. Must be between 1 and 10. Using more
        components may help the fit be more accurate, but could make it less precise.

    Returns
    -------
    lc : lk.LightCurve
        The corrected light curve
    """
    npca_components = np.min([np.max([1, npca_components]), 10])
    b = ScatteredLightBackground.from_tpf(tpf, dir=dir)
    if aperture_mask is None:
        aper = tpf.create_threshold_mask(3)
    else:
        aper = aperture_mask

    lc = (tpf - b.model()).to_lightcurve(aperture_mask=aper)
    k = np.isfinite(lc.flux.value) & np.isfinite(lc.flux_err.value)

    # Do a first pass with a simple offset term to get the jitter and bkg roughly right
    bkg = lk.DesignMatrix(b.bkg_pack[k, :npca_components], name="bkg").to_sparse()
    jitter = (
        lk.DesignMatrix(b.jitter_pack[k, :npca_components], name="jitter")
        .append_constant()
        .to_sparse()
    )
    dm = lk.SparseDesignMatrixCollection([bkg, jitter])
    r = lk.RegressionCorrector(lc[k])
    r.correct(dm)
    bkg.prior_mu = r.coefficients[:npca_components]
    bkg.prior_sigma = np.abs(r.coefficients[:npca_components]) * 0.1
    jitter = lk.DesignMatrix(
        b.jitter_pack[k, :npca_components], name="jitter"
    ).to_sparse()
    jitter.prior_mu = r.coefficients[npca_components : npca_components * 2]
    jitter.prior_sigma = (
        np.abs(r.coefficients[npca_components : npca_components * 2]) * 0.1
    )

    # Estimate the priors for the spline
    model_lc = r.diagnostic_lightcurves["bkg"] + r.diagnostic_lightcurves["jitter"]
    model_lc -= np.median(model_lc.flux)
    temp_lc = lc - model_lc

    # Break anywhere where there is a significant gap in the data
    dt = np.diff(lc.time.value[k])
    breaks = list(np.where(((dt / np.median(dt)) > 4.9))[0] + 1)

    nb = int((lc.time[k].value[-1] - lc.time[k].value[0]) // timescale)
    nb = nb if nb % 2 == 1 else nb + 1
    knots = list(np.linspace(lc.time[k].value[0], lc.time.value[k][-1], nb + 2)[1:-1])

    spline = lk.designmatrix.create_sparse_spline_matrix(
        lc.time.value[k], knots=knots
    ).split(breaks)
    spline.prior_mu += temp_lc.flux.value.mean()
    _, _, std = sigma_clipped_stats(temp_lc.flux.value)
    spline.prior_sigma = np.ones(spline.shape[1]) * std

    # Recalculate the fit
    dm = lk.SparseDesignMatrixCollection([spline, bkg, jitter])
    _ = r.correct(dm)
    clc = lc - (r.diagnostic_lightcurves["bkg"] + r.diagnostic_lightcurves["jitter"])
    return clc


def remove_scattered_light_model(tpf, dir=None):
    """
    Removes the scattered light from a TPF, returning a corrected TPF.

    Will load a `ScatteredLightBackground` object to create the model.

    Parameters
    ----------
    tpf : lk.TargetPixelFile
        The TPF to correct
    dir : str
        Optional directory where the `ScatteredLightBackground` solutions are stored

    Returns
    -------
    ctpf : lk.TargetPixelFile
        The corrected TPF
    """
    b = ScatteredLightBackground.from_tpf(tpf, dir=dir)
    return tpf - b.model()


def remove_sky_model(
    tpf,
    dir=None,
    npca_components=10,
    timescale=1,
    cadence_mask=None,
    timeseries=None,
):
    """
    Removes the scattered light -and- stars from a TPF, returning a corrected TPF.

    This TPF should have a mean of ~0.

    Will load a `ScatteredLightBackground` object to create the model.

    Parameters
    ----------
    tpf : lk.TargetPixelFile
        The TPF to correct
    dir : str
        Optional directory where the `ScatteredLightBackground` solutions are stored
    npca_components: int
        The number of PCA components to detrend. Must be between 1 and 10. Using more
        components may help the fit be more accurate, but could make it less precise.
    timescale: float
        The timescale to use for splines when detrending. By default, will use 1 day.
    cadence_mask: np.ndarray
        An np.ndarray of bools to use to correct the data. Cadences that are False will
        not be used to fit the model. Use this to e.g. remove known transiting events.
    timeseries: np.ndarray
        Optional 2D np.ndarray of floats with shape [tpf.shape[0] x N]. Pass in time series
        within the TPF you know are present, and want to detrend against. e.g. Pass
        in models of expected light curves with transits, or pass in the SAP photometry of known
        variable stars. These will be used in the model fit, but will not be detrended out.
        This will ensure we get a good fit to the systematics in each pixel.

    Returns
    -------
    ctpf : lk.TargetPixelFile
        The corrected TPF
    """
    b = ScatteredLightBackground.from_tpf(tpf, dir=dir)
    npca_components = np.min([np.max([1, npca_components]), 10])
    if timeseries is None:
        timeseries = np.ones((tpf.shape[0], 1))
    timeseries = np.atleast_2d(timeseries)
    if (timeseries.shape[1] == tpf.shape[0]) & (timeseries.shape[0] != tpf.shape[0]):
        timeseries = timeseries.T
    if timeseries.shape[0] != tpf.shape[0]:
        raise ValueError(f"Please pass `timeseries` with shape [{tpf.shape[0]} x N]")

    # Break anywhere where there is a significant gap in the data
    dt = np.diff(tpf.time.value)
    breaks = list(np.where(((dt / np.median(dt)) > 4.9))[0] + 1)

    pix = (tpf.flux.value - b.model())[:, np.ones(tpf.shape[1:], bool)]
    pix_err = tpf.flux_err.value[:, np.ones(tpf.shape[1:], bool)]

    # Find an "outlier" mask where there are significant outliers in the average light curve.
    lc = (
        tpf_to_lightcurve(
            tpf,
            aperture_mask="all",
            dir=dir,
            timescale=timescale,
            npca_components=npca_components,
        ).normalize()
        - 1
    )
    outlier_mask = lc.remove_outliers(sigma=3, return_mask=True)[1]
    for count in range(3):
        outlier_mask |= np.gradient(outlier_mask.astype(float)) != 0
    outlier_mask = ~outlier_mask
    if cadence_mask is None:
        cadence_mask = np.ones(len(tpf.time.value), bool)

    # Systematics DM
    npoly = 2
    Xs = []
    X = sparse.lil_matrix((len(tpf.time.value), npoly * (len(breaks) + 1)))
    for idx, t in enumerate(np.array_split(tpf.time.value, breaks)):
        ar = np.asarray([(t - t.mean()) ** jdx for jdx in np.arange(npoly)])
        X[np.in1d(tpf.time.value, t), idx * npoly : (idx + 1) * npoly] = ar.T
    Xs.append(X)
    for pack in [b.bkg_pack[:, :npca_components], b.jitter_pack[:, :npca_components]]:
        X = sparse.lil_matrix(
            (len(tpf.time.value), npca_components * (len(breaks) + 1))
        )
        for idx, t, ar in zip(
            np.arange(len(breaks) + 1),
            np.array_split(tpf.time.value, breaks),
            np.array_split(pack, breaks),
        ):
            X[
                np.in1d(tpf.time.value, t),
                idx * npca_components : (idx + 1) * npca_components,
            ] = ar
        Xs.append(X)
    systematics = sparse.hstack(Xs, "csr")

    # Spline DM
    nb = int((tpf.time.value[-1] - tpf.time.value[0]) // timescale)
    nb = nb if nb % 2 == 1 else nb + 1
    knots = list(np.linspace(tpf.time.value[0], tpf.time.value[-1], nb + 2)[1:-1])
    spline = lk.designmatrix.create_sparse_spline_matrix(tpf.time.value, knots=knots)
    astrophysics = sparse.hstack(
        [spline.X, sparse.csr_matrix(lc.flux.value[:, None])], "csr"
    )

    # First pass, priors are broad
    X = systematics
    prior_mu = np.zeros((pix.shape[1], X.shape[1]))
    prior_sigma = np.ones((pix.shape[1], X.shape[1])) * np.inf

    ks = (cadence_mask & outlier_mask)[:, None] * np.ones(pix.shape, bool)
    for jdx in [1, 2]:
        for count in range(3):
            w = np.asarray(
                [
                    np.linalg.solve(
                        X[k].T.dot(X[k] / pe[k, None] ** 2) + np.diag(1 / ps ** 2),
                        X[k].T.dot(p[k] / pe[k] ** 2) + pm / ps ** 2,
                    )
                    for k, p, pe, pm, ps in zip(
                        ks.T, pix.T, pix_err.T, prior_mu, prior_sigma
                    )
                ]
            )
            model = X.dot(w.T)  # .reshape(tpf.shape)
            ks &= ~sigma_clip((pix - model) / pix_err, sigma=3).mask
        if jdx == 2:
            break
        # For second pass, we add in an astrophysics model with some reasonable priors
        prior_mu_sys, prior_sigma_sys = w, np.abs(w) * 0.1
        prior_mu_ast, prior_sigma_ast = np.zeros(pix.shape[1])[:, None] * np.ones(
            astrophysics.shape[1]
        ), sigma_clipped_stats(np.ma.masked_array(pix - model, ~ks), axis=0)[2][
            :, None
        ] * np.ones(
            astrophysics.shape[1]
        )
        prior_mu, prior_sigma = np.hstack([prior_mu_sys, prior_mu_ast]), np.hstack(
            [prior_sigma_sys, prior_sigma_ast]
        )
        X = sparse.hstack([systematics, astrophysics], "csr")
    model = (
        systematics.dot(w[:, : systematics.shape[1]].T).reshape(tpf.shape) + b.model()
    )
    return tpf - model
