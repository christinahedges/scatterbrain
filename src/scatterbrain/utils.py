"""basic utility functions"""
import fitsio
import numpy as np
from fbpca import pca
from scipy.signal import medfilt

from .cupy_numpy_imports import xp


def _spline_basis_vector(x, degree, i, knots):
    """Recursive function to create a single spline basis vector for an ixput x,
    for the ith knot.
    See https://en.wikipedia.org/wiki/B-spline for a definition of B-spline
    basis vectors

    NOTE: This is lifted out of the funcs I wrote for lightkurve

    Parameters
    ----------
    x : cp.ndarray
        Ixput x
    degree : int
        Degree of spline to calculate basis for
    i : int
        The index of the knot to calculate the basis for
    knots : cp.ndarray
        Array of all knots
    Returns
    -------
    B : cp.ndarray
        A vector of same length as x containing the spline basis for the ith knot
    """
    if degree == 0:
        B = xp.zeros(len(x))
        B[(x >= knots[i]) & (x <= knots[i + 1])] = 1
    else:
        da = knots[degree + i] - knots[i]
        db = knots[i + degree + 1] - knots[i + 1]
        if (knots[degree + i] - knots[i]) != 0:
            alpha1 = (x - knots[i]) / da
        else:
            alpha1 = xp.zeros(len(x))
        if (knots[i + degree + 1] - knots[i + 1]) != 0:
            alpha2 = (knots[i + degree + 1] - x) / db
        else:
            alpha2 = xp.zeros(len(x))
        B = (_spline_basis_vector(x, (degree - 1), i, knots)) * (alpha1) + (
            _spline_basis_vector(x, (degree - 1), (i + 1), knots)
        ) * (alpha2)
    return B


def get_star_mask(f):
    """False where stars are. Keep in mind this might be a bad
    set of hard coded parameters for some TESS images!"""
    # This removes pixels where there is a steep flux gradient
    star_mask = (xp.hypot(*xp.gradient(f)) < 30) & (f < 9e4)
    # This broadens that mask by one pixel on all sides
    star_mask = (
        ~(xp.asarray(xp.gradient(star_mask.astype(float))) != 0).any(axis=0) & star_mask
    )
    return star_mask


def _find_saturation_column_centers(mask):
    """
    Finds the center point of saturation columns.
    Parameters
    ----------
    mask : xp.ndarray of bools
        Mask where True indicates a pixel is saturated
    Returns
    -------
    centers : xp.ndarray
        Array of the centers in XY space for all the bleed columns
    """
    centers = []
    radii = []
    idxs = xp.where(mask.any(axis=0))[0]
    for idx in idxs:
        line = mask[:, idx]
        seq = []
        val = line[0]
        jdx = 0
        while jdx <= len(line):
            while line[jdx] == val:
                jdx += 1
                if jdx >= len(line):
                    break
            if jdx >= len(line):
                break
            seq.append(jdx)
            val = line[jdx]
        w = xp.array_split(line, seq)
        v = xp.array_split(xp.arange(len(line)), seq)
        coords = [(idx, v1.mean().astype(int)) for v1, w1 in zip(v, w) if w1.all()]
        rads = [len(v1) / 2 for v1, w1 in zip(v, w) if w1.all()]
        for coord, rad in zip(coords, rads):
            centers.append(coord)
            radii.append(rad)
    centers = xp.asarray(centers)
    radii = xp.asarray(radii)
    return centers, radii


def get_sat_mask(f):
    """False where saturation spikes are. Keep in mind this might be a bad
    set of hard coded parameters for some TESS images!"""
    sat = f > 9e4
    l, r = _find_saturation_column_centers(sat)
    col, row = xp.mgrid[: f.shape[0], : f.shape[1]]
    l, r = l[r > 1], r[r > 1]
    for idx in range(len(r)):
        sat |= (xp.hypot(row - l[idx, 0], col - l[idx, 1]) < (r[idx] * 2)) & (
            xp.abs(col - l[idx, 1]) < xp.ceil(xp.min([r[idx] * 0.5, 7]))
        )
        sat |= xp.hypot(row - l[idx, 0], col - l[idx, 1]) < (r[idx] * 0.75)

    return ~sat


def _package_jitter(backdrop, xpca_components=20):
    """Packages the jitter terms into detrending vectors similar to CBVs.
    Splits the jitter into timescales of:
        - t < 0.5 days
        - t > 0.5 days
    Parameters
    ----------
    backdrop: tess_backdrop.FullBackDrop
        Ixput backdrop to package
    xpca_components : int
        Number of pca components to compress into. Default 20, which will result
        in an ntimes x 40 matrix.
    Returns
    -------
    matrix : xp.ndarray
        The packaged jitter matrix will contains the top principle components
        of the jitter matrix.
    """

    backdrop.jitter = xp.asarray(backdrop.jitter)
    finite = np.isfinite(backdrop.jitter).all(axis=1)
    # If there aren't enough jitter components, just return them.
    if backdrop.jitter.shape[0] < 40:
        # Not enough times
        return None
    if finite.sum() < 50:
        # Not enough pixels
        return backdrop.jitter.copy()

    # We split at data downlinks where there is a gap of at least 0.2 days
    breaks = xp.where(xp.diff(backdrop.tstart[finite]) > 0.2)[0] + 1
    breaks = xp.hstack([0, breaks, len(backdrop.tstart[finite])])

    jitter_short = backdrop.jitter[finite].copy()

    nb = int(0.5 / xp.median(xp.diff(backdrop.tstart)))
    nb = [nb if (nb % 2) == 1 else nb + 1][0]

    def smooth(x):
        return xp.asarray([medfilt(x[:, tdx], nb) for tdx in range(x.shape[1])])

    jitter_medium = xp.hstack(
        [
            smooth(backdrop.jitter[finite][x1:x2])
            for x1, x2 in zip(breaks[:-1], breaks[1:])
        ]
    ).T

    U1, s, V = pca(jitter_short - jitter_medium, xpca_components, n_iter=10)
    U2, s, V = pca(jitter_medium, xpca_components, n_iter=10)

    X = xp.hstack(
        [
            U1,
            U2,
        ]
    )
    X = xp.hstack([X[:, idx::xpca_components] for idx in range(xpca_components)])
    Xall = np.zeros((backdrop.tstart.shape[0], X.shape[1]))
    Xall[finite] = X
    backdrop.jitter = Xall


def test_strip(fname):
    """Test whether any of the CCD strips are saturated"""
    f = np.median(
        np.abs(fitsio.read(fname)[:10, 45 : 2048 + 45].mean(axis=0)).reshape((4, 512)),
        axis=1,
    )
    return f > 10000
