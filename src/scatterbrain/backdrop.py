import numpy as np
from tqdm import tqdm

from .cupy_numpy_imports import *
from .designmatrix import (
    cartesian_design_matrix,
    radial_design_matrix,
    spline_design_matrix,
    strap_design_matrix,
)
from .utils import get_star_mask, get_sat_mask


class BackDrop(object):
    def __init__(
        self, column=None, row=None, ccd=3, sigma_f=None, nknots=40, cutout_size=2048
    ):
        self.A1 = radial_design_matrix(
            column=column,
            row=row,
            ccd=ccd,
            sigma_f=sigma_f,
            prior_mu=2,
            prior_sigma=3,
            cutout_size=cutout_size,
        ) + cartesian_design_matrix(
            column=column,
            row=row,
            ccd=ccd,
            sigma_f=sigma_f,
            prior_mu=2,
            prior_sigma=3,
            cutout_size=cutout_size,
        )
        self.A2 = spline_design_matrix(
            column=column,
            row=row,
            ccd=ccd,
            sigma_f=sigma_f,
            prior_sigma=100,
            nknots=nknots,
            cutout_size=cutout_size,
        ) + strap_design_matrix(
            column=column,
            row=row,
            ccd=ccd,
            sigma_f=sigma_f,
            prior_sigma=100,
            cutout_size=cutout_size,
        )
        self.cutout_size = cutout_size
        if row is None:
            self.column = np.arange(self.cutout_size)
        else:
            self.column = column
        if row is None:
            self.row = np.arange(self.cutout_size)
        else:
            self.row = row
        self.ccd = ccd
        self.weights_basic = []
        self.weights_full = []
        self.jitter = []
        self._average_frame = xp.zeros(self.shape)
        self._average_frame_count = 0

    def update_sigma_f(self, sigma_f):
        self.A1.update_sigma_f(sigma_f)
        self.A2.update_sigma_f(sigma_f)

    def clean(self):
        self.weights_basic = []
        self.weights_full = []
        self._average_frame = xp.zeros(self.shape)
        self._average_frame_count = 0

    def __repr__(self):
        return f"BackDrop CCD:{self.ccd} ({len(self.weights_basic)} frames)"

    def _build_masks(self, frame):
        # if frame.shape != (2048, 2048):
        #     raise ValueError("Pass a frame that is (2048, 2048)")
        star_mask = get_star_mask(frame)
        sat_mask = get_sat_mask(frame)
        sigma_f = xp.ones(frame.shape)
        sigma_f[~star_mask | ~sat_mask] = 1e5
        self.update_sigma_f(sigma_f)
        if (~star_mask & sat_mask).sum() > 5000:
            s = np.random.choice(
                (~star_mask & sat_mask).sum(), size=5000, replace=False
            )
            l = np.asarray(np.where(~star_mask & sat_mask))
            l = l[:, s]
            self.jitter_mask = np.zeros((self.cutout_size, self.cutout_size), bool)
            self.jitter_mask[l[0], l[1]] = True
        else:
            self.jitter_mask = (~star_mask & sat_mask).copy()
        return

    def model(tdx):
        return self.A1.dot(self.w[tdx])

    def _fit_basic(self, flux):
        self.weights_basic.append(self.A1.fit_frame(xp.log10(flux)))

    def _fit_full(self, flux):
        self.weights_full.append(self.A2.fit_frame(flux))

    def _model_basic(self, tdx):
        return xp.power(10, self.A1.dot(self.weights_basic[tdx])).reshape(self.shape)

    def _model_full(self, tdx):
        return self.A2.dot(self.weights_full[tdx]).reshape(self.shape)

    def model(self, tdx):
        return self._model_basic(tdx) + self._model_full(tdx)

    def fit_frame(self, frame):
        if not frame.shape == (self.cutout_size, self.cutout_size):
            raise ValueError(f"Frame is not ({self.cutout_size}, {self.cutout_size})")
        self._fit_basic(frame)
        res = frame - self._model_basic(-1)
        self._fit_full(res)
        res = res - self._model_full(-1)
        self.jitter.append(res[self.jitter_mask])
        self._average_frame += res
        self._average_frame_count += 1
        return

    @property
    def average_frame(self):
        return self._average_frame / self._average_frame_count

    def fit_model(self, flux_cube, test_frame=0):
        if flux_cube.ndim != 3:
            raise ValueError("`flux_cube` must be 3D")
        if not flux_cube.shape[1:] == (self.cutout_size, self.cutout_size):
            raise ValueError(f"Frame is not ({self.cutout_size}, {self.cutout_size})")
        self._build_masks(flux_cube[test_frame])
        for flux in tqdm(flux_cube, desc="Fitting Frames"):
            self.fit_frame(flux)

    @property
    def shape(self):
        if self.column is not None:
            return (self.row.shape[0], self.column.shape[0])
        else:
            return

    def save(self, outfile="backdrop_weights.npz"):
        xp.savez(outfile, xp.asarray(self.weights_basic), xp.asarray(self.weights_full))

    def load(self, infile="backdrop_weights.npz"):
        cpzfile = xp.load(infile)
        self.weights_basic = list(cpzfile["arr_0"])
        self.weights_full = cpzfile["arr_1"]
        if self.column is not None:
            self.weights_full = np.hstack(
                [
                    self.weights_full[:, : -self.cutout_size],
                    self.weights_full[
                        :, self.weights_full.shape[1] - self.cutout_size + self.column
                    ],
                ]
            )
        self.weights_full = list(self.weights_full)
        return self
