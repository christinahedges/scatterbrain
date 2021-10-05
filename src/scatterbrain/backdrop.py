from tqdm import tqdm

from .cupy_numpy_imports import np, xp
from .designmatrix import (
    cartesian_design_matrix,
    radial_design_matrix,
    spline_design_matrix,
    strap_design_matrix,
)
from .utils import get_sat_mask, get_star_mask


class BackDrop(object):
    """Class for working with TESS data to fit and use scattered background models.

    BackDrop will automatically set up a reasonable background model for you. If you
    want to tweak the model, check out the `design_matrix` API.
    """

    def __init__(
        self, column=None, row=None, ccd=3, sigma_f=None, nknots=40, cutout_size=2048
    ):
        """Initialize a `BackDrop` object either for fitting or loading a model.

        Parameters
        ----------
        column : None or xp.ndarray
            The column numbers to evaluate the design matrix at. If None, uses all pixels.
        row : None or xp.ndarray
            The column numbers to evaluate the design matrix at. If None, uses all pixels.
        ccd : int
            CCD number
        sigma_f : xp.ndarray
            The weights for each pixel in the design matrix. Default is equal weights.
        nknots : int
                Number of knots to for spline matrix
        cutout_size : int
                Size of a "cutout" of images to use. Default is 2048. Use a smaller cut out to test functionality

        """

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
        """Resets the weights and average frame"""
        self.weights_basic = []
        self.weights_full = []
        self._average_frame = xp.zeros(self.shape)
        self._average_frame_count = 0

    def __repr__(self):
        return f"BackDrop CCD:{self.ccd} ({len(self.weights_basic)} frames)"

    def _build_masks(self, frame):
        """Builds a set of pixel masks for the frame, which downweight saturated pixels or pixels with stars."""
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

    def _fit_basic(self, flux):
        """Fit the first design matrix"""
        self.weights_basic.append(self.A1.fit_frame(xp.log10(flux)))

    def _fit_full(self, flux):
        """Fit the second design matrix"""
        self.weights_full.append(self.A2.fit_frame(flux))

    def _model_basic(self, tdx):
        """Model from the first design matrix"""
        return xp.power(10, self.A1.dot(self.weights_basic[tdx])).reshape(self.shape)

    def _model_full(self, tdx):
        """Model from the second design matrix"""
        return self.A2.dot(self.weights_full[tdx]).reshape(self.shape)

    def model(self, time_index):
        """Build a model for a frame with index `time_index`"""
        return self._model_basic(time_index) + self._model_full(time_index)

    def fit_frame(self, frame):
        """Fit a single frame of TESS data.
        This will append to the list properties `self.weights_basic`, `self.weights_full`, `self.jitter`.

        Parameters
        ----------
        frame : np.ndarray
            2D array of values, must be shape
            `(self.cutout_size, self.cutout_size)`
        """
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
        """Fit a model to a flux cube, fitting each frame individually

        This will append to the list properties `self.weights_basic`, `self.weights_full`, `self.jitter`.

        Parameters
        ----------
        flux_cube : xp.ndarray or list
            3D array of frames.
        test_frame : int
            The index of the frame to use as the "reference frame".
            This reference frame will be used to build masks to set `sigma_f`.
            It should be the lowest background frame.
        """
        if isinstance(flux_cube, list):
            if not np.all(
                [f.shape == (self.cutout_size, self.cutout_size) for f in flux_cube]
            ):
                raise ValueError(
                    f"Frame is not ({self.cutout_size}, {self.cutout_size})"
                )
        elif isinstance(flux_cube, xp.ndarray):
            if flux_cube.ndim != 3:
                raise ValueError("`flux_cube` must be 3D")
            if not flux_cube.shape[1:] == (self.cutout_size, self.cutout_size):
                raise ValueError(
                    f"Frame is not ({self.cutout_size}, {self.cutout_size})"
                )
        else:
            raise ValueError("Pass an `xp.ndarray` or a `list`")
        self._build_masks(flux_cube[test_frame])
        for flux in tqdm(flux_cube, desc="Fitting Frames"):
            self.fit_frame(flux)

    def _fit_basic_batch(self, flux):
        """Fit the first design matrix, in a batched mode"""
        # weights = list(self.A1.fit_batch(xp.log10(flux)))
        return self.A1.fit_batch(xp.log10(flux))

    def _fit_full_batch(self, flux):
        """Fit the second design matrix, in a batched mode"""
        #        weights = list(self.A2.fit_batch(flux))
        return self.A2.fit_batch(flux)

    def _fit_batch(self, flux_cube):
        """Fit the both design matrices, in a batched mode"""
        weights_basic = self._fit_basic_batch(flux_cube)
        for tdx in range(len(weights_basic)):
            flux_cube[tdx] -= xp.power(10, self.A1.dot(weights_basic[tdx])).reshape(
                self.shape
            )
        weights_full = self._fit_full_batch(flux_cube)

        for tdx in range(len(weights_basic)):
            flux_cube[tdx] += xp.power(10, self.A1.dot(weights_basic[tdx])).reshape(
                self.shape
            )
        return weights_basic, weights_full

    def fit_model_batched(self, flux_cube, batch_size=50, test_frame=0):
        """Fit a model to a flux cube, fitting frames in batches of size `batch_size`.

        This will append to the list properties `self.weights_basic`, `self.weights_full`, `self.jitter`.

        Parameters
        ----------
        flux_cube : xp.ndarray
            3D array of frames.
        batch_size : int
            Number of frames to fit at once.
        test_frame : int
            The index of the frame to use as the "reference frame".
            This reference frame will be used to build masks to set `sigma_f`.
            It should be the lowest background frame.
        """
        if isinstance(flux_cube, list):
            if not np.all(
                [f.shape == (self.cutout_size, self.cutout_size) for f in flux_cube]
            ):
                raise ValueError(
                    f"Frame is not ({self.cutout_size}, {self.cutout_size})"
                )
        elif isinstance(flux_cube, xp.ndarray):
            if flux_cube.ndim != 3:
                raise ValueError("`flux_cube` must be 3D")
            if not flux_cube.shape[1:] == (self.cutout_size, self.cutout_size):
                raise ValueError(
                    f"Frame is not ({self.cutout_size}, {self.cutout_size})"
                )
        else:
            raise ValueError("Pass an `xp.ndarray` or a `list`")
        self._build_masks(flux_cube[test_frame])
        nbatches = xp.ceil(len(flux_cube) / batch_size).astype(int)
        weights_basic, weights_full = [], []
        l = xp.arange(0, nbatches + 1, dtype=int) * batch_size
        if l[-1] > len(flux_cube):
            l[-1] = len(flux_cube)
        for l1, l2 in zip(l[:-1], l[1:]):
            w1, w2 = self._fit_batch(flux_cube[l1:l2])
            weights_basic.append(w1)
            weights_full.append(w2)
        self.weights_basic = list(xp.vstack(weights_basic))
        self.weights_full = list(xp.vstack(weights_full))
        return

    @property
    def shape(self):
        if self.column is not None:
            return (self.row.shape[0], self.column.shape[0])
        else:
            return

    def save(self, outfile="backdrop_weights.npz"):
        """Save the best fit weights to a file"""
        xp.savez(outfile, xp.asarray(self.weights_basic), xp.asarray(self.weights_full))

    def load(self, infile="backdrop_weights.npz"):
        """Load a file for a set of input parameters."""
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
