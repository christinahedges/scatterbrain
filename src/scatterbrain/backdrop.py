import os

import fitsio
from astropy.io import fits
from tqdm import tqdm

from . import PACKAGEDIR
from .cupy_numpy_imports import load_image, np, xp
from .designmatrix import (
    cartesian_design_matrix,
    radial_spline_design_matrix,
    spline_design_matrix,
    strap_design_matrix,
)
from .utils import (
    _asteroid_mask,
    _package_jitter,
    get_sat_mask,
    get_star_mask,
    test_strip,
)
from .version import __version__


class BackDrop(object):
    """Class for working with TESS data to fit and use scattered background models.

    BackDrop will automatically set up a reasonable background model for you. If you
    want to tweak the model, check out the `design_matrix` API.
    """

    def __init__(
        self,
        sector,
        camera,
        ccd,
        column=None,
        row=None,
        sigma_f=None,
        nknots=50,
        cutout_size=2048,
        tstart=None,
        tstop=None,
        quality=None,
        verbose=False,
    ):
        """Initialize a `BackDrop` object either for fitting or loading a model.

        Parameters
        ----------
        sector : int
            Sector number
        camera : int
            Camera number
        ccd : int
            CCD number
        column : None or xp.ndarray
            The column numbers to evaluate the design matrix at. If None, uses all pixels.
        row : None or xp.ndarray
            The column numbers to evaluate the design matrix at. If None, uses all pixels.
        sigma_f : xp.ndarray
            The weights for each pixel in the design matrix. Default is equal weights.
        nknots : int
                Number of knots to for spline matrix
        cutout_size : int
                Size of a "cutout" of images to use. Default is 2048. Use a smaller cut out to test functionality

        """

        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.verbose = verbose
        self.A1 = radial_spline_design_matrix(
            column=column,
            row=row,
            ccd=ccd,
            sigma_f=sigma_f,
            prior_mu=2,
            prior_sigma=100,
            cutout_size=cutout_size,
        ) + cartesian_design_matrix(
            column=column,
            row=row,
            ccd=ccd,
            sigma_f=sigma_f,
            prior_mu=2,
            prior_sigma=100,
            cutout_size=cutout_size,
            npoly=2,
        )

        self.A2 = (
            spline_design_matrix(
                column=column,
                row=row,
                ccd=ccd,
                sigma_f=sigma_f,
                prior_sigma=200,
                nknots=nknots,
                cutout_size=cutout_size,
            )
            + radial_spline_design_matrix(
                column=column,
                row=row,
                ccd=ccd,
                sigma_f=sigma_f,
                prior_sigma=300,
                cutout_size=cutout_size,
            )
            + strap_design_matrix(
                column=column,
                row=row,
                ccd=ccd,
                sigma_f=sigma_f,
                prior_sigma=30,
                cutout_size=cutout_size,
            )
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
        self.weights_basic = []
        self.weights_full = []
        self.jitter = []
        self._average_frame = xp.zeros(self.shape)
        self._average_frame_count = 0
        self.tstart = tstart
        self.tstop = tstop
        self.quality = quality

    def update_sigma_f(self, sigma_f):
        self.A1.update_sigma_f(sigma_f)
        self.A2.update_sigma_f(sigma_f)

    def clean(self):
        """Resets the weights and average frame"""
        self.weights_basic = []
        self.weights_full = []
        self._average_frame = xp.zeros(self.shape)
        self._average_frame_count = 0
        self.jitter = []

    def __repr__(self):
        return f"BackDrop CCD:{self.ccd} ({len(self.weights_basic)} frames)"

    def _build_masks(self, frame):
        """Builds a set of pixel masks for the frame, which downweight saturated pixels or pixels with stars."""
        # if frame.shape != (2048, 2048):
        #     raise ValueError("Pass a frame that is (2048, 2048)")
        if not hasattr(self, "star_mask"):
            self.star_mask = get_star_mask(frame)
        if not hasattr(self, "sat_mask"):
            self.sat_mask = get_sat_mask(frame)
        if not hasattr(self, "jitter_mask"):
            if (~self.star_mask & self.sat_mask).sum() > 6000:
                s = np.random.choice(
                    (~self.star_mask & self.sat_mask).sum(), size=6000, replace=False
                )
                l = np.asarray(np.where(~self.star_mask & self.sat_mask))
                l = l[:, s]
                self.jitter_mask = np.zeros((self.cutout_size, self.cutout_size), bool)
                self.jitter_mask[l[0], l[1]] = True
            else:
                self.jitter_mask = (~self.star_mask & self.sat_mask).copy()
        sigma_f = xp.ones(frame.shape)
        sigma_f[~self.star_mask | ~self.sat_mask] = 1e6
        self.update_sigma_f(sigma_f)
        return

    def _build_asteroid_mask(self, flux_cube):
        """Creates a mask for asteroids, and increases the flux errors in those
        pixels to reduce their contribution to the model

        Parameters
        ----------
        flux_cube: xp.ndarray or list of xp.ndarray
            3D flux cube of data. Be aware that this should be a short "batch"
            of flux data (e.g. 50 frames).
        """
        ast_mask = _asteroid_mask(flux_cube)
        sigma_f = xp.ones(flux_cube[0].shape)
        # More than the saturation limit
        sigma_f[~self.star_mask | ~self.sat_mask] = 1e6
        # Here we downweight asteroids and other variable pixels
        # asteroids are probably not much brighter than 100s of counts
        sigma_f[ast_mask] += 200
        self.update_sigma_f(sigma_f)
        return ast_mask

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
        return (self._average_frame / self._average_frame_count)[self.row, :][
            :, self.column
        ]

    def fit_model(self, flux_cube, test_frame=0, verbose=False):
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
        for flux in tqdm(
            flux_cube,
            desc="Fitting Frames",
            leave=True,
            position=0,
        ):
            self.fit_frame(flux)
        _package_jitter(self)

    def _fit_basic_batch(self, flux):
        """Fit the first design matrix, in a batched mode"""
        # weights = list(self.A1.fit_batch(xp.log10(flux)))
        return self.A1.fit_batch(xp.log10(flux))

    def _fit_full_batch(self, flux):
        """Fit the second design matrix, in a batched mode"""
        #        weights = list(self.A2.fit_batch(flux))
        return self.A2.fit_batch(flux)

    def _fit_batch(self, flux_cube, mask_asteroids=True):
        """Fit the both design matrices, in a batched mode"""
        weights_basic = self._fit_basic_batch(flux_cube)
        for tdx in range(len(weights_basic)):
            flux_cube[tdx] -= xp.power(10, self.A1.dot(weights_basic[tdx])).reshape(
                self.shape
            )
        if mask_asteroids:
            ast_mask = self._build_asteroid_mask(flux_cube)
        weights_full = self._fit_full_batch(flux_cube)

        jitter = []
        for tdx in range(len(weights_full)):
            flux_cube[tdx] -= self.A2.dot(weights_full[tdx]).reshape(self.shape)
            jitter.append(flux_cube[tdx][self.jitter_mask])
            self._average_frame += flux_cube[tdx]
            self._average_frame_count += 1
            flux_cube[tdx] += self.A2.dot(weights_full[tdx]).reshape(self.shape)

        for tdx in range(len(weights_basic)):
            flux_cube[tdx] += xp.power(10, self.A1.dot(weights_basic[tdx])).reshape(
                self.shape
            )
        return weights_basic, weights_full, jitter, ast_mask

    def fit_model_batched(
        self, flux_cube, batch_size=50, test_frame=0, mask_asteroids=True
    ):
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
        weights_basic, weights_full, jitter = [], [], []
        l = xp.arange(0, nbatches + 1, dtype=int) * batch_size
        if l[-1] > len(flux_cube):
            l[-1] = len(flux_cube)
        for l1, l2 in zip(l[:-1], l[1:]):
            w1, w2, j, ast_mask = self._fit_batch(
                flux_cube[l1:l2], mask_asteroids=mask_asteroids
            )
            weights_basic.append(w1)
            weights_full.append(w2)
            jitter.append(j)
        self.weights_basic = list(xp.vstack(weights_basic))
        self.weights_full = list(xp.vstack(weights_full))
        self.jitter = list(xp.vstack(j))
        _package_jitter(self)
        return

    @property
    def shape(self):
        if self.column is not None:
            return (self.row.shape[0], self.column.shape[0])
        else:
            return

    def _package_weights_hdulist(self):
        hdu0 = self.hdu0
        s = np.argsort(self.tstart)
        cols = []
        for key in ["tstart", "tstop", "quality"]:
            if getattr(self, key) is not None:
                cols.append(
                    fits.Column(
                        name=key,
                        format="D",
                        unit="BJD - 2457000",
                        array=getattr(self, key)[s],
                    )
                )
        hdu1 = fits.BinTableHDU.from_columns(cols, name="TIME")
        hdu2 = fits.ImageHDU(np.asarray(self.weights_basic)[s], name="WEIGHTS1")
        hdu3 = fits.ImageHDU(np.asarray(self.weights_full)[s], name="WEIGHTS2")
        hdu4 = fits.ImageHDU(np.asarray(self.jitter_pack)[s], name="JITTER")
        hdu5 = fits.ImageHDU(np.asarray(self.average_frame), name="AVGFRAME")
        hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5])
        return hdul

    def save(self, output_dir=None, overwrite=False):
        """Save a model fit to the scatterbrain data directory."""
        if not hasattr(self, "weights_basic"):
            raise ValueError("There is no model to save")
        self.hdu0 = fits.PrimaryHDU()
        self.hdu0.header["ORIGIN"] = "tess-backdrop"
        self.hdu0.header["AUTHOR"] = "christina.l.hedges@nasa.gov"
        self.hdu0.header["VERSION"] = __version__
        for key in ["sector", "camera", "ccd", "cutout_size"]:
            self.hdu0.header[key] = getattr(self, key)

        if output_dir is None:
            output_dir = f"{PACKAGEDIR}/data/sector{self.sector:03}/camera{self.camera:02}/ccd{self.ccd:02}/"
            if output_dir != "":
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

        hdul = self._package_weights_hdulist()
        fname = (
            f"tessbackdrop_sector{self.sector}_camera{self.camera}_ccd{self.ccd}.fits"
        )
        hdul.writeto(output_dir + fname, overwrite=overwrite)

    def load(self, input, dir=None):
        """
        Load a model fit to the tess-backrop data directory.

        Parameters
        ----------
        input: tuple or string
            Either pass a tuple with `(sector, camera, ccd)` or pass
            a file name in `dir` to load
        dir : str
            Optional tring with the directory name

        Returns
        -------
        self: `tess_backdrop.SimpleBackDrop` object
        """
        if isinstance(input, tuple):
            if len(input) == 3:
                sector, camera, ccd = input
                fname = f"tessbackdrop_sector{sector}_camera{camera}_ccd{ccd}.fits"
            else:
                raise ValueError("Please pass tuple as `(sector, camera, ccd)`")
        elif isinstance(input, str):
            fname = input
        else:
            raise ValueError("Can not parse input")
        if dir is None:
            dir = f"{PACKAGEDIR}/data/sector{sector:03}/camera{camera:02}/ccd{ccd:02}/"
        if dir != "":
            if not os.path.isdir(dir):
                raise ValueError("No solutions exist")

        with fits.open(dir + fname, lazy_load_hdus=True) as hdu:
            for key in [
                "sector",
                "camera",
                "ccd",
                "cutout_size",
            ]:
                setattr(self, key, hdu[0].header[key])
            if "tstart" in hdu[1].data.names:
                self.tstart = hdu[1].data["tstart"]
            if "tstop" in hdu[1].data.names:
                self.tstop = hdu[1].data["tstop"]
            if "quality" in hdu[1].data.names:
                self.quality = hdu[1].data["QUALITY"]
            self.weights_basic = list(hdu[2].data)
            weights_full = hdu[3].data
            self.weights_full = np.hstack(
                [weights_full[:, :-2048], weights_full[:, -2048:][:, self.column]]
            )
            self.jitter_pack = hdu[4].data
            self._average_frame = hdu[5].data
            self._average_frame_count = 1
        return self

    @staticmethod
    def from_disk(sector, camera, ccd, row=None, column=None, dir=None):
        return BackDrop(
            sector=sector, camera=camera, ccd=ccd, row=row, column=column
        ).load((sector, camera, ccd), dir=dir)

    @staticmethod
    def from_tpf(tpf, dir=None):
        return BackDrop(
            sector=tpf.sector,
            camera=tpf.camera,
            ccd=tpf.ccd,
            row=np.arange(tpf.shape[1]) + tpf.row,
            column=np.arange(tpf.shape[2]) + tpf.column,
        ).load((tpf.sector, tpf.camera, tpf.ccd), dir=dir)

    @staticmethod
    def from_tess_images(
        fnames,
        batch_size=50,
        test_frame=None,
        cutout_size=2048,
        sector=None,
        camera=None,
        ccd=None,
        verbose=False,
        quality_mask=32,
    ):
        """Creates a backdrop model from filenames

        Parameters
        ----------
        fnames : list
            List of filenames to compute the background model for
        batch_size : int
            Number of files to process in a given batch
        test_frame : None or int
            The frame to use as a "test" frame for building masks.
            If None, a reasonable frame will be chosen.

        Returns
        -------
        b : `scatterbrain.backdrop.BackDrop`
            BackDrop object
        """
        if not isinstance(fnames, (list, xp.ndarray)):
            raise ValueError("Pass an array of file names")
        if not isinstance(fnames[0], (str)):
            raise ValueError("Pass an array of strings")

        if sector is None:
            try:
                sector = int(fnames[0].split("-s")[1].split("-")[0])
            except ValueError:
                raise ValueError("Can not parse file name for sector number")
        if camera is None:
            try:
                camera = fitsio.read_header(fnames[0], ext=1)["CAMERA"]
            except ValueError:
                raise ValueError("Can not find a camera number")
        if ccd is None:
            try:
                ccd = fitsio.read_header(fnames[0], ext=1)["CCD"]
            except ValueError:
                raise ValueError("Can not find a CCD number")

        blown_out_strips = np.asarray(
            [
                test_strip(fname)
                for fname in tqdm(
                    fnames,
                    desc="Finding blown out strips",
                    position=0,
                    leave=True,
                )
            ]
        )
        # make a backdrop object
        self = BackDrop(
            sector=sector,
            camera=camera,
            ccd=ccd,
            cutout_size=cutout_size,
        )
        self.tstart = np.asarray(
            [fitsio.read_header(fname, ext=0)["TSTART"] for fname in fnames]
        )
        s = np.argsort(self.tstart)
        self.tstart, fnames = self.tstart[s], np.asarray(fnames)[s]
        self.tstop = np.asarray(
            [fitsio.read_header(fname, ext=0)["TSTOP"] for fname in fnames]
        )
        self.quality = np.asarray(
            [fitsio.read_header(fname, ext=1)["DQUALITY"] for fname in fnames]
        )
        self.quality[blown_out_strips.any(axis=1)] |= 8192
        # Step 1: find a good test frame
        if test_frame is None:
            # Test frame is a low flux frame, close to the middle of the dataset.
            f = np.asarray(
                [
                    fitsio.read(fname)[
                        np.min([self.A1.bore_pixel[0], 2047]),
                        45 + np.min([self.A1.bore_pixel[1], 0]),
                    ]
                    for fname in tqdm(
                        fnames,
                        desc="Finding test frame",
                        leave=True,
                        position=0,
                    )
                ]
            )

            l = np.where((f < np.nanpercentile(f, 5)) & (self.quality == 0))[0]
            if len(l) == 0:
                test_frame = len(fnames) // 2
            else:
                test_frame = l[np.argmin(np.abs(l - len(f) // 2))]

        self._build_masks(load_image(fnames[test_frame], cutout_size=cutout_size))

        # Step 2: run as a batch...
        def run_batch(fnames):
            nbatches = xp.ceil(len(fnames) / batch_size).astype(int)
            weights_basic, weights_full, jitter = [], [], []
            l = xp.arange(0, nbatches + 1, dtype=int) * batch_size
            if l[-1] > len(fnames):
                l[-1] = len(fnames)
            asteroid_mask = np.zeros((self.cutout_size, self.cutout_size), dtype=int)
            for l1, l2 in tqdm(
                zip(l[:-1], l[1:]),
                desc=f"Fitting frames in batches of {batch_size}",
                total=len(l) - 1,
                leave=True,
                position=0,
            ):
                flux_cube = [
                    load_image(fnames[idx], cutout_size=cutout_size)
                    for idx in np.arange(l1, l2)
                ]
                w1, w2, j, ast_mask = self._fit_batch(flux_cube)
                weights_basic.append(w1)
                weights_full.append(w2)
                jitter.append(j)
                asteroid_mask += ast_mask
            return (
                xp.vstack(weights_basic),
                xp.vstack(weights_full),
                xp.vstack(jitter),
                asteroid_mask,
            )

        ok = (self.quality & (8192 | quality_mask)) == 0
        w1, w2, j, self.asteroid_mask = run_batch(fnames[ok])

        self.weights_basic = np.zeros((len(fnames), w1.shape[1]))
        self.weights_basic[ok] = w1
        self.weights_basic[~ok] = np.nan
        self.weights_full = np.zeros((len(fnames), w2.shape[1]))
        self.weights_full[ok] = w2
        self.weights_full[~ok] = np.nan
        self.jitter = np.zeros((len(fnames), j.shape[1]))
        self.jitter[ok] = j
        self.jitter[~ok] = np.nan
        _package_jitter(self)
        return self
