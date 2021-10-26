import os
from dataclasses import dataclass

import fitsio
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from tqdm import tqdm

from . import PACKAGEDIR
from .asteroids import get_asteroid_locations
from .background import ScatteredLightBackground
from .cupy_numpy_imports import load_image, xp
from .utils import minmax
from .version import __version__


@dataclass
class StarScene:
    """Class to remove stars from TESS images

    Parameters
    ----------
    sector : int
        TESS sector number
    camera: int
        TESS camera number
    ccd : int
        TESS CCD number
    cutout_size: int
        This will process the fits images in "cutouts". The larger the cutout,
        the more memory fitting will take. 512 is a sane default.
    """

    sector: int
    camera: int
    ccd: int
    cutout_size: int = 512

    def __post_init__(self):
        self.background = ScatteredLightBackground.from_disk(
            sector=self.sector,
            camera=self.camera,
            ccd=self.ccd,
            row=np.asarray([1]),
            column=np.asarray([1]),
        )
        self.break_point = np.where(
            (
                np.diff(self.background.tstart)
                / np.median(np.diff(self.background.tstart))
            )
            > 10
        )[0][0]
        self.orbit_masks = [
            np.arange(len(self.background.tstart)) <= self.break_point,
            np.arange(len(self.background.tstart)) > self.break_point,
        ]
        self.Xs = [
            self._get_design_matrix(self.orbit_masks[0]),
            self._get_design_matrix(self.orbit_masks[1]),
        ]
        self.weights = [
            np.zeros((self.Xs[0].shape[1], 2048, 2048), np.float32),
            np.zeros((self.Xs[0].shape[1], 2048, 2048), np.float32),
        ]
        return

    @property
    def tstart(self):
        return self.background.tstart

    @property
    def tstop(self):
        return self.background.tstop

    @property
    def quality(self):
        return self.background.quality

    @property
    def empty(self):
        return (self.weights[0] == 0).all() & (self.weights[1] == 0).all()

    @property
    def shape(self):
        return (len(self.tstart), self.cutout_size, self.cutout_size)

    def _load_background(self, row=None, column=None):
        self.background = ScatteredLightBackground.from_disk(
            sector=self.sector, camera=self.camera, ccd=self.ccd, row=row, column=column
        )

    def __repr__(self):
        return f"StarScene {self.shape}, Sector {self.sector}, Camera {self.camera}, CCD {self.ccd}"

    @property
    def path(self):
        return (
            f"{PACKAGEDIR}/data/sector{self.sector:03}/camera{self.camera:02}/ccd{self.ccd:02}/"
            f"tessstarscene_sector{self.sector}_camera{self.camera}_ccd{self.ccd}.fits"
        )

    def _get_design_matrix(self, mask=None, ncomps=20):
        """Construct a design matrix from the ScatteredLightBackground object"""
        if mask is None:
            mask = np.ones(self.background.tstart.shape[0], bool)
        t = (self.background.tstart[mask] - self.background.tstart.mean()) / 30
        X = np.hstack(
            [
                self.background.jitter_pack[mask, :ncomps],
                self.background.bkg_pack[mask, :ncomps],
                np.vstack([t ** 0, t, t ** 2]).T,
            ]
        )
        return X

    @staticmethod
    def from_tess_images(fnames, sector=None, camera=None, ccd=None, cutout_size=512):
        """Class to remove stars from TESS images

        Parameters
        ----------
        fnames : list of str
            File names to use to build the object
        sector : int
            TESS sector number
        camera: int
            TESS camera number
        ccd : int
            TESS CCD number
        cutout_size: int
            This will process the fits images in "cutouts". The larger the cutout,
            the more memory fitting will take. 512 is a sane default.
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
        self = StarScene(sector=sector, camera=camera, ccd=ccd, cutout_size=cutout_size)
        self._load_background(row=np.arange(cutout_size), column=np.arange(cutout_size))
        return self

    @property
    def locs(self):
        """The locations of each of the cutouts. List of lists. Format is

        [[mininmum row, maxmimum row], [minimum column, maximum column]]
        """
        locs = []
        nbatches = 2048 // self.cutout_size
        for bdx1 in range(nbatches):
            for bdx2 in range(nbatches):
                locs.append(
                    [
                        [self.cutout_size * bdx1, self.cutout_size * (bdx1 + 1)],
                        [self.cutout_size * bdx2, self.cutout_size * (bdx2 + 1)],
                    ]
                )
        return locs

    def load(self, input, dir=None):
        """
        Load a model fit from the data directory.

        Parameters
        ----------
        input: tuple or string
            Either pass a tuple with `(sector, camera, ccd)` or pass
            a file name in `dir` to load
        dir : str
            Optional tring with the directory name

        Returns
        -------
        self: `scatterbrain.StarScene` object
        """
        if isinstance(input, tuple):
            if len(input) == 3:
                sector, camera, ccd = input
                fname = f"tessstarscene_sector{sector}_camera{camera}_ccd{ccd}.fits"
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
            ]:
                setattr(self, key, hdu[0].header[key])
            setattr(self, "cutout_size", hdu[0].header["CUTSIZE"])

            self.weights = [hdu[1].data, hdu[2].data]
        return self

    @staticmethod
    def from_disk(sector, camera, ccd, dir=None):
        return StarScene(sector=sector, camera=camera, ccd=ccd, cutout_size=16).load(
            (sector, camera, ccd), dir=dir
        )

    def _package_weights_hdulist(self):
        hdu0 = self.hdu0
        hdu1 = fits.ImageHDU(np.asarray(self.weights[0]), name="ORBIT1")
        hdu2 = fits.ImageHDU(np.asarray(self.weights[1]), name="ORBIT2")
        hdul = fits.HDUList([hdu0, hdu1, hdu2])
        return hdul

    def save(self, output_dir=None, overwrite=False):
        """Save a StarScene"""
        self.hdu0 = fits.PrimaryHDU()
        self.hdu0.header["ORIGIN"] = "scatterbrain"
        self.hdu0.header["AUTHOR"] = "christina.l.hedges@nasa.gov"
        self.hdu0.header["VERSION"] = __version__
        for key in ["sector", "camera", "ccd"]:
            self.hdu0.header[key] = getattr(self, key)
        self.hdu0.header["CUTSIZE"] = getattr(self, "cutout_size")

        if output_dir is None:
            output_dir = f"{PACKAGEDIR}/data/sector{self.sector:03}/camera{self.camera:02}/ccd{self.ccd:02}/"
            if output_dir != "":
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

        hdul = self._package_weights_hdulist()
        fname = (
            f"tessstarscene_sector{self.sector}_camera{self.camera}_ccd{self.ccd}.fits"
        )
        hdul.writeto(output_dir + fname, overwrite=overwrite)
        return

    def get_images(self, fnames, loc, orbit=1):
        """Load the TESS FFIs, remove the best fitting scattered light model, and
        the best "average" frame, and return as an array.


        Parameters
        ----------
        fnames : list of str
            File names to use to build the object
        loc : list of lists
            Location of the cut out in format [[min row, max row], [min col, max col]]
        orbit: int, [1 or 2]
            Which orbit to get
        """
        if loc != [
            [self.background.row[0], self.background.row[-1] + 1],
            [self.background.column[0], self.background.column[-1] + 1],
        ]:
            self._load_background(row=np.arange(*loc[0]), column=np.arange(*loc[1]))
        y = np.zeros(
            (
                (self.background.quality[self.orbit_masks[orbit - 1]] == 0).sum(),
                *np.diff(loc).ravel(),
            )
        )
        idx = 0
        for tdx in np.where(self.orbit_masks[orbit - 1])[0]:
            if self.background.quality[tdx] != 0:
                continue
            y[idx] = (
                load_image(np.asarray(fnames)[tdx], loc=loc)
                - self.background.model(tdx)
                - self.background.average_frame
            )
            idx += 1
        return y

    def _mask_asteroids(self, flux_cube, loc, orbit):
        """Zero out any asteroids in a flux cube.

        Parameters
        ----------
        flux_cube : np.ndarray
            3D flux object, with shape [time, row, column]
        loc : list of lists
            Location of the cut out in format [[min row, max row], [min col, max col]]
        orbit: int, [1 or 2]
            Which orbit you are processing
        """
        tmask = (self.background.quality == 0) & self.orbit_masks[orbit - 1]
        row, col = get_asteroid_locations(
            self.sector, self.camera, self.ccd, times=self.tstart[tmask]
        )
        t = np.arange(row.shape[1])

        for idx in np.arange(row.shape[0]):
            k = (
                (row[idx] >= loc[0][0])
                & (row[idx] < loc[0][1])
                & (col[idx] >= loc[1][0])
                & (col[idx] < loc[1][1])
            )
            if not k.any():
                continue
            rdx, cdx = row[idx][k] - loc[0][0], col[idx][k] - loc[1][0]
            RDX, CDX = np.array_split(
                np.vstack(
                    [
                        minmax(v + offset, flux_cube.shape[1])
                        for v in [rdx, cdx]
                        for offset in np.arange(-3, 4)
                    ]
                ),
                2,
            )
            bad_pix = [np.vstack([r, c]) for r in RDX for c in CDX]
            for bdx in range(len(bad_pix)):
                flux_cube[t[k], bad_pix[bdx][0], bad_pix[bdx][1]] = 0

    def _fill_weights_block(self, fnames, loc, iter=False):
        """Process a cutout of the TESS FFIs. Will process the `loc` cutout.

        Parameters
        ----------
        fnames : list of str
            File names to use to build the object
        loc : list of lists
            Location of the cut out in format [[min row, max row], [min col, max col]]
        iter: bool
            Whether to iterate the fit. If True, will repeat the fit, masking
            the largest values in each pixel time series.
        """
        for orbit in [1, 2]:
            y = self.get_images(fnames, loc, orbit=orbit)
            self._mask_asteroids(y, loc=loc, orbit=orbit)
            s = (y.shape[0], np.product(y.shape[1:]))
            X = self.Xs[orbit - 1][
                self.background.quality[self.orbit_masks[orbit - 1]] == 0
            ]
            ws = np.linalg.solve(X.T.dot(X), X.T.dot(y.reshape(s)))
            if iter:
                res = y - X.dot(ws).reshape(y.shape)
                res[
                    sigma_clip(
                        np.abs(res).sum(axis=(1, 2)), sigma_upper=5, sigma_lower=1e10
                    ).mask
                ] = 0
                for idx in range(5):
                    k = res == np.max(res, axis=0)
                    k = (
                        k
                        | np.vstack([k[1:], k[0][None, :, :]])
                        | np.vstack([k[-1][None, :, :], k[:-1]])
                    )
                    res[k] = 0
                ws2 = np.linalg.solve(X.T.dot(X), X.T.dot((y * (res != 0)).reshape(s)))
                self.weights[orbit - 1][
                    :, loc[0][0] : loc[0][1], loc[1][0] : loc[1][1]
                ] = ws2.reshape((ws.shape[0], *y.shape[1:])).astype(np.float32)
            else:
                self.weights[orbit - 1][
                    :, loc[0][0] : loc[0][1], loc[1][0] : loc[1][1]
                ] = ws.reshape((ws.shape[0], *y.shape[1:])).astype(np.float32)

    @staticmethod
    def fit_model(fnames, cutout_size=512):
        """Fit a StarScene model to a set of TESS filenames"""
        self = StarScene.from_tess_images(fnames, cutout_size=cutout_size)
        for loc in tqdm(self.locs):
            self._fill_weights_block(fnames=fnames, loc=loc)
        return self

    def model(self, row, column, time_indices=None):
        """Returns a model for a given row and column.

        Parameters
        ----------
        row: np.ndarray
            Row to evaluate at
        column: np.ndarray
            Column to evaluate at
        time_indices: None, int, list of int
            Which indices to evaluate the model at. If None will use all.

        Returns
        -------
        model : np.ndarray
            Array of shape (time_indices, row, column) which has the full scene
            model, including scattered light.
        """
        if time_indices is None:
            time_indices = np.arange(self.shape[0])
        single_frame = False
        if isinstance(time_indices, int):
            time_indices = [time_indices]
            single_frame = True
        self._load_background(row=row, column=column)
        w1 = self.weights[0][:, row][:, :, column].reshape(
            (self.weights[0].shape[0], row.shape[0] * column.shape[0])
        )
        w2 = self.weights[1][:, row][:, :, column].reshape(
            (self.weights[1].shape[0], row.shape[0] * column.shape[0])
        )
        xmask1 = np.in1d(np.where(self.orbit_masks[0])[0], time_indices)
        xmask2 = np.in1d(np.where(self.orbit_masks[1])[0], time_indices)
        model = np.vstack([self.Xs[0][xmask1].dot(w1), self.Xs[1][xmask2].dot(w2)])
        model = model.reshape((model.shape[0], row.shape[0], column.shape[0]))
        background = np.asarray([self.background.model(tdx) for tdx in time_indices])
        if single_frame:
            return model[0] + self.background.average_frame + background[0]
        return model + self.background.average_frame + background

    def model_moving(self, row3d, column3d, star_model=True):
        if (row3d.shape[0] != self.shape[0]) | (column3d.shape[0] != self.shape[0]):
            raise ValueError("Pass Row and Column positions for all times. ")
        model = np.zeros((row3d.shape[0], row3d.shape[1], column3d.shape[1]))
        for orbit in [0, 1]:
            mask = self.orbit_masks[orbit]
            s = (self.weights[orbit].shape[0], row3d.shape[1] * column3d.shape[1])
            r1, r2 = row3d[mask].min(), row3d[mask].max() + 1
            c1, c2 = column3d[mask].min(), column3d[mask].max() + 1

            self._load_background(np.arange(r1, r2), np.arange(c1, c2))
            background = np.asarray(
                [self.background.model(tdx) for tdx in np.where(mask)[0]]
            )
            for idx, tdx, row, column in zip(
                np.arange(mask.sum()), np.where(mask)[0], row3d[mask], column3d[mask]
            ):

                model[tdx] = (
                    +background[idx][row - r1][:, column - c1]
                    + self.background.average_frame[row - r1][:, column - c1]
                )
                if star_model:
                    w1 = self.weights[orbit][:, row][:, :, column].reshape(s)
                    model[tdx] += (
                        self.Xs[orbit][idx]
                        .dot(w1)
                        .reshape((row.shape[0], column.shape[0]))
                    )
        return model
