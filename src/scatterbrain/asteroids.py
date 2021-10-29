"""Tools that require the internet, and get asteroids"""
import pickle

import numpy as np
import pandas as pd
from astropy.time import Time
from tqdm import tqdm

from . import PACKAGEDIR
from .utils import minmax


def get_ffi_times():
    """Gets the times of all the FFI images from MAST by looking at all the file names."""
    from urllib.request import HTTPError

    """Get a dictionary of the times for each TESS FFI and save it as a pkl file"""
    time_dict = {}
    for sector in tqdm(np.arange(1, 300), total=300):
        try:
            df = pd.read_csv(
                f"https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{sector}_ffir.sh",
                skiprows=1,
                header=None,
            )
        except HTTPError:
            break
        df[["time", "camera", "ccd"]] = np.asarray(
            [
                (int(d[0][20:33]), int(d[0][40]), int(d[0][42]))
                for idx, d in df.iterrows()
            ]
        )
        time_dict[sector] = (
            Time.strptime(
                np.sort(df[(df.camera == 1) & (df.ccd == 1)].time).astype(str),
                "%Y%j%H%M%S",
            ).jd
            + 0.000809
            - 2457000
        )
    pickle.dump(time_dict, open(f"{PACKAGEDIR}/data/tess_sector_times.pkl", "wb"))


def get_asteroid_files(catalog_fname, sectors, magnitude_limit=18):
    """Get files for each sector containing asteroid locations in the image."""
    import os

    import tess_ephem as te

    sector_times = pickle.load(open(f"{PACKAGEDIR}/data/tess_sector_times.pkl", "rb"))
    df_raw = pd.read_csv(catalog_fname, low_memory=False)
    for sector in np.atleast_1d(sectors):
        df = (
            df_raw[
                (df_raw.max_Vmag != 0)
                & (df_raw.sector == sector)
                & (df_raw.max_Vmag <= magnitude_limit)
            ]
            .drop_duplicates("pdes")
            .reset_index(drop=True)
        )
        t = Time(sector_times[sector] + 2457000, format="jd")
        t += np.median(np.diff(t.value)) / 2

        asteroid_df = pd.DataFrame(
            columns=np.hstack(
                [
                    "camera",
                    "ccd",
                    "vmag",
                    [f"{i}r" for i in np.arange(len(t))],
                    [f"{i}c" for i in np.arange(len(t))],
                ]
            ),
            dtype=np.int16,
        )
        names = []
        jdx = 0
        for idx, d in tqdm(df.iterrows(), total=len(df), desc=f"Sector {sector}"):
            ast = te.ephem(
                d.pdes, interpolation_step="6H", time=t, sector=sector, verbose=True
            )[["sector", "camera", "ccd", "column", "row", "vmag"]]

            ast.replace(np.nan, -1, inplace=True)
            for camera in ast.camera.unique():
                for ccd in ast[ast.camera == camera].ccd.unique():
                    j = np.asarray((ast.camera == camera) & (ast.ccd == ccd))
                    k = np.in1d(t.jd, [i.jd for i in ast[j].index])
                    row, col = np.zeros((2, k.shape[0])) - 1
                    row[k] = ast[j].row
                    col[k] = ast[j].column
                    names.append(d.pdes)
                    if (ast["vmag"] > 0).sum() == 0:
                        vmagmean = -99
                    else:
                        vmagmean = np.round(ast["vmag"][ast["vmag"] > 0].mean())
                    asteroid_df.loc[jdx] = np.hstack(
                        [camera, ccd, vmagmean, row, col]
                    ).astype(np.int16)
                    jdx += 1
        path = f"{PACKAGEDIR}/data/sector{sector:03}/"
        if not os.path.isdir(path):
            os.mkdir(path)
        if os.path.isfile(f"{path}bright_asteroids.hdf"):
            os.remove(f"{path}bright_asteroids.hdf")
        asteroid_df.to_hdf(
            f"{path}bright_asteroids.hdf",
            **{"key": f"asteroid_sector{sector}", "format": "fixed", "complevel": 9},
        )


def get_asteroid_locations(sector=1, camera=1, ccd=1, times=None):
    """Get the row and column positions of asteroids in sector, camera, ccd

    Returns
    -------

    row:

    col:
    """
    df = pd.read_hdf(f"{PACKAGEDIR}/data/sector{sector:03}/bright_asteroids.hdf")
    df = df[(df.camera == camera) & (df.ccd == ccd)].reset_index(drop=True)

    vmag = np.asarray(df["vmag"])
    row = np.asarray(df)[:, 2:][
        :, np.asarray([d.endswith("r") for d in df.columns[2:]])
    ]
    col = (
        np.asarray(df)[:, 2:][:, np.asarray([d.endswith("c") for d in df.columns[2:]])]
        - 44
    )
    if times is None:
        time_mask = np.ones(row.shape[1], bool)
    else:
        sector_times = pickle.load(
            open(f"{PACKAGEDIR}/data/tess_sector_times.pkl", "rb")
        )[sector]
        time_mask = np.any(
            [np.isclose(sector_times, t, atol=1e-6) for t in times], axis=0
        )

    return vmag, row[:, time_mask], col[:, time_mask]


def get_asteroid_mask(sector=1, camera=1, ccd=1, cutout_size=2048, times=None):
    """Load a saved bright asteroid file as a 2048x2048 pixel mask.

    Use time mask to specify which times to use.
    """

    mask = np.zeros((cutout_size, cutout_size), bool)

    vmag, row, col = get_asteroid_locations(
        sector=sector, camera=camera, ccd=ccd, times=times
    )

    def func(row, col, ap=3):
        X, Y = np.mgrid[-ap : ap + 1, -ap : ap + 1]
        aper = np.hypot(X, Y) <= ap
        aper_locs = np.asarray(np.where(aper)).T - ap
        for idx in range(row.shape[0]):
            k = (
                (row[idx] >= 0)
                & (col[idx] > 0)
                & (row[idx] < cutout_size)
                & (col[idx] < cutout_size)
            )
            for loc in aper_locs:
                l1 = minmax(row[idx, k] + loc[0], shape=cutout_size)
                l2 = minmax(col[idx, k] + loc[1], shape=cutout_size)
                mask[l1, l2] = True

    func(row[vmag >= 14], col[vmag >= 14], ap=5)
    func(row[(vmag < 14) & (vmag >= 11)], col[(vmag < 14) & (vmag >= 11)], ap=7)
    func(row[(vmag < 11)], col[(vmag < 11)], ap=9)
    return mask
