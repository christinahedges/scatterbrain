# flake8: noqa
import os

import fitsio
import numpy as np

try:
    if os.getenv("USE_CUPY") in ["True", "T", "true"]:
        import cupy as xp
        from cupy import sparse

        def load_image_cupy(fname, cutout_size=2048):
            return xp.asarray(load_image_numpy(fname, cutout_size=cutout_size))

        def load_image(fname, cutout_size=2048):
            return load_image_cupy(fname, cutout_size=cutout_size).astype(xp.float32)

    else:
        raise ImportError

except ImportError:
    import numpy as xp
    from scipy import sparse

    def load_image(fname, cutout_size=2048):
        return load_image_numpy(fname, cutout_size=cutout_size).astype(xp.float32)


def load_image_numpy(fname, cutout_size=2048):
    image = np.asarray(fitsio.read(fname)[:cutout_size, 45 : 45 + cutout_size])
    image[~np.isfinite(image)] = 1e-5
    image[image <= 0] = 1e-5
    return image
