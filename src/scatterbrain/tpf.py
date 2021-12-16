"""Tools to work with TPFs"""
from .scene import StarScene
from .cupy_numpy_imports import load_image, xp


def correct_tpf(tpf, jitter=True):
    """Correct a TESS Target Pixel File

    Parameters
    ----------
    tpf: lightkurve.TargetPixelFile
        TESSCut target pixel file.
    """
    ss = StarScene.from_tpf(tpf)
    time_mask = ss.get_tpf_mask(tpf)
    quality_mask = ss.quality_mask(175)
    if jitter:
        model = ss.model(mask_bad_pixels=False)
        model -= ss.background.average_frame
    else:
        model = ss.background.model()
    model -= xp.nanmedian(model, axis=0)
    return (tpf[time_mask] - model)[quality_mask]
