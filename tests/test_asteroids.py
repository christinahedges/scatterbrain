import numpy as np

from scatterbrain import sector_times
from scatterbrain.asteroids import get_asteroid_locations, get_asteroid_mask


def test_asteroid_locs():
    vmag, row, col = get_asteroid_locations(1, 1, 3)
    assert len(row.shape) == 2
    assert row.shape == col.shape
    assert isinstance(row[0, 0], np.int16)
    assert row.shape[1] == len(sector_times[1])


def test_asteroid_mask():
    mask = get_asteroid_mask(1, 1, 3)
    assert mask.shape == (2048, 2048)
    assert mask.sum() != 0
    mask2 = get_asteroid_mask(1, 1, 3, times=sector_times[1][:10])
    assert mask2.shape == (2048, 2048)
    assert mask2.sum() < mask.sum()

    mask3 = get_asteroid_mask(1, 1, 3, cutout_size=256)
    assert mask3.shape == (256, 256)
