#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

import pickle  # noqa

from .background import ScatteredLightBackground  # noqa
from .cupy_numpy_imports import load_image  # noqa
from .scene import StarScene  # noqa
from .version import __version__  # noqa
from .tpf import correct_tpf

sector_times = pickle.load(open(f"{PACKAGEDIR}/data/tess_sector_times.pkl", "rb"))
