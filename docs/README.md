# scatterbrain
<a href="https://github.com/christinahedges/scatterbrain/workflows/tests.yml"><img src="https://github.com/christinahedges/scatterbrain/workflows/pytest/badge.svg" alt="Test status"/></a> <a href="https://github.com/christinahedges/scatterbrain/workflows/flake8.yml"><img src="https://github.com/christinahedges/scatterbrain/workflows/flake8/badge.svg" alt="flake8 status"/></a>[![Generic badge](https://img.shields.io/badge/documentation-live-blue.svg)](https://christinahedges.github.io/scatterbrain)

`scatterbrain` is our GPU hack for processing TESS images, see [tess-backdrop](https://ssdatalab.github.io/tess-backdrop/) for our current tool.

# TODO

* add cholesky solve?
* Make starscene cupy
* Make asteroid_mask supersampled in time for fast, bright asteroids...?
* Add files on Zenodo so we can load them if they aren't local
* Add a TESSCut Corrector class
* Consider swapping third order polynomial in StarScene to a low order spline for nicer priors and better variability removal?
