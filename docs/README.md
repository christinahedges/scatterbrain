# scatterbrain
<a href="https://github.com/christinahedges/scatterbrain/workflows/tests.yml"><img src="https://github.com/christinahedges/scatterbrain/workflows/pytest/badge.svg" alt="Test status"/></a> <a href="https://github.com/christinahedges/scatterbrain/workflows/flake8.yml"><img src="https://github.com/christinahedges/scatterbrain/workflows/flake8/badge.svg" alt="flake8 status"/></a>[![Generic badge](https://img.shields.io/badge/documentation-live-blue.svg)](https://christinahedges.github.io/scatterbrain)

`scatterbrain` is our GPU hack for processing TESS images, see [tess-backdrop](https://ssdatalab.github.io/tess-backdrop/) for our current tool.

# TODO

* Make starscene cupy
* Add files on Zenodo so we can load them if they aren't local
* Add docs to asteroid functions
* Add some tutorials for different use cases
* Come up with some diagnostics for backdrops/starscenes
* Check starscenes for multiple sectors
* Run on supercomputer
* Investigate using GPUs to run on super computer
* Increase the asteroid masking depth to mask fainter asteroids?
* Make asteroid_mask supersampled in time for fast, bright asteroids...?

# NOTES:
* Consider swapping third order polynomial in StarScene to a low order spline for nicer priors and better variability removal?
  - Did this, now has the option. Overall, seems better to just use a polynomial.
