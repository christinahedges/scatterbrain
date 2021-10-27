v 0.1.1
-------

* Added `asteroids` module, which will use `tess-locator` and `tess-ephem` to find asteroids in FFIs
* Added `scene` module which will correct the stars in the scene (including jitter)
* Added polynomial strap model
* Improved speed of data loading by using `fitsio.FITS` instead of `fitsio.read`
* Renamed `backdrop` to `background` and `BackDrop` to `ScatteredLightBackground`
* Improved the quality masking for bad FFI frames
* Added a `movie` function to the `utils` module for easy data inspection
* Added hdf5 files to module for asteroid locations in Cycles 1 and 2 down to 18th magnitude. This increases the space required for the module quite significantly, but will mean that we don't require internet connection if we run this on the supercomputer.
