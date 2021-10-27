# ScatteredLightBackground Flowchart

Below is a flowchart showing the decisions made in the [`ScatteredLightBackground`](background.md) class. This flowchart shows how `scatterbrain` converts TESS FFI images from MAST into a *fits* file containing weights. In this case we are fitting a **2D model** to each TESS FFI **frame**. We are allowing for correlations in pixels in the **spatial** dimension. The resultant weight file can be combined with [`design_matrix`](design_matrix.md) objects to create a model of the TESS scattered light at any pixel, at any time. The tool will also extract a set of `jitter` components. These components are the top principal components of 5000 test pixel time series, chosen to be close to stars in the image. These jitter components can be used to correct pixel time series in TESS data. These components are used in [`StarScene`](scene.md) to remove stars in the TESS FFIs, which is described below.

{%
   include-markdown "flowchart1.md"
%}

# StarScene Flowchart


Below is a flowchart showing the decisions made in the [`StarScene`](scene.md) class. This flowchart shows how [`StarScene`](scene.md) removes the stars from TESS FFI images, after the scattered light has been modeled with the  [`ScatteredLightBackground`](background.md). In the case of [`StarScene`](scene.md) we are fitting a **1D model** to each TESS FFI **pixel**. We are allowing for correlations in pixels in the **temporal** dimension.

[`StarScene`](scene.md) fits each TESS **orbit** separately. The final weight file fit by this class contains 2 (43 x 2048 x 2048) image stacks, i.e. one 2048 x 2048 pixel image per each of the 43 weights. There are two of these sets, because there are two orbits per sector. 

{%
   include-markdown "flowchart2.md"
%}
