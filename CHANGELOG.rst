=============
Release Notes
=============

Below are the notes from all logistigate releases.

Release 0.1.2
-------------

:Date: November 10, 2021

* README is updated.
* Unit tests are added for Jacobian and Hessian methods.
* Unit test is added for the SciPy optimizer.
* Unit tests are added for the NUTS and Langevin MCMC samplers.
* methods.FormEstimates() function for Lapalce approximation has optional arguments for capturing results from the SciPy optimizer as well as printing processing updates.
* utilities.testresultsfiletotable() can take Python list inputs.
* utilities.printEstimates() can provide results for a subset of indices.
* utilities.plotPostSamples() can provide interval plots, take index subsets for plotting, and handle user-input subtitles for plots.
* utilities.generateRandSystem() and .generateRandDataDict are added; these functions provide randomly generated two-echelon supply chains, and are useful for simulation studies.
* utilities.scorePostSamplesIntervals is added; this function provides scoring for intervals and underlying SFP rates.
* [TO-DO] All methods are updated to handle data obtained via different testing tools.
* [TO-DO] Examples, variables, and documentation are updated to reflect terminology of "test nodes," "supply nodes," and "sourcing probability."
* [TO-DO] Break utilities.generateRandSystem() into its component parts.
* [TO-DO] methods updated to properly import mcmsamplers.
* [TO-DO] lg updated to properly import methods and utilities.

Release 0.1.1
-------------

:Date: February 19, 2021

* Small changes are made to the example data sets.
* Documentation is added for the readthedocs online portal.

Release 0.1.0
-------------

:Date: February 18, 2021

* Initial release.
