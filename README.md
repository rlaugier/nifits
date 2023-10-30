# A standard to handle nulling interferometry data

## Spirit

This standard library aims to facilitate the exchange of nulling interferometry data and the proliferation of nulling data reduction methods among the community. It should make available all the instrumental information necessary to deploy the most advanced data reduction algorithms. It should be suitable to hold the raw data (to the exception of detector data) for reduction, or the reduced and co-added data for interpretation.

It should rely on OIFITS standard data format as much possible so as to facilitate reuse of the relevant code base. It will provide the additional information relevant to the specificities of nulling interferometry. Candidates for OIFITS reference:

* [oifits-sim, by bkloppenborg](https://github.com/bkloppenborg/oifits-sim)
* [oifits, by Paul Boley](https://github.com/pboley/oifits/forks)

Nulling interferometry can take many forms. Simple Bracewell, Double Bracewell, Kernel Nuller, active chopping etc. For this reason, the data is useless without the corresponding description of the instrument.

## Requirements

### Common to reduction and interpretation stage
The data standard shoudl be compatible with ground-based existing facilities and space-based instrument, with minimal 

* The `OI_ARRAY`. **A solution must be found for arbitrary motion of the array** as possible with space formation flying.
  - I think some of this should already be possible in the OIFITS framework. Typically, each integration has its own UV-coordinates assigned to it because the Earth also rotates and the UV-coordinates evolve throughout a night. The thing that might be an issue is if the arrays move significantly within an individual integration. In that case, one would need to save the path of the motion, so that reduction/calibration algorithms can simulate the same path for e.g., model observables.
* The `OI_TARGET`
* The `OI_WAVELENGTH`
* Data of the combiner complex amplitude transfer matrix as it is best definde by simulations or measured in calibration. This constitutes the *static part of the instrumental function*.
* Data of additional input phasor as a function of time, constituting the *variable part of the instrumental function*.
  - Should include spectral information
  - Could be described either:
    + The phase for each wavelength
    + The OPL of different materials (air, glass, gaz) of the combination (is this too instrument-specific?)
  - Should include polarization information
* The intensity estimates 
* An error estimation:
  - The variance of each intensity estimates
  - The covariance matrix of intensity estimates (here we need to think carefully about which correlations we want to keep track of; OIFITS itself is very vague here and allows correlation matrices of many different shapes, depending on which level of correlations they describe (only within a single integration, or across different observables, or across time). However, this makes it extremely complicated to write a code that does something with the correlations, because you never know which kind of correlations to expect from the data in advance. This will have a big impact in the most efficient way to store your data and perform model fitting though).


### For reduction stage

* The intensity estimates for all the detector reads

### For interpretation stage


## Acknowledgement

NOIFITS is a development carried out in the context of the [SCIFY project](http://denis-defrere.com/scify.php). [SCIFY](http://denis-defrere.com/scify.php) has received funding from the **European Research Council (ERC)** under the European Union's Horizon 2020 research and innovation program (*grant agreement No 866070*).  

