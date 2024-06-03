# A standard to handle nulling interferometry data

## Spirit

This data standard aims to facilitate the exchange of nulling interferometry data and the proliferation of nulling data reduction methods among the community. It should make available all the instrumental information necessary to deploy the most advanced data reduction algorithms. It should be suitable to hold the raw data (to the exception of detector data) for reduction, or the reduced and co-added data for interpretation.

Nulling interferometry can take many forms. Simple Bracewell, Double Bracewell, Kernel Nuller, active chopping etc. For this reason, the data is useless without the corresponding description of the instrument.

The work of this consortium will focus on laying out and demonstrating the principle of operation of the NIFITS standard, including the respective roles of *the creator* of files *the user* of files, and third party libraries.

## Requirements
### Top level
The data standard should be compatible with ground-based existing facilities and space-based instrument.

It should rely on OIFITS standard data format as much possible so as to facilitate reuse of the relevant code base. It will provide the additional information relevant to the specificities of nulling interferometry. Candidates for OIFITS reference:

* [oifits-sim, by bkloppenborg](https://github.com/bkloppenborg/oifits-sim)
* [oifits, by Paul Boley](https://github.com/pboley/oifits/forks)


### Creator, user, and library.

The goal is to allow interpretation by scientific teams outside of the expert instrument team. For this the standard must ensure that the *the user* can perform interpretation of the data without any expertise in the instrument.

The responsibility of expertise in the instrument lies onto *the creator*, and in particular on the data recording, and preliminary reduction stage, which are typically built by the instrument team. The standard dictates in what form they should save the metadata.

The standard dictates the normalized algorithm using the metadata as an instrument simulator to be used by *the user* in interpretation and inference. This operation is ensured by third party *libraries* following the guidelines of the standard, serving in ways similar to **OIMODELER** or **PMOIRED**.


### Common to reduction and interpretation stage
Since part of the metadata must be recorded during the acquisition, it makes sense that partial forms of the standard be created durin acquisition to be completed during the preliminary reduction, before archiving.

## Provisional definition (06/2024)

*Table 1: Summary of the NIFITS extensions*

|  Extension  |  Required   |  Content |
| ----------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  `OI_ARRAY` |  yes        |  Interferometer description for compatibility with OIFITS. |
|  `NI_MOD`   |  yes        |  Contains the time-varying information of the model, in particular the an interna modulation phasor vector, and the projected location of collecting apertures. |
|  `NI_CATM`  |  referenced |  The complex amplitude transfer matrix containing all static behavior of the system. |
|  `NI_KMAT`  |  no         |  Identity is assumed if absent. |
|  `NI_IOUT`  |  yes        |  Contains the collected output flux. |
|  `NI_KIOUT` |  no         |  Contains post-processed output fluxes. |
|  `NI_OSAMP` |  no         |  Identity is assumed if absent. |
|  `NI_FOV`   |  referenced |  Contains the complex spatial filtering function. |

*Table 2: Content of the NI_MOD data table*

|  Item        |  format                       |  unit            | comment |
| ------------ | ----------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------- |
|  `APP_INDEX` |  int                          |  NA              | Index of subaperture (starts at 0) |
|  `TARGET_ID` |  int                          |  d               | Index of target in `OI_TARGET` |
|  `TIME`      |  float                        |  s               | Backwards compatibility |
|  `MJD`       |  float                        |  day             |  |
|  `INT_TIME`  |  float                        |  s               | Exposure time |
|  `MOD_PHAS`  |  $n_{\lambda} \times$ complex |                  | Complex phasor of modulation for the collector |
|  `APPXY`     |  $2 \times$ float             |  m               | Projected location of subapertures in the plane orthogonal to the line of sight and oriented as $(\alpha, \delta)$ |
|  `ARRCOL`    |  float                        |  $\mathrm{m}^2$  | Collecting area of the subaperture |
|  `FOV_INDEX` |  int                          |  NA              | The entry of the `NI_FOV` to use for this subaperture. |

Important implementation hints:
`NI_CATM` contains the static properties of the combiner that should rarely need to be updated.
`NI_MOD` contains phasors applied to each inputs, and that can be different for each temporal data point. It also contains the projected location of the collecting elements into the plane orthogonal to the line of sight.

Additional data still under discussion:

* An error estimation:
  - Error estimation of raw outputs (`NI_IOUT`) is complicated because it is non-Gaussian. Since the nulling self-calibration computes a posterior PDF of the outputs, providing a PDF to the user seems relevant.
  - Error estimates are of differential outputs are close to Gaussian but are correlated in wavelength and between output pairs. Providing a covariance matrix estimate is relevant in that case, but this requires unambiguous flattening of these dimensions.
* Handling of polarization information
  - Expansion of the current system should allow the standard to handle polarized light.
* Handling of reference star calibration



## Acknowledgement

NOIFITS is a development carried out in the context of the [SCIFY project](http://denis-defrere.com/scify.php). [SCIFY](http://denis-defrere.com/scify.php) has received funding from the **European Research Council (ERC)** under the European Union's Horizon 2020 research and innovation program (*grant agreement No 866070*).  Part of this work has been carried out within the framework of the NCCR PlanetS supported by the Swiss National Science Foundation under grants 51NF40_18290 and 51NF40_205606.

