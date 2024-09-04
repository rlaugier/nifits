# A standard to handle nulling interferometry data

## The `nifits` package

The `nifits` package has two roles:

* Help create and manipulate NIFITS files in python with the `io` module
* Offer a simple backend to use the instrument *model in a kit* packaged within the files with the `backend` module.

The documentation is a work in progress and can be found here: [API documentation](https://rlaugier.github.io/nifits_doc.github.io/)
The basic functionalities of the package are demonstrated in the `examples/quick_start.ipynb` notebook.

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

## Provisional definition (as of 06/2024)

### Basic working principle

To allow for full flexibility and the behavior of the instrument is packaged under the form of a complex amplitude transfer matrix (see e.g. Guyon 2013) which describes the linear combinations of complex input electric fields into complex output electric fields. This matrix has a column for each input (and therefore collecting telescope) and a row for each output. This matrix is stored for each wavelength into a datacube in the `NI_CATM` extension.

The vector of complex amplitude inputs can be computed for each elementary source of the field of view, in particular based on its position in the field of view ($\alpha$, $\beta$), which gives it a given phase based on the relative projected position of each collector ($x$, $y$) stored for each frame into the `NI_MOD` extension described below. This extension also contains for each frame the timestamps and the complex phasors expressing the state of a potential internal modulation applied by the instrument. Additional phasors representing the field of view falloff (expressed in `NI_FOV`) are also applied to the individual input electric fields.

The forward model of an observation can be computed by multiplying the resulting vector of complex input electric fields by the transfer matrix of the combiner to obtain its outputs, and taking its square modulus corresponds to the output intensity. Doing that for each spectral channel gives the forward model that can be compared to the direct observation output data stored in `NI_IOUT`.

For the case of staged beam combiners (double Bracewells, kernel nullers...) that use differential output the "kernel" matrix can be provided in the `NI_KMAT` under the form of a matrix operating on output intensities. The vector resulting from this additional transformation can then be compared to the processed data stored into the `NI_KIOUT` extension.

### Description of the main extensions

*Table 1: Summary of the NIFITS extensions*

|  Extension  |  Required   |  Content |
| ----------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  `OI_ARRAY` | fixed       |  Interferometer description for compatibility with OIFITS. |
|  `OI_WAVELENGTH` | fixed  |  Contains the wavelength information for the observation |
|  `NI_MOD`   | varying     |  Contains the time-varying information of the model, in particular the an interna modulation phasor vector, and the projected location of collecting apertures. |
|  `NI_CATM`  | fixed       |  The complex amplitude transfer matrix containing all static behavior of the system. |
|  `NI_KMAT`  | fixed       |  Identity is assumed if absent. |
|  `NI_IOUT`  | measuremetn |  Contains the collected output flux. |
|  `NI_KIOUT` | measurement |  Contains post-processed output fluxes. |
|  `NI_OSAMP` | fixed       |  Identity is assumed if absent, (optional). |
|  `NI_FOV`   | varying     |  Contains the complex spatial filtering function. |

*Table 2: Content of the NI_MOD data table*

|  Item        |  format                       |  unit            | comment |
| ------------ | ----------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------- |
|  `APP_INDEX` |  $n_a \times $$int            |  NA              | Indices of subaperture (starts at 0) |
|  `TARGET_ID` |  int                          |  d               | Index of target in `OI_TARGET` |
|  `TIME`      |  float                        |  s               | Backwards compatibility |
|  `MJD`       |  float                        |  day             |  |
|  `INT_TIME`  |  float                        |  s               | Exposure time |
|  `MOD_PHAS`  |  $n_{\lambda} \times n_a_\times $ complex |                  | Complex phasor of modulation for the collector |
|  `APPXY`     |  $n_a \times 2 \times $ float |  m               | Projected location of subapertures in the plane orthogonal to the line of sight and oriented as $(\alpha, \delta)$ |
|  `ARRCOL`    |  $n_a \times $ float          |  $\mathrm{m}^2$  | Collecting area of the subaperture |
|  `FOV_INDEX` |  $n_a \times$ $ int           |  NA              | The entry of the `NI_FOV` to use for this subaperture. |

Important implementation hints:
`NI_CATM` contains the static properties of the combiner that should rarely need to be updated.
`NI_MOD` contains phasors applied to each inputs, and that can be different for each temporal data point. It also contains the projected location of the collecting elements into the plane orthogonal to the line of sight.

Additional data still under discussion:

* Raw null posterior estimate:
  - Error estimation of raw outputs (`NI_IOUT`) is complicated because it is non-Gaussian. Since the nulling self-calibration computes a posterior PDF of the outputs, providing a PDF to the user seems relevant.
  - Error estimates are of differential outputs are close to Gaussian but are correlated in wavelength and between output pairs. Providing a covariance matrix estimate is relevant in that case, but this requires unambiguous flattening of these dimensions.
* Handling of polarization information
  - Expansion of the current system should allow the standard to handle polarized light.
* Handling of reference star calibration

## The NIFITS team

R. Laugier, J. Kammerer, M.-A. Martinod, F. Dannert, P. Huber


## Acknowledgement

NIFITS is a development carried out in the context of the [SCIFY project](http://denis-defrere.com/scify.php). [SCIFY](http://denis-defrere.com/scify.php) has received funding from the **European Research Council (ERC)** under the European Union's Horizon 2020 research and innovation program (*grant agreement No 866070*).  Part of this work has been carried out within the framework of the NCCR PlanetS supported by the Swiss National Science Foundation under grants 51NF40_18290 and 51NF40_205606.


Although this project has initially contained a branch of the OIFITS package from Paul Boley, the project has since evolved in entirely different implementation, and essentially all this original source code has disappeared.