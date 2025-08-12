"""
NIFITS Extra
-------------

The NIFITS extra contains some additional tools to go somewhat **beyond** the
a simple implementation of the NIFITS standard, and facilitate the usage of 
common powerful techniques of nulling interferometry data reduction and
interpretation.

This includes:

* statistical whitening of observables (Ceau et al. 2019, Laugier et al. 2023)
* Correlation maps (work in progress)
* Test statistics Te and Tnp from (Ceau et al. 2019, Laugier et al. 2023)
* Monte-Carlo simulation of instrumental noise (coming soon)
* Various plotting macros (coming soon)

At present, this is done through a new class Post that inherits from Backend
offering dirctly whitened forward models prfixed with ``w_``, and ``NI_KIOUT``

The extra module is currently considered unstable. Please contact us if you make
use of its feature: we can give you warnings and help when our updates are at 
risk of breaking your code. It is likely that the installation of the extra
module becomes optional.

"""


# extra.py
import nifits.io as io
from nifits.io.oifits import NIFITS_EXTENSIONS, STATIC_EXTENSIONS
from nifits.io.oifits import nifits as NIFITSClass
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as units
import nifits.backend as be
from scipy.linalg import sqrtm
import types
from einops import rearrange
from copy import copy
from scipy.stats import ncx2 as ncx2
from scipy.stats import norm as norm
from scipy.stats import chi2 as chi2

def local_sqrtm(amatrix):
    """
    Arguments:
        amatrix : A square matrix

    Retrurns: 
        sqrt_matrx : The square root of the input
    """
    # Eigendecomposition
    evalues, evectors = np.linalg.eig(amatrix)
    # Checking for singularity
    assert not np.isclose(evalues, 0).any(), "The matrix is singular (or too close to it)"
    sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)
    return sqrt_matrix

class Post(be.NI_Backend):
    """
    This variant of the backend class offers a statistically whitened alternate
    forward model with directly whitened observables by calling ``w_`` prefixed
    methods. (Ceau et al. 2019, Laugier et al. 2023)

    After normal construction, use ``create_whitening_matrix()`` to update the
    whitening matrix based on the ``NI_KCOV`` data.
    
    Use ``w_get_all_outs`` and ``get_moving_outs`` in the same way, but they
    return whitened observables. Compare it to ``self.nifits.ni_kiout.w_kiout``
    instead of ``self.nifits.ni_kiout.kiout``.

    """
    def create_whitening_matrix(self,
                                replace: bool = False,
                                md: types.ModuleType = np,
                                local: bool = True):
        """
            Updates the whitening matrix:

        Args:
            replace : If true, the forward model methods are replaced by the
                whitened ones. The old ones get a ``old_`` prefix.
            md :  A numpy-like backend module.

        The pile of whitening matrices is stored as ``self.Ws`` (one for
        each frame).
        """
        self.md = md
        # Assertion: assert
        assert hasattr(self.nifits, "ni_kcov")
        if self.nifits.ni_kcov.header["NIFITS SHAPE"] != "frame (wavelength output)":
            raise NotImplementedError("Covariance shape expected: frame (wavelength output)")

        Ws = []
        for amat in self.nifits.ni_kcov.data_array:
            if local:
                Ws.append(np.linalg.inv(local_sqrtm(amat)))
            else:
                Ws.append(np.linalg.inv(sqrtm(amat)))
        self.Ws = md.array(Ws)
        self.W_unit = self.nifits.ni_kcov.unit**(-1/2)
        if hasattr(self.nifits, "ni_kiout"):
            self.datashape = (self.nifits.ni_kiout.shape) 
        self.flatshape = (len(self.nifits.ni_mod), self.nifits.ni_kcov.shape[1])
        self.fullyflatshape = np.prod(self.flatshape)

        if replace:
            # Backup the forward model mehtods, then decorate them
            self.old_get_all_outs = copy(self.get_all_outs)
            self.get_all_outs = self.whitened_outputs(self.get_all_outs)
            self.old_get_moving_outs = copy(self.get_moving_outs)
            self.get_moving_outs = self.whitened_outputs(self.get_moving_outs)

        else:
            # Adds new whitening methods
            self.w_get_all_outs = self.whitened_outputs(self.get_all_outs)
            self.w_get_moving_outs = self.whitened_outputs(self.get_moving_outs)
            
        if hasattr(self.nifits, "ni_kiout"):
            self.nifits.ni_kiout.Ws = self.Ws
            self.nifits.ni_kiout.md = self.md
            self.nifits.ni_kiout.__class__.w_kiout = whitened_kiout

    def whiten_signal(self, signal):
        """
        Whitens a signal so that error covariance
        is identity in the new axis.

        Args:
            signal: The direct signal to whiten (differential observable
                a.k.a kernel-null)

        Returns:
            wout_full: the whitened signal ($\\mathbf{W}\\cdot \\mathbf{s}$)
                in the new basis.

        """
        full_shape = signal.shape
        # Flatten the spectral and output dimension
        flat_full_shape = (*self.flatshape, full_shape[-1])
        flat_out = rearrange(signal, "frame wavelength output source -> frame (wavelength output) source")
        wout = self.md.einsum("f o i , f i m -> f o m", self.Ws, flat_out)
        return wout


    def whitened_outputs(self, func):
        """
        A decorator methods that gives applies statistical whitening
        to the output of the function. The whitening relies on 
        ``self.W`` which is computed as by ``create_whitening_matrix``.

        Args:
            func: The function to decorate.

        Returns:
            inner: The decorated function

        """
        def inner(*args, **kwargs):
            output = func(*args, **kwargs)
            wout_full = self.whiten_signal(output)
            return wout_full
        inner.__doc__ = """**Modified to whiten the observables
                            (output and wavelength dimensions are fused)**\n\n"""\
                    + func.__doc__
        return inner

    def add_blackbody(self, temperature):
        """
            Initializes the blackbody for a given temperature

        Args:
            temperature: units.Quantity

        The blackbody ``BB`` object is stored in ``self.bb``.
        """
        self.bb = BB(temperature)

    def get_pfa_Te(self, signal=None,
                        md=np):
        """
            Compute the Pfa for the energy detector test.
        Args:
            signal: The raw signal (non-whitened)

        Returns:
            pfa the false alarm probability (p-value) associated
                to the given signal.
        """
        w_signal = self.whiten_signal(signal)
        wf_signal = md.flatten(w_signal)
        pfa = 1 - ncx2.cdf(wf_signal, df=self.fullyflatshape, nc=0.)
        return pfa

    def get_pfa_Tnp(self, alphas, betas,
                    signal=None,
                    model_signal=None,
                        md=np):
        """
            Compute the Pfa for the Neyman-Pearson test.
        """
        pass


    def get_blackbody_native(self, ):
        """
        
        Returns:
            blackbody_spectrum: in units consistent with the native
                units of the file $[a.sr^{-1}.m^{-2}]$ (where [a] is the
        flux unit used in the file, typically [ph/s]).
        """
        # Typically given there in erg/Hz/s/sr/cm^2
        myspectral_density = self.bb(self.nifits.oi_wavelength.lambs * units.m)
        # Photon_energies in J/ph
        photon_energies = (self.nifits.oi_wavelength.lambs*units.m).to(
                            units.J, equivalencies=units.equivalencies.spectral())\
                                / units.photon
        dnus = self.nifits.oi_wavelength.dnus * (units.Hz)
        print((myspectral_density * dnus / photon_energies).unit)
        blackbody_spectrum = (myspectral_density * dnus / photon_energies).to(
                                 self.nifits.ni_iout.unit / units.sr / (units.m**2))
        return blackbody_spectrum


    def get_blackbody_collected(self, alphas, betas,
                        kernels=True, whiten=True,
                            to_si=True):
        """
            Obtain the output spectra of a blackbody at the given blackbody
        temperature.

        Args:
            alphas: ArrayLike: Relative position in rad
            betas:  ArrayLike: Relative position in rad
            kernels: Bool (True) Whether to work in the kernel postprocessing space
                (False is not implemented yet)
            whiten: Bool (True) whether to use whitening post-processing (False
                is not implemented yet)
            to_si: Bool (True) convert to SI units
        """
        collecting_map_q = units.m**2 * self.get_all_outs(alphas, betas, kernels=kernels)
        blackbody_spectrum = self.get_blackbody_native()
        collected_flux = blackbody_spectrum[None,:,None,None] \
                                * collecting_map_q[:,:,:,:]
        if whiten:
            blackbody_signal = self.W_unit * self.whiten_signal(collected_flux)
        else:
            blackbody_signal = collected_flux
        # collected_flux is in equivalent W / rad^2 
        # That is a power per solid angle of source
        
        if to_si:
            return blackbody_signal.to(blackbody_signal.unit.to_system(units.si)[0])
        else:
            return blackbody_signal

    def get_Te(self):
        """
            Computes the Te test statistic of the current file. This test statistic
        is supposed to be distributed as a chi^2 under H_0.
        Returns:
            Te : x.T.dot(x) where x is the whitened signal.
        """
        if hasattr(self.nifits, "ni_kiout"):
            kappa = self.whiten_signal(self.nifits.ni_kiout.kiout)
        else:
            raise NotImplementedError("Needs a NI_KIOUT extension")
        x = kappa.flatten()
        return x.T.dot(x)

    def get_pdet_te(self, alphas, betas,
                    solid_angle,
                    kernels=True, pfa=0.046,
                    temperature=None):
        """
            Computes a detection probability of a point-source at a given
        position based on the energy detector defined in Ceau et al. 2019.
        This is a simple test of deviation from the expected distribution,
        entirely agnostic to the family of signal of interest. The spectrum
        of the source is a blackbody spectrum of fixed temperature.

        Args: 
            alpha         : [rad] The coordinate matched to X in the array geometry
            beta          : [rad] The coordinate matched to Y in the array geometry
            solid_angle   : [sr] Solid angle of the source
            temperature   : [K] The temperature of the blackbody
            kernels       : Bool (True) Whether to use the K matrix to work in differential outputs
            pfa           : float The false-alarm rate (p-value) to consider

        Returns:
            pdet_pfa      : An array of the detection probability for all points.
        
        pfa:
        * 1 sigma: 0.32
        * 2 sigma: 0.046
        * 3 sigma: 0.0027
        """
        if temperature is not None:
            self.add_blackbody(temperature)
        ref_spectrum = solid_angle * self.get_blackbody_collected(alphas, betas,
                                                    kernels=kernels,
                                                    whiten=True,
                                                    to_si=True)
        print(ref_spectrum.unit)
        threshold = ncx2.ppf(1-pfa, df=self.fullyflatshape, nc=0.)
        x = ref_spectrum.reshape(-1, 1000)
        xTx = np.einsum("o m , o m -> m", x, x)
        pdet_Pfa = 1 - ncx2.cdf(threshold, self.fullyflatshape, xTx)
        return pdet_Pfa

    def get_sensitivity_te(self, alphas, betas,
                    kernels=True,
                    temperature=None, pfa=0.046, pdet=0.90,
                    distance=None, radius_unit=units.Rjup,
                    md=np):
        """
            Computes a sensitivity of the observation to point source(s) located
        at the given coordinates, based on the energy detector test defined in
        Ceau et al. 2019. This is a simple test of deviation from the expected
        Gaussian distribution, entirely agnostic to the family of signal of interest.
        The spectrum of the source is a blackbody spectrum of fixed temperature.
        The sensitivity is converted to a sphere diameter

        Args: 
            alpha         : [rad] The coordinate matched to X in the array geometry
            beta          : [rad] The coordinate matched to Y in the array geometry
            temperature   : [K] The temperature of the hypothetical source
            pfa           : (0.046) False alarm rate (p-value) for signficance
            pdet          : (0.9) Detection probability.
            distance      : (distance Quantity) Optional. The distance to the system.
                            If provided, used to translate the solid angle limit to
                            to a sphere radius limit.
            radius_unit   : (length Unit) Optional. The unit to express the
                            radius limit.
            kernels       : Bool (True) Whether to use the K matrix to work in differential outputs
            pfa           : float The false-alarm rate (p-value) to consider

        Returns:
            lim_radius : (astropy.Quantity) An estimate of the sensitivity limit
                            of the observation. If distance is not providede, limit
                            is the solid angle of the source seen from the observer.
                            If a distance is provided, the limit is converted to the
                            radius of a sphere at this distance
                        
        
        pfa:
        * 1 sigma: 0.32
        * 2 sigma: 0.046
        * 3 sigma: 0.0027
         
        .. code-block:: python

            from scipy.stats import ncx2
            xs = np.linspace(-10, 10, 100)
            ys = np.linspace(1e-6, 0.999, 100)
            u = 1 - ncx2.cdf(xs, df=10, nc=0)
            v = ncx2.ppf(1 - ys, df=10, nc=0)

            plt.figure()
            plt.plot(xs, u)
            plt.show()

            plt.figure()
            plt.plot(ys, v)
            plt.plot(u, xs)
            plt.show()
        """
        from scipy.optimize import leastsq
        if temperature is not None:
            self.add_blackbody(temperature)
        ref_spectrum = self.get_blackbody_collected(alphas=alphas,betas=betas,
                                                    kernels=kernels, whiten=True,
                                                    to_si=True)
        print("Ref spectrum unit: ", ref_spectrum.unit)
        threshold = ncx2.ppf(1-pfa, df=self.fullyflatshape, nc=0.)
        x = (ref_spectrum).reshape((-1, ref_spectrum.shape[-1]))
        print("Ref signal (x) unit: ", x.unit)
        print("Ref signal (x) shape: ", x.shape)
        xtx = md.einsum("m i, i m -> m", x.T, x)
        lambda0 = 1.0e-3 * self.fullyflatshape
        # The solution lambda is the x^T.x value satisfying Pdet and Pfa
        sol = leastsq(residual_pdet_Te, lambda0,
                        args=(threshold, self.fullyflatshape, pdet))# AKA lambda
        lamb = sol[0][0]
        # Concatenate the wavelengths
        lim_solid_angle = np.sqrt(lamb) / np.sqrt(xtx)
        if distance is None:
            return lim_solid_angle
        elif isinstance(distance, units.Quantity):
            dist_converted = distance.to(radius_unit)
            lim_radius = dist_converted*md.sqrt(lim_solid_angle/md.pi)
            return lim_radius.to(radius_unit,
                                equivalencies=units.equivalencies.dimensionless_angles())


    def get_tnp(self, x):
        """
        Untested
        
        Computes the Tnp test statistic of the current file. This test statistic
        is supposed to be distributed as a chi^2 under H_0.

        Arguments:
            x : Known signature of the signal of interest.
        
        Returns:
            Tnp : y.T.dot(x) where y is the whitened signal recorded
            and x is the target .
        """
        if hasattr(self.nifits, "ni_kiout"):
            kappa = self.whiten_signal(self.nifits.ni_kiout.kiout)
        else:
            raise NotImplementedError("Needs a NI_KIOUT extension")
        y = kappa.flatten()
        return y.T.dot(x)

    def get_pdet_tnp(self, alphas, betas,
                    solid_angle,
                    kernels=True, pfa=0.046,
                    whiten=True,
                    temperature=None):
        """
        Untested:

            Computes a detection probability of a point-source at a given
        position based on the energy detector defined in Ceau et al. 2019.
        This is a simple test of deviation from the expected distribution,
        entirely agnostic to the family of signal of interest. The spectrum
        of the source is a blackbody spectrum of fixed temperature.

        Args: 
            alpha         : [rad] The coordinate matched to X in the array geometry
            beta          : [rad] The coordinate matched to Y in the array geometry
            solid_angle   : [sr] Solid angle of the source
            temperature   : [K] The temperature of the blackbody
            whiten        : Bool (True) Whether to use the whitening transform
            kernels       : Bool (True) Whether to use the K matrix to work in differential outputs
            pfa           : float The false-alarm rate (p-value) to consider
        
        pfa:
        * 1 sigma: 0.32
        * 2 sigma: 0.046
        * 3 sigma: 0.0027
        """
        if temperature is not None:
            self.add_blackbody(temperature)
        ref_spectrum = solid_angle * self.get_blackbody_collected(alphas, betas,
                                                    kernels=kernels,
                                                    whiten=True,
                                                    to_si=True)
        print(ref_spectrum.unit)
        x = ref_spectrum.reshape(-1, 1000)
        xTx = np.einsum("o m , o m -> m", x, x)
        threshold = np.sqrt(xTx)*norm.ppf(1-pfa)
        # threshold = ncx2.ppf(1-pfa, df=self.fullyflatshape, nc=0.)
        pdet_Pfa = 1 - norm.cdf((threshold - xTx)/np.sqrt(xTx))
        return pdet_Pfa

    def get_sensitivity_tnp(self, alphas, betas,
                    kernels=True,
                    temperature=None, pfa=0.046, pdet=0.90,
                    distance=None, radius_unit=units.Rjup,
                    md=np):
        """
            Computes a sensitivity of the observation to point source(s) located
        at the given coordinates, based on the Neyman-Pearson test defined in
        Ceau et al. 2019. This is a idealized test that represents the
        theoretical limit if we perfectly know the signature of interest.
        The spectrum of the source is a blackbody spectrum of fixed temperature.
        The sensitivity is converted to a sphere diameter.

        Args: 
            alphas        : [rad] The coordinate matched to X in the array geometry
            betas         : [rad] The coordinate matched to Y in the array geometry
            temperature   : [K] The temperature of the hypothetical source
            pfa           : (0.046) False alarm rate (p-value) for signficance
            pdet          : (0.9) Detection probability.
            distance      : (distance Quantity) Optional. The distance to the system.
                            If provided, used to translate the solid angle limit to
                            to a sphere radius limit.
            radius_unit   : (length Unit) Optional. The unit to express the
                            radius limit. Default : Rjup
            kernels       : Bool (True) Whether to use the K matrix to work
                            in differential outputs
            pfa           : float The false-alarm rate (p-value) to consider

        Returns:
            lim_radius : (astropy.Quantity) An estimate of the sensitivity limit
                            of the observation. If distance is not providede, limit
                            is the solid angle of the source seen from the observer.
                            If a distance is provided, the limit is converted to the
                            radius of a sphere at this distance
        
        .. code-block:: python

            from scipy.stats import ncx2
            xs = np.linspace(-10, 10, 100)
            ys = np.linspace(1e-6, 0.999, 100)
            u = 1 - ncx2.cdf(xs, df=10, nc=0)
            v = ncx2.ppf(1 - ys, df=10, nc=0)

            plt.figure()
            plt.plot(xs, u)
            plt.show()

            plt.figure()
            plt.plot(ys, v)
            plt.plot(u, xs)
            plt.show()
        """
        if temperature is not None:
            self.add_blackbody(temperature)
        ref_solid_angle = 1e-20*units.rad**2
        ref_spectrum = ref_solid_angle * self.get_blackbody_collected(alphas=alphas,betas=betas,
                                                    kernels=kernels, whiten=True,
                                                    to_si=True)
        print("Ref spectrum unit: ", ref_spectrum.unit)
        x = (ref_spectrum).reshape((-1, ref_spectrum.shape[-1]))
        print("Ref signal (x) unit: ", x.unit)
        print("Ref signal (x) shape: ", x.shape)
        xTx = md.einsum("m i, i m -> m", x.T, x)
        threshold = np.sqrt(xTx)*norm.ppf(1-pfa)
        # threshold = ncx2.ppf(1-pfa, df=self.fullyflatshape, nc=0.)
        # lambda0 = 1.0e-3 * self.fullyflatshape
        f_map = ref_solid_angle\
                / np.sqrt(xTx) * ( norm.ppf(1 - pfa, loc=0, scale=1) - norm.ppf(1 - pdet, loc=0, scale=1))

        # The solution lambda is the x^T.x value satisfying Pdet and Pfa
        # sol = leastsq(residual_pdet_Tnp, lambda0,
        #                 args=(threshold, self.fullyflatshape, pdet))# AKA lambda
        # lamb = sol[0][0]
        # Concatenate the wavelengths

        # from pdb import set_trace
        # set_trace()
        lim_solid_angle = f_map
        if distance is None:
            return lim_solid_angle
        elif isinstance(distance, units.Quantity):
            dist_converted = distance.to(radius_unit)
            lim_radius = dist_converted*md.sqrt(lim_solid_angle/md.pi)
            return lim_radius.to(radius_unit, equivalencies=units.equivalencies.dimensionless_angles())


massq2sr = (units.mas**2).to(units.sr)
sr2massq = (units.sr).to(units.mas**2)

from astropy.modeling.models import BlackBody as BB
import astropy.constants as cst

def e2ph(energy, wl):
    """
        Utility conversion from energy to photons:

    Args:
        energy   : Carried energy [J]
        wl       : Wavelength [m]

    Returns:
        nphot    : number of photons
    """
    nu = cst.c / (wl * units.m)
    ephot = cst.h * nu
    nphot = energy/ephot
    return nphot.to(nphot.unit.to_system(units.si)[0])

def ph2e(nphot, wl):
    """
        Utility conversion from photons to energy  (SI)
    Args:
        nphot   : A number of photons
        wl      : Wavelength [m]
    Returns:
        Energy  : Converted to SI units
    """
    nu = cst.c / (wl * units.m)
    ephot = cst.h * nu
    energy = nphot * ephot
    return energy.to(energy.unit.to_system(units.si)[0])

def whitened_kiout(self):
    """
        The function that is imported a method to produce a 
    statistical whitening of observation. It is added to the
    object by `create_whitening_matrix`

    Returns:
        wout_full  : [ArrayLike] The resulting outputs with shape
                    (nframes, )
    """
    data = self.data_table["value"].data
    full_shape = data.shape
    flat_full_shape = (full_shape[0], full_shape[1]*full_shape[2])
    flat_out = rearrange(data, "frame wavelength output -> frame (wavelength output)")
    wout = self.md.einsum("f o i , f i -> f o", self.Ws, flat_out)
    wout_full = wout.reshape((full_shape))
    return wout_full

# def whitening_transform(self, input):
#     """
#         Currently unused
#     """
    
#     full_shape = data.shape
#     flat_full_shape = (full_shape[0], full_shape[1]*full_shape[2])
#     flat_out = rearrange(data, "frame wavelength output -> frame (wavelength output)")
#     output = self.md.einsum("f o i , f i -> f o", self.Ws, flat_out)
#     return output

def residual_pdet_Te(lamb, xsi, rank, targ):
    """
    Computes the residual of Pdet
    
    **Arguments:**
    
    * lamb     : The noncentrality parameter representing the feature
    * targ     : The target Pdet to converge to
    * xsi      : The location of threshold
    * rank     : The rank of observable
    
    **Returns** the Pdet difference. See *Ceau et al. 2019* for more information
    """
    respdet = 1 - ncx2.cdf(xsi,rank,lamb) - targ
    return respdet

def residual_pdet_Tnp(xTx, xsi, targ):
    """
    Computes the residual of Pdet in a NP test.
    
    Arguments:
        xTx     : The noncentrality parameter representing the feature
        targ     : The target Pdet to converge to
        xsi      : The location of threshold
    
    Returns:
        The Pdet difference.
    
    """
    
    Pdet_Pfa = 1 - norm.cdf((xsi - xTx)/np.sqrt(xTx))
    return Pdet_Pfa - targ


        
