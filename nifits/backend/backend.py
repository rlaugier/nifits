

import nifits.io as io
from nifits.io.oifits import NIFITS_EXTENSIONS, STATIC_EXTENSIONS
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as units


class NI_Backend(object):
    # def __init__(self, myfits: type(io.oifits.NIFITS)):
    def __init__(self, nifits: io.oifits.nifits = None,
                    module=np):
        self.nifits = nifits

    def add_instrument_definition(self, nifits_instrument: io.oifits.nifits = None,
                                    force: bool = False,
                                    verbose: bool = True):
        if nifits_instrument is not None:
            if self.nifits is None:
                self.nifits = nifits_instrument
            else:
                for anext in NIFITS_EXTENSIONS[STATIC_EXTENSIONS]:
                    if hasattr(nifits_instrument, anext.lower()):
                        if (not hasattr(self.nifits, anext.lower())) \
                            or force:
                            self.__setattr__(anext.lower(),
                                nifits_instrument.__getattribute__(anext.lower()))
                        else:
                            print(f"Local nifits, already has {anext.lower()}")
                    else:
                        print(f"Could not find {anext.lower()}")
    
    def add_observation_data(self, nifits_data: io.oifits.nifits = None,
                                    force: bool = False,
                                    verbose: bool = True):
        if nifits_data is not None:
            if self.nifits is None:
                self.nifits = nifits_data
            else:
                for anext in NIFITS_EXTENSIONS[np.logical_not(STATIC_EXTENSIONS)]:
                    if hasattr(nifits_data, anext.lower()):
                        if (not hasattr(self.nifits, anext.lower())) \
                            or force:
                            self.__setattr__(anext.lower(),
                                nifits_data.__getattribute__(anext.lower()))
                        else:
                            print(f"Local nifits, already has {anext.lower()}")
                    else:
                        print(f"Could not find {anext.lower()}")
    def propagate_array(self, xys, vals):
        pass
    def propagate_source(self, xy, val):
        pass

    def create_fov_function_all(self, md=np):
        """
        Constructs the method to get the chromatic phasor
        given by injection for all the time series.

        *Arguments:*

        Sets up `self.ni_fov.xy2phasor`
        """
        assert self.nifits.ni_fov.header["FOV_MODE"] == "diameter_gaussian_radial"
        D = (self.nifits.ni_fov.header["FOV_TELDIAM"] \
                *units.Unit(self.nifits.ni_fov.header["FOV_TELDIAM_UNIT"]))\
                    .to(units.m).value
        r_0 = (1/2*self.nifits.oi_wavelength.lambs/D)*units.rad.to(units.mas)
        offset = md.array(self.nifits.ni_fov.data_table["offsets"])
        def xy2phasor(x,y):
            r = md.hypot(x[None, None,:]-offset[:,:,0,None], y[None,None,:]-offset[:,:,1,None])
            phasor = md.exp(-(r[:,:]/r_0[:,None])**2)
            return phasor.astype(md.complex)
        self.nifits.ni_fov.xy2phasor = xy2phasor

    def get_modulation_phasor(self, md=np):
        """
        Computes and returns the modulation phasor [n_wl, n_input]
        """
        mods = md.array([a  for a in self.ni_mod.phasors]).T
        return mods
    def geometric_phasor(self, alpha, beta, include_mod=True,
                            md=np):
        """
        Returns the complex phasor corresponding to the locations
        of the family of sources
        
        **Parameters:**
        
        * alpha         : The coordinate matched to X in the array geometry
        * beta          : The coordinate matched to Y in the array geometry
        * anarray       : The array geometry (n_input, 2)
        * include_mod   : Include the modulation phasor
        
        **Returns** : A vector of complex phasors
        """
        xy_array = md.vstack(self.NI_MOD_table["APPXY"])
        lambs = md.array(self.nifits.ni_wavelength.lambs)
        k = 2*md.pi/lambs
        a = md.array((alpha, beta), dtype=md.float64)
        phi = k[:,None,None] * xy_array.dot(a)[None,:]
        b = md.exp(1j*phi)
        if include_mod:
            mods = self.get_modulation_phasor(md=md)
            b *= mods[:,:,None]
        return b
        
    def get_Is(self, xs, md=np):
        E = md.einsum("w o i , w i m -> w o m", self.nifits.ni_catm.M, xs)
        I = md.abs(E)**2
        return I

    def get_KIs(self, I, md=np):
        KI = md.einsum("w i m, o i -> w o m", I, self.nifits.ni_kmat.K[:,:])
        return KI
        
        
    def dot_all_fov(self,  xs):
        I = self.get_Is(xs)
        KI = self.get_KIs(I)
        return KI
        
    def get_all_ks(self, alphas, betas):
        xs = self.geometric_phasor(alphas, betas)
        KIs = self.dot_all_fov(xs)
        return KIs
    
        
    

