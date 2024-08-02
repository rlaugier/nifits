

"""
Backend module
--------------

This module handles basic computation based on the NIFITS standard
for representing the instrument. It reconstructs the model-in-a-kit
following the standard.

It offers a limited capability to be powered with computing backends
that have a numpy-compatible API.
"""










import nifits.io as io
from nifits.io.oifits import NIFITS_EXTENSIONS, STATIC_EXTENSIONS
from nifits.io.oifits import nifits as NIFITSClass
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as units
from numpy.typing import ArrayLike
from types import ModuleType

from dataclasses import dataclass

mas2rad = units.mas.to(units.rad)
rad2mas = units.rad.to(units.mas)



# TODO Add methods to __add__ PointCollections for variable sampling
# TODO Add a shorthand to surface element ds, taken from the overal surface
# divided by the number of points. WARNING: we will have to handle redundancy.

@dataclass
class PointCollection(object):
    """
        A class to hold arrays of coordinates. Handy to compute
    the transmission map on a large number of points.

    **Units default to mas.**

    ** Arguments:**

    * ``aa``    : [unit (mas)] (ArrayLike) first coordinate flat array, 
      typically RA.
    * ``bb``    : [unit (mas)] (ArrayLike) second coordinate flat array, 
      typically Dec.

    **Constructors:**

    * ``from_uniform_disk`` : 
    * ``from_grid``         :
    * ``from_centered_square``   :

    **Handles:**

    * ``coords``         : The array values as first provided
    * ``coords_rad``     : The array values, converted from 
      ``self.unit`` into radians.
    * ``coords_quantity``: Return the values as a quantity.
    * ``coords_radial``  : Returns the radial coordinates (rho,theta)
    * ``extent``         : The [left, right, top, bottom] extent
      (used for some plots).
    """
    aa: ArrayLike = None
    bb: ArrayLike = None
    unit: units.Unit = None
    def __post_init__(self):
        self.shape = self.aa.shape
        if self.unit is None:
            self.unit = units.mas
        if not hasattr(self, "orig_shape"):
            self.orig_shape = self.shape

    @classmethod
    def from_uniform_disk(cls, radius=None,
                n: int = 10,
                phi_0: float = 0.,
                offset: ArrayLike = np.array((0.,0.)),
                md: ModuleType = np,
                unit: units.Unit = units.mas):
        """
            Create a point collection as a uniformly sampled disk.

        **Arguments:**

        * ``a_coords`` : The array of samples along the first axis
          (typically alpha)
        * ``b_coords`` : The array of samples along the second axis
          (typically beta, the second dimension)

        **Handles:**
        """
        alpha = md.pi * (3 - md.sqrt(5))    # the "golden angle"
        points = []
        for k in md.arange(n):
          theta = k * alpha + phi_0
          r = radius * md.sqrt(float(k)/n)
          points.append((r * md.cos(theta), r * md.sin(theta)))
        points = np.array(points).T + offset[:,None]
        myobj = cls(*points, unit=unit)
        return myobj

    @classmethod
    def from_grid(cls, a_coords: ArrayLike, b_coords: ArrayLike,
                        md: ModuleType = np,
                        unit: units.Unit = units.mas):
        """
            Create a point collection as a cartesian grid.

        **Arguments:**

        * ``a_coords`` : The array of samples along the first axis
          (typically alpha)
        * ``b_coords`` : The array of samples along the second axis
          (typically beta, the second dimension)

        **Handles:**
        """
        aa, bb = md.meshgrid(a_coords, b_coords)
        original_shape = aa.shape
        aa = aa.flatten()
        bb = bb.flatten()
        myobj = cls(aa=aa, bb=bb)
        myobj.extent = [md.min(a_coords), md.max(a_coords),
                        md.min(b_coords), md.max(b_coords)]
        myobj.orig_shape = original_shape
        return myobj

    @classmethod
    def from_centered_square_grid(cls, radius, resolution,
                            md: ModuleType = np):
        """
            Create a centered square cartesian grid object

        **Arguments:**

        * ``radius``      : The radial extent of the grid.
        * ``resolution``  : The number of pixels across the width.
        """
        a_coords = md.linspace(-radius, radius, resolution)
        b_coords = md.linspace(-radius, radius, resolution)
        myobj = cls.from_grid(a_coords=a_coords,
                            b_coords=b_coords,
                            md=md)
        return myobj
        

    @property
    def coords(self):
        """
        Returns a tuple with the ``alpha`` and ``beta`` coordinates in 
        flat arrays.
        """
        return (self.aa, self.bb)
    
    @property
    def coords_rad(self):
        return (mas2rad*self.aa, mas2rad*self.bb)

    @property
    def coords_radial(self):
        """
        Returns the radial coordinates of points. (rho, theta) ([unit], [rad]).
        """
        cpx = self.aa + 1j*self.bb
        return (np.abs(cpx), np.angle(cpx))
    @property
    def coords_shaped(self):
        if hasattr(self, "orig_shape"):
            return (self.aa.reshape(self.orig_shape), self.bb.reshape(self.orig_shape))
        else:
            raise AttributeError("Did not have an original shape")


class NI_Backend(object):
    """
    A class to produce calculations based on the NIFITS standard.

    """
    # def __init__(self, myfits: type(io.oifits.NIFITS)):
    def __init__(self, nifits: NIFITSClass = None,
                    module=np):
        self.nifits = nifits
        """
        Backend object. Typically created empty. The nifits
        instrument information are added later with:

        * ``.add_instrument_definition``
        * ``.add_observation_data``
        
        **Arguments:**

        * ``nifits``    : NIFITSClass 
        * ``module``    : A backend module for advanced math.
        """

    def add_instrument_definition(self, nifits_instrument: NIFITSClass = None,
                                    force: bool = False,
                                    verbose: bool = True):
        """
        Adds the instrument definition to the model.
        
        **Arguments:**

        * ``nifits_instrument``   : NIFITS object
        * ``force``               : ``Bool`` if True, then the existing extensions
          will be replaced
        * ``verbose``             : Get more printed information
        """
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
    
    def add_observation_data(self, nifits_data: NIFITSClass = None,
                                    force: bool = False,
                                    verbose: bool = True):
        """
        Adds the observation data to the model.
        
        **Arguments:**

        * ``nifits_instrument``   : NIFITS object
        * ``force``               : ``Bool`` if True, then the existing extensions
          will be replaced
        * ``verbose``             : Get more printed information
        """
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

    def create_fov_function_all(self, md=np):
        """
        Constructs the method to get the chromatic phasor
        given by injection for all the time series.

        **Arguments:**

        Sets up ``self.ni_fov.xy2phasor``
        """
        assert self.nifits.ni_fov.header["FOV_MODE"] == "diameter_gaussian_radial"
        D = (self.nifits.ni_fov.header["FOV_TELDIAM"] \
                *units.Unit(self.nifits.ni_fov.header["FOV_TELDIAM_UNIT"]))\
                    .to(units.m).value
        r_0 = (1/2*self.nifits.oi_wavelength.lambs/D)# *units.rad.to(units.mas)
        offset = md.array(self.nifits.ni_fov.data_table["offsets"])
        def xy2phasor(x,y):
            """
            x and y in rad.

            **Arguments:**

            * x     : ArrayLike [rad] Coordinate in the Fov.
            * y     : ArrayLike [rad] Coordinate in the Fov.
            """
            r = md.hypot(x[None, None,:]-offset[:,:,0,None], y[None,None,:]-offset[:,:,1,None])
            phasor = md.exp(-(r[:,:]/r_0[:,None])**2)
            return phasor.astype(complex)
        self.nifits.ni_fov.xy2phasor = xy2phasor

    def get_modulation_phasor(self, md=np):
        """
        Computes and returns the modulation phasor [n_wl, n_input]

        The modulation phasor includes is computed in units of collected amplitude
        so that the output of the transmission map in square modulus provides equivalent
        collecting power in m^2 for each point of the map. This is done to facilitate the
        usag of the map with models in jansky.
        """
        # mods = md.array([a  for a in self.nifits.ni_mod.phasors]).T
        mods = md.array(self.nifits.ni_mod.all_phasors)
        col_area = md.array(self.nifits.ni_mod.arrcol)
        return mods*md.sqrt(col_area)[:,None,:]

    def geometric_phasor(self, alpha, beta, include_mod=True,
                            md=np):
        """
        Returns the complex phasor corresponding to the locations
        of the family of sources
        
        **Parameters:**
        
        * ``alpha``         : The coordinate matched to X in the array geometry
        * ``beta``          : The coordinate matched to Y in the array geometry
        * ``anarray``       : The array geometry (n_input, 2)
        * ``include_mod``   : Include the modulation phasor
        
        **Returns** : A vector of complex phasors
        """
        xy_array = self.nifits.ni_mod.appxy
        lambs = md.array(self.nifits.oi_wavelength.lambs)
        k = 2*md.pi/lambs
        a = md.array((alpha, beta), dtype=md.float64)
        # print(xy_array.shape)
        # print(a.shape)
        # phi = k[:,None,None,None] * md.array([anxy_array[:,:].dot(a[:,:]) for anxy_array in xy_array])
        phi = k[:,None,None,None] * md.einsum("t a x, x m -> t a m", xy_array[:,:,:], a[:,:])
        # print(a.shape)
        b = md.exp(1j*phi)
        if include_mod:
            mods = self.get_modulation_phasor(md=md)
            b *= mods[:,:,None]
        # print(b.shape)
        return b.transpose((1,0,2,3))
        
    def get_Is(self, xs, md=np):
        """
        Get intensity from an array of sources.
        """
        E = md.einsum("w o i , t w i m -> t w o m", self.nifits.ni_catm.M, xs)
        I = md.abs(E)**2
        return I

    def get_KIs(self, Iarray:ArrayLike, md:ModuleType=np):
        r"""
        Get the prost-processed observable from an array of output intensities. The
        post-processing matrix K is taken from ``self.nifits.ni_kmat.K``

        **Arguments:**

        * ``I``     : (n_frames, n_wl, n_outputs, n_batch)
        * ``md``    : a python module with a numpy-like interface.

        **Returns:**
        The vector :math:`\boldsymbol{\kappa} = \mathbf{K}\cdot\mathbf{I}`
        """
        KI = md.einsum("k o, t w o m -> t w k m", self.nifits.ni_kmat.K[:,:], Iarray)
        return KI
        
    def get_all_outs(self, alphas, betas, kernels=False):
        """
        Compute the transmission map for an array of coordinates. The map can be seen
        as equivalent collecting power expressed in [m^2] for each point sampled so as
        to facilitate comparison with models in Jansky multiplied by the exposure time
        of each frame (available in `nifits.ni_mod.int_time`).

        **Argrguments:**

        * ``alphas``  : ArrayLike [rad] 1D array of coordinates in right ascension
        * ``betas``   : ArrayLike [rad] 1D array of coordinates in declination
        * ``kernels`` : (bool) if True, then computes the post-processed
          observables as per the KMAT matrix.

        **Returns:**

        * if ``kernels`` is False: the *raw transmission output*.
        * if ``kernels`` is True: the *differential observable*.

        **Shape:** (n_frames, n_wl, n_outputs, n_points)
        """
        # The phasor from the incidence on the array:
        xs = self.geometric_phasor(alphas, betas, include_mod=False)
        # print("xs", xs)
        
        # The phasor from the spatial filtering:
        x_inj = self.nifits.ni_fov.xy2phasor(alphas, betas)
        # print("x_inj", x_inj)
        
        # The phasor from the internal modulation
        # x_mod = self.nifits.ni_mod.all_phasors
        x_mod = self.get_modulation_phasor()
        # print("x_mod", x_mod)
        
        Is = self.get_Is(xs * x_inj[:,:,None,:] * x_mod[:,:,:,None])
        if kernels:
            KIs = self.get_KIs(Is)
            return KIs
        else:
            return Is
    
        
    

