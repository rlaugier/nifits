

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
import numpy.typing
import matplotlib.pyplot as plt
import astropy.units as units
import types
ModuleType = types.ModuleType

# from numpy.typing import ArrayLike
# A hack to fix the documentation of type hinting
ArrayLike = np.typing.ArrayLike


from dataclasses import dataclass

from einops import rearrange

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

    Args:
        aa    : [unit (mas)] (ArrayLike) first coordinate flat array, 
              typically RA.
        bb    : [unit (mas)] (ArrayLike) second coordinate flat array, 
              typically Dec.

    Constructors:
        * ``from_uniform_disk``   : 
        * ``from_grid``           :
        * ``from_centered_square``:
        * ``from_segment``        :

    Modificators:
        * ``__add__``         : basically a concatenation
        * ``transform``       : Linear transformation in 3D by a matrix

    Handles:
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
    def from_uniform_disk(cls, radius: float = None,
                        n: int = 10,
                        phi_0: float = 0.,
                        offset: ArrayLike = np.array((0.,0.)),
                        md: ModuleType = np,
                        unit: units.Unit = units.mas):
        """
            Create a point collection as a uniformly sampled disk.

        Args:
            radius :   [mas] The angular radius of the disk to model
            n      :   The total number of points to create
            phi_0  :   [rad] Arbitrary angle to initialiaze pattern
            offset :   [mas] An offset of the disk location with respect
                to the center of the field of view
            md     :   The numerical module to use as backend
            unit   :   The unit to use for interactions

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

        Args:
            a_coords : The array of samples along the first axis
                  (typically alpha)
            b_coords : The array of samples along the second axis
                  (typically beta, the second dimension)
            md : Module of choice for the computation
            unit : Units for ``a_coords`` and ``b_coords``

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
    def from_segment(cls, start_coords: ArrayLike,
                        end_coords: ArrayLike,
                        n_samples: int,
                        md: ModuleType = np,
                        unit: units.Unit = units.mas):
        """
            Create a point collection as a cartesian grid.

        Args:
            start_coords : The (a,b) array of the starting point.
                  (typically alpha, beta)
            end_coords : The (a,b) array of the ending point.
                  (typically alpha, beta)
            n_sameples     : The number of samples along the line.

        **Handles:**
        """
        aa = md.linspace(start_coords[0], end_coords[0], n_samples)
        bb = md.linspace(start_coords[1], end_coords[1], n_samples)
        original_shape = aa.shape
        aa = aa.flatten()
        bb = bb.flatten()
        myobj = cls(aa=aa, bb=bb)
        return myobj

    @classmethod
    def from_centered_square_grid(cls,
                        radius,
                        resolution,
                        md: ModuleType = np):
        """
            Create a centered square cartesian grid object

        Args:
            radius      : The radial extent of the grid.
            resolution  : The number of pixels across the width.
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
    def plot_frame(self, z=None, frame_index=0, wl_index=0,
                        out_index=0, mycmap=None, marksize_increase=1.0,
                        colorbar=True, xlabel=True, title=True):
        import matplotlib.pyplot as plt
        marksize = marksize_increase * 50000/self.shape[0]
        if len(self.orig_shape) == 1:
            plt.scatter(*self.coords, c=z[frame_index,wl_index,out_index,:],
                    cmap=mycmap, s=marksize)
            plt.gca().set_aspect("equal")
        else:
            plt.imshow(z[frame_index,wl_index,out_index,:].reshape((self.orig_shape)),
                cmap=mycmap, extent=self.extent)
            plt.gca().set_aspect("equal")
            
        if colorbar:
            plt.colorbar()
        if xlabel is True:
            plt.xlabel("Relative position [mas]")
        elif xlabel is not False:
            plt.xlabel(xlabel)
        if title is True:
            plt.title(f"Output {out_index} for frame {frame_index}")
        elif title is not False:
            plt.title(title)

            

    def __add__(self, other):
        """
        Add together two collection of points.

        The result inerits properties of the first argument.
        """
        from copy import copy
        if hasattr(self, "md"):
            md = self.md
        elif hasattr(other, "md"):
            md = other.md
        else:
            md = np
        new = copy(self)
        new.aa = md.concatenate((new.aa, other.aa*other.unit.to(self.unit)))
        new.bb = md.concatenate((new.bb, other.bb*other.unit.to(self.unit)))
        if hasattr(self, "cc"):
            if not hasattr(other, "cc"):
                other.cc = md.zeros_like(other.aa)
            new.cc = md.concatenate((new.cc, other.cc*other.unit.to(self.unit)))

        if hasattr(new, "extent"):
            del new.extent
        if hasattr(new, "ds"):
            del new.ds
        if hasattr(new, "orig_shape"):
            del new.orig_shape
        return new
    def transform(self, matrix, md=np):
        """
        Produce a linear transform of the coordinates.

        Args:
            matrix: A transformation matrix (3D)
            md    : A module to do the operation
        """
        if not hasattr(self, "cc"):
            self.cc = md.zeros_like(self.aa)
        vectors = md.vstack((self.aa, self.bb, self.cc))
        transformed = md.dot(matrix, vectors)
        self.aa = transformed[0,:]
        self.bb = transformed[1,:]
        self.cc = transformed[2,:]
        

@dataclass
class MovingCollection(object):
    series: PointCollection

    @property
    def coords_rad(self):
        arraypoints = np.array([thecollec.coords_rad for thecollec in self.series])
        arranged = rearrange(arraypoints, "time coord points -> coord time points")
        return arranged
        
    @property
    def coords(self):
        arraypoints = np.array([thecollec.coords for thecollec in self.series])
        arranged = rearrange(arraypoints, "time coord points -> coord time points")
        return arranged
        
    @property
    def coords_radial(self):
        """
        Returns the radial coordinates of points. (rho, theta) ([unit], [rad]).

        """
        cpx = self.aa + 1j*self.bb
        return (np.abs(cpx), np.angle(cpx))
        
    @property
    def coords_shaped(self):
        if hasattr(self.series[0], "orig_shape"):
            reshaped = self.coords[:,:,self.series[0].orig_shape]
            return reshaped
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
        
        Args:
            nifits    : NIFITSClass 
            module    : A backend module for advanced math.

        """

    def add_instrument_definition(self, nifits_instrument: NIFITSClass = None,
                                    force: bool = False,
                                    verbose: bool = True):
        """
        Adds the instrument definition to the model.
        
        Args:
            nifits_instrument   : NIFITS object
            force               : ``Bool`` if True, then the existing extensions
                  will be replaced
            verbose             : Get more printed information

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
        
        Args:
            nifits_instrument   : NIFITS object
            force               : ``Bool`` if True, then the existing extensions
                  will be replaced
            verbose             : Get more printed information

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

        Args:
            md : A module for the computations

        Sets up ``self.ni_fov.xy2phasor``

        """
        assert self.nifits.ni_fov.header["FOV_MODE"] == "diameter_gaussian_radial"
        D = (self.nifits.ni_fov.header["FOV_TELDIAM"] \
                *units.Unit(self.nifits.ni_fov.header["FOV_TELDIAM_UNIT"]))\
                    .to(units.m).value
        r_0 = (1/2*self.nifits.oi_wavelength.lambs/D)# *units.rad.to(units.mas)
        offset = md.array(self.nifits.ni_fov.data_table["offsets"])
        def xy2phasor(x,y, md=md):
            """
            x and y in rad.

            Args:
                x     : ArrayLike [rad] Coordinate in the Fov.
                y     : ArrayLike [rad] Coordinate in the Fov.

            """
            r = md.hypot(x[None, None,:]-offset[:,:,0,None], y[None,None,:]-offset[:,:,1,None])
            phasor = md.exp(-(r[:,:]/r_0[:,None])**2)
            return phasor.astype(complex)
        self.nifits.ni_fov.xy2phasor = xy2phasor

        def xy2phasor_moving(x,y):
            """
            Employed to deal with point-samples that move in the FoV
            during the series of frames.
            
            x and y in rad.

            Args:
                x     : ArrayLike [rad] (time, point) Coordinate in the Fov.
                y     : ArrayLike [rad] (time, point) Coordinate in the Fov.

            """
            r = md.hypot(x[:, None,:]-offset[:,:,0,None], y[:,None,:]-offset[:,:,1,None])
            phasor = md.exp(-(r[:,:]/r_0[:,None])**2)
            return phasor.astype(complex)
        self.nifits.ni_fov.xy2phasor_moving = xy2phasor_moving

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
        
        Args:
            alpha         : The coordinate matched to X in the array geometry
            beta          : The coordinate matched to Y in the array geometry
            anarray       : The array geometry (n_input, 2)
            include_mod   : Include the modulation phasor
        
        Returns:
            A vector of complex phasors
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

    def get_KIs(self,
                    Iarray:ArrayLike,
                    md: ModuleType = np):
        r"""
        Get the prost-processed observable from an array of output intensities. The
        post-processing matrix K is taken from ``self.nifits.ni_kmat.K``

        Args:
            I     : (n_frames, n_wl, n_outputs, n_batch)
            md    : a python module with a numpy-like interface.

        Returns:
            The vector :math:`\boldsymbol{\kappa} = \mathbf{K}\cdot\mathbf{I}`

        """
        KI = md.einsum("k o, t w o m -> t w k m", self.nifits.ni_kmat.K[:,:], Iarray)
        return KI
        
    def get_all_outs(self, alphas, betas,
                        kernels=False,
                        md=np):
        """
        Compute the transmission map for an array of coordinates. The map can be seen
        as equivalent collecting power expressed in [m^2] for each point sampled so as
        to facilitate comparison with models in Jansky multiplied by the exposure time
        of each frame (available in `nifits.ni_mod.int_time`).

        Args:
            alphas  : ArrayLike [rad] 1D array of coordinates in right ascension
            betas   : ArrayLike [rad] 1D array of coordinates in declination
            kernels : (bool) if True, then computes the post-processed
                  observables as per the KMAT matrix.

        Returns:
            if ``kernels`` is False: the *raw transmission output*.
            if ``kernels`` is True: the *differential observable*.

        .. hint:: **Shape:** (n_frames, n_wl, n_outputs, n_points)

        """
        # The phasor from the incidence on the array:
        xs = self.geometric_phasor(alphas, betas, include_mod=False, md=md)
        # print("xs", xs)
        
        # The phasor from the spatial filtering:
        x_inj = self.nifits.ni_fov.xy2phasor(alphas, betas, md=md)
        # print("x_inj", x_inj)
        
        # The phasor from the internal modulation
        # x_mod = self.nifits.ni_mod.all_phasors
        x_mod = self.get_modulation_phasor()
        # print("x_mod", x_mod)
        
        Is = self.get_Is(xs * x_inj[:,:,None,:] * x_mod[:,:,:,None], md=md)
        if kernels:
            KIs = self.get_KIs(Is, md=md)
            return KIs
        else:
            return Is

    def moving_geometric_phasor(self, alphas, betas, include_mod=True,
                            md=np):
        """
        Returns the complex phasor corresponding to the locations
        of the family of sources
        
        **Parameters:**
        
        * ``alpha``         : (n_frames, n_points) The coordinate matched to X in the array geometry
        * ``beta``          : (n_frames, n_points) The coordinate matched to Y in the array geometry
        * ``anarray``       : The array geometry (n_input, 2)
        * ``include_mod``   : Include the modulation phasor
        
        **Returns** : A vector of complex phasors

        """
        xy_array = md.array(self.nifits.ni_mod.appxy)
        lambs = md.array(self.nifits.oi_wavelength.lambs)
        k = 2*md.pi/lambs
        a = md.array((alphas, betas), dtype=md.float64)
        phi = k[:,None,None,None] * md.einsum("t a x, x t m -> t a m", xy_array[:,:,:], a[:,:,:])
        b = md.exp(1j*phi)
        if include_mod:
            mods = self.get_modulation_phasor(md=md)
            b *= mods[:,:,None]
        return b.transpose((1,0,2,3))

    def get_moving_outs(self, alphas, betas,
                        kernels=False,
                        md=np):
        """
        Compute the transmission map for an array of coordinates. The map can be seen
        as equivalent collecting power expressed in [m^2] for each point sampled so as
        to facilitate comparison with models in Jansky multiplied by the exposure time
        of each frame (available in `nifits.ni_mod.int_time`).

        Args:
            alphas  : ArrayLike [rad] 1D array of coordinates in right ascension
            betas   : ArrayLike [rad] 1D array of coordinates in declination
            kernels : (bool) if True, then computes the post-processed
                      observables as per the KMAT matrix.

        Returns:
            if ``kernels`` is False: the *raw transmission output*.
            if ``kernels`` is True: the *differential observable*.

        .. hint:: **Shape:** (n_frames, n_wl, n_outputs, n_points)

        """
        # The phasor from the incidence on the array:
        xs = self.moving_geometric_phasor(alphas, betas, include_mod=False)
        # print("xs", xs)
        
        # The phasor from the spatial filtering:
        x_inj = self.nifits.ni_fov.xy2phasor_moving(alphas, betas)
        # print("x_inj", x_inj)
        
        # The phasor from the internal modulation
        # x_mod = self.nifits.ni_mod.all_phasors
        x_mod = self.get_modulation_phasor()
        # print("x_mod", x_mod)
        
        Is = self.get_Is(xs * x_inj[:,:,None,:] * x_mod[:,:,:,None], md=md)
        if kernels:
            KIs = self.get_KIs(Is, md=md)
            return KIs
        else:
            return Is
    

