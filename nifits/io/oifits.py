"""
A module for reading/writing NIFITS files and handling the data.

To open an existing NIFITS file, use ``nifits.from_nifits`` constructor.

To save an NIFITS object to a file, use ``nifits.to_nifits`` method.

A summary of the information in the oifits object can be obtained by
using the info() method:

   > import oifits
   > oifitsobj = oifits.open('foo.fits')
   > oifitsobj.info()

For further information, contact R. Laugier

"""

import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.table
Table = astropy.table.Table
from astropy.coordinates import EarthLocation
import datetime
import warnings
from packaging import version

import sys
from dataclasses import dataclass, field
# from numpy.typing import ArrayLike
# A hack to fix the documentation of type hinting
import numpy.typing
ArrayLike = np.typing.ArrayLike


__version__ = "0.0.9"
__standard_version__ = "0.7"

_mjdzero = datetime.datetime(1858, 11, 17)

matchtargetbyname = False
matchstationbyname = False
refdate = datetime.datetime(2000, 1, 1)


import warnings
def check_item(func):
    """
    A decorator for the `fits.Header.__getitem__`.
    This is here to save from compatibility issues with files of
    standard version <= 0.2 while warning that that this version will 
    """
    def inner(*args, **kwargs):
        good_kw = True
        try :
            item = func(*args, **kwargs)
            good_kw = True
        except KeyError: 
            good_kw = False
        if good_kw:
            return item
        bad_kw = True
        try : 
            akw = args[1]
            mykw = akw.split(" ")[-1]
            baditem = func(args[0], mykw, **kwargs)
            bad_kw = True
        except KeyError: 
            bad_kw = False
        
        if bad_kw and not good_kw:
            warnings.warn(f"Keyword deprecation. Expected `{args[1]}` (`HIERARCH` keyword)\n Found `{mykw}`\n This file was generated for NIFITS standard version <= 0.2. It will stop working for library versions >= 0.1.0 .")
            item = baditem
            return item
        elif not bad_kw and not good_kw:
            raise KeyError(f"Neither {args[1]} nor {mykw} found.")
            return None
            
            
        return item
    return inner
fits.Header.__getitem__ = check_item(fits.Header.__getitem__)

def _plurals(count):
    if count != 1: return 's'
    return ''

def array_eq(a: ArrayLike,
              b: ArrayLike):
    """
    Test whether all the elements of two arrays are equal.

    Args:
        a: one input.
        b: another input.
    """

    if len(a) != len(b):
        return False
    try:
        return not (a != b).any()
    except:
        return not (a != b)



def _isnone(x):
    """Convenience hack for checking if x is none; needed because numpy
    arrays will, at some point, return arrays for x == None."""

    return type(x) == type(None)

def _notnone(x):
    """Convenience hack for checking if x is not none; needed because numpy
    arrays will, at some point, return arrays for x != None."""

    return type(x) != type(None)




class OI_STATION(object):
    """ This class corresponds to a single row (i.e. single
    station/telescope) of an OI_ARRAY table."""

    def __init__(self, tel_name=None, sta_name=None, diameter=None, staxyz=[None, None, None], fov=None, fovtype=None, revision=1):

        if revision > 2:
            warnings.warn('OI_ARRAY revision %d not implemented yet'%revision, UserWarning)

        self.revision = revision
        self.tel_name = tel_name
        self.sta_name = sta_name
        self.diameter = diameter
        self.staxyz = staxyz

        if revision >= 2:
            self.fov = fov
            self.fovtype = fovtype
        else:
            self.fov = self.fovtype = None

    def __eq__(self, other):

        if type(self) != type(other): return False

        return not (
            (self.revision != other.revision) or
            (self.tel_name != other.tel_name) or
            (self.sta_name != other.sta_name) or
            (self.diameter != other.diameter) or
            (not _array_eq(self.staxyz, other.staxyz)) or
            (self.fov != other.fov) or
            (self.fovtype != other.fovtype))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):

        if self.revision >= 2:
            return '%s/%s (%g m, fov %g arcsec (%s))'%(self.sta_name, self.tel_name, self.diameter, self.fov, self.fovtype)
        else:
            return '%s/%s (%g m)'%(self.sta_name, self.tel_name, self.diameter)


        


@dataclass
class NI_CATM(object):
    """Contains the complex amplitude transfer matrix CATM of the instrument.
    The CATM is a complex matrix representing the transformation from the each
    of the complex amplitude of electric field from the inputs to the outputs
    of the instrument. The dimensions are (n_ch, n_out, n_in) where n_ch
    represents the spectral channels.
    
    It is expected that
    
    :math:`\\textbf{n}_{out} = \\textbf{M}_{CATM}.\\textbf{m}_{mod} \circ \\textbf{x}_{in}`
    with :math:`\\textbf{m}_{mod}` containded in NI_MOD.
    
    """
    Mcatm: ArrayLike
    header: fits.Header

    @classmethod
    def from_hdu(cls, hdu: type(fits.hdu.ImageHDU)):
        """
        Create the NI_CATM object from the HDU extension of an opened fits file.
        """
        Mcatm = hdu.data
        header = hdu.header
        myobj = cls(Mcatm, header)
        return myobj

    def to_hdu(self):
        """
        Returns and hdu object to save into fits
        """
        myhdu = fits.hdu.ImageHDU(data=self.Mcatm)
        myhdu.header = self.header
        return myhdu
    # TODO add a check method


def nulfunc(self, *args, **kwargs):
    raise TypeError


NI_OITAG_DEFAULT_HEADER = fits.Header(cards=[("HIERARCH NIFITS IOSWAPS", False, "The units for output values")])

NI_MOD_DEFAULT_HEADER = fits.Header(cards=[("HIERARCH NIFITS AMOD_PHAS_UNITS", "rad", "The units for modulation phasors"),
                                        ("HIERARCH NIFITS ARRCOL_UNITS", "m^2", "The units for collecting area")
                                            ])

# Possible to use "chromatic_gaussian_radial", "diameter_gaussian_radial".
# Simplest default is a gaussian with r0 = lambda/D
NI_FOV_DEFAULT_HEADER = fits.Header(cards=[("HIERARCH NIFITS FOV_MODE","diameter_gaussian_radial","Type of FOV definition"),
                                        ("HIERARCH NIFITS FOV_offset"),
                                        ("HIERARCH NIFITS FOV_TELDIAM", 8.0, "diameter of a collecting aperture for FOV"),
                                        ("HIERARCH NIFITS FOV_TELDIAM_UNIT", "m", ""),])

"""
Obtaining the goecentric location of observatories. (unchecked)
```python
from astroplan import Observer
import astropy.coordinates as coords
myobs = Observer.at_site("CHARA")
print([acoord.value for acoord in myobs.location.to_geocentric()]) 
```
VLTI:     1946404.3410388362, -5467644.290798524, -2642728.2014442487
CHARA:    -2484228.6029109913, -4660044.467216573, 3567867.961141405
"""
OI_ARRAY_DEFAULT_VLTI_HEADER = fits.Header(cards=[
    ("OI_REVN", 1, "Revision number of the table definition (refers no OIFITS version, not NIFITS)."),
    ("ARRNAME", "VLTI", "Array name, for cross-referencing"),
    ("FRAME", "GEOCENTRIC", "Coordinate frame"),
    ("ARRAYX", 1946404.3410388362, "Array center coordinates (m)"),
    ("ARRAYY", -5467644.290798524, "Array center coordinates (m)"),
    ("ARRAYZ", -2642728.2014442487, "Array center coordinates (m)"),
])
    
OI_ARRAY_DEFAULT_CHARA_HEADER = fits.Header(cards=[
    ("OI_REVN", 1, "Revision number of the table definition (refers no OIFITS version, not NIFITS)."),
    ("ARRNAME", "CHARA", "Array name, for cross-referencing"),
    ("FRAME", "GEOCENTRIC", "Coordinate frame"),
    ("ARRAYX", -2484228.6029109913, "Array center coordinates (m)"),
    ("ARRAYY", -4660044.467216573, "Array center coordinates (m)"),
    ("ARRAYZ", 3567867.961141405, "Array center coordinates (m)"),
])

    

@dataclass
class NI_EXTENSION(object):
    """
    ``NI_EXTENSION`` Generic class for NIFITS extensions

    **Inherited methods:**

    * ``from_hdu``: Creates the object from an ``astropy.io.fits.TableHDU`` object
    * ``to_hdu``  : Returns the ``TableHDU`` from itself.

    Args:
        data_table: [ArrayLike] The data to stored 
        header: [fits.Header] A fits header (optional)
        unit: [astropy.units.Unit] Units of the data stored
            (mandatory of NI_IOUT, NI_KIOUT and NI_KCOV)

    """
    data_table: Table = field(default_factory=Table)
    header: fits.Header = field(default_factory=fits.Header)
    unit: u.Unit = None

    # TODO: Potentially, this should be a None by default, while still being a 
    # fits.Header type hint... We can if we have a None, we can catch it with 
    # a __post_init__ method. TODO this will help cleanup the signature in the doc.

    @classmethod
    def from_hdu(cls, hdu: type(fits.hdu.TableHDU)):
        """
        Create the data object from the HDU extension of an opened fits file.
        
        Args:
            hdu : TableHDU object containing the relevant data
        """
        data_table = Table(hdu.data)
        header = hdu.header
        if "NIFITS IUNIT" in header.keys():
            return cls(data_table=data_table, header=header, unit=u.Unit(header["NIFITS IUNIT"]))
        elif "IUNIT" in header.keys(): # Backwards compatibility
            return cls(data_table=data_table, header=header, unit=u.Unit(header["IUNIT"]))
        else:
            return cls(data_table=data_table, header=header)

    def to_hdu(self):
        """
        Returns and hdu object to save into fits
        
        .. admonition::
        
            This also updates the header if dimension changes
        """
        if hasattr(self, "unit"):
            if self.unit is not None:
                self.header["NIFITS IUNIT"] = (self.unit.to_string(), "Unit for the content")
        # TODO this looks like a bug in astropy.fits: the header should update on its own.
        myhdu = fits.hdu.BinTableHDU(name=self.name, data=self.data_table, header=self.header)
        # myhdu = fits.hdu.BinTableHDU(name=self.name, data=self.data_table)
        
        
        # TODO: fix the diffing?
        # print("Updating header:\n", fits.HeaderDiff(myhdu.header, self.header).__repr__)
        self.header = myhdu.header
        return myhdu

    def __len__(self):
        return len(self.data_table)

    def __info__(self):
        """
        Generic method to return information.
        """
        myinfostring = f"""
## Generic NIFITS table extension


+ Length : {self.__len__()}
+ Unit : [{self.unit}]
+ Columns :

"""
        for acolname in self.data_table.colnames:
            myinfostring += f"   - **{acolname}** [{self.data_table[acolname].unit}] , shape : {self.data_table[acolname].unit}, dtype : {self.data_table[acolname].dtype} \n"
        return myinfostring

@dataclass
class NI_EXTENSION_ARRAY(NI_EXTENSION):
    """
    Generic class for NIFITS array extensions

    Args:
        data_array: [ArrayLike] The data to stored 
        header: [fits.Header] A fits header (optional)
        unit: [astropy.units.Unit] Units of the data stored
            (mandatory of NI_IOUT, NI_KIOUT and NI_KCOV)

    """
    data_array: ArrayLike = field(default_factory=np.array)
    header: fits.Header = field(default_factory=fits.Header)
    unit: u.Unit = None

    @classmethod
    def from_hdu(cls, hdu: type(fits.hdu.ImageHDU)):
        """
        Create the data object from the HDU extension of an opened fits file.
        """
        data_array = hdu.data
        header = hdu.header
        if "NIFITS IUNIT" in header.keys():
            return cls(data_array=data_array, header=header, unit=u.Unit(header["NIFITS IUNIT"]))
        elif "IUNIT" in header.keys(): # Backwards compatibility
            return cls(data_array=data_array, header=header, unit=u.Unit(header["IUNIT"]))
        else:
            return cls(data_array=data_array, header=header)
    
    def to_hdu(self,):
        """
        Returns and hdu object to save into fits
        
        .. admonition:: Note
        
            This also updates the header if dimension changes
        """
        if hasattr(self, "unit"):
            if self.unit is not None:
                self.header["NIFITS IUNIT"] = (self.unit.to_string(), "Unit for the content")
        myhdu = fits.hdu.ImageHDU(name=self.name,data=self.data_array, header=self.header)
        print("Updating header:\n", fits.HeaderDiff(myhdu.header, self.header))
        self.header = myhdu.header
        return myhdu

    def __len__(self):
        pass

    @property
    def shape(self):
        return self.data_array.shape


    def __info__(self):
        """
        Generic method to return information.
        """
        myinfostring = f"""
## Generic NIFITS array extension


+ Shape : {self.data_array.shape}
+ Data type : {self.data_array.dtype}
+ Unit : [{self.unit}]
"""
        return myinfostring

@dataclass
class NI_EXTENSION_CPX_ARRAY(NI_EXTENSION):
    """
    Generic class for NIFITS array extensions.

    The array is kept locally as complex valued, but it is
    stored to and loaded from a real-valued array with an
    extra first dimension of length 2 for (real, imaginary) parts.
    """
    data_array: ArrayLike = field(default_factory=np.array)
    header: fits.Header = field(default_factory=fits.Header)

    @classmethod
    def from_hdu(cls, hdu: type(fits.hdu.ImageHDU)):
        """
        Create the data object from the HDU extension of an opened fits file.
        """
        assert hdu.data.shape[0] == 2,\
                f"Data should have 2 layers for real and imag. {hdu.data.shape}"
        data_array = hdu.data[0] + 1j*hdu.data[1]
        header = hdu.header
        return cls(data_array=data_array, header=header)
    
    def to_hdu(self,):
        """
        Returns an hdu object to save into fits
        
        .. admonition:: Note
        
            This also updates the header if dimension changes
        """
        real_valued_data = np.array([self.data_array.real,
                                    self.data_array.imag], dtype=float)
        myhdu = fits.hdu.ImageHDU(name=self.name,data=real_valued_data, header=self.header)
        print("Updating header:\n", fits.HeaderDiff(myhdu.header, self.header))
        self.header = myhdu.header
        return myhdu

    def __len__(self):
        pass

    @property
    def shape(self):
        return self.data_array.shape

    def __info__(self):
        """
        Generic method to return information.
        """
        myinfostring = f"""
## Generic NIFITS array extension


+ Shape : {self.data_array.shape}
+ Data type : {self.data_array.dtype}
+ Unit : [{self.unit}]
"""
        return myinfostring


class OI_ARRAY(NI_EXTENSION):
    __doc__ = """
    ``OI_ARRAY`` definition

    Args:
        data_table: The data to hold
        header: The associated fits header

    
    """ + NI_EXTENSION.__doc__
    name="OI_ARRAY"


from astropy.constants import c as cst_c


class OI_WAVELENGTH(NI_EXTENSION):
    __doc__ = """
    An object storing the OI_WAVELENGTH information, in compatibility with
    OIFITS practices.

    **Shorthands:**

    * ``self.lambs`` : ``ArrayLike`` [m] returns an array containing the center
      of each spectral channel.
    * ``self.dlmabs`` : ``ArrayLike`` [m] an array containing the spectral bin
      widths.
    * ``self.nus`` : ``ArrayLike`` [Hz] an array containing central frequencies
      of the
      spectral channels.
    * ``self.dnus`` : ``ArrayLike`` [Hz] an array containing the frequency bin
      widths.

    """ + NI_EXTENSION.__doc__
    name = "OI_WAVELENGTH"

    @property
    def lambs(self):
        return self.data_table["EFF_WAVE"].data
    @property
    def dlambs(self):
        return self.data_table["EFF_BAND"].data
    @property
    def nus(self):
        return (cst_c/(self.lambs*u.m)).value
    @property
    def dnus(self):
        return np.abs(np.gradient(self.nus))
        

class NI_OSWAVELENGTH(NI_EXTENSION):
    __doc__ = """
    An object storing the wavelength before a downsampling. This must have the
    wavelength for each of the slice of the CATM matrix, each of the ``NI_MOD``
    phasors and each column of the ``NI_DSAMP`` matrix.

    If ``NI_OSWAVELENGTH`` is absent, assume that there is no over or down-
    sampling and take the values directly from ``OI_WAVELENGTH``.

    **Shorthands:**

    * ``self.lambs`` : ``ArrayLike`` [m] returns an array containing the center
      of each spectral channel.
    * ``self.dlmabs`` : ``ArrayLike`` [m] an array containing the spectral bin
      widths.
    * ``self.nus`` : ``ArrayLike`` [Hz] an array containing central frequencies
      of the
      spectral channels.
    * ``self.dnus`` : ``ArrayLike`` [Hz] an array containing the frequency bin
      widths.

    """ + NI_EXTENSION.__doc__
    name = "NI_OSWAVELENGTH"

    @property
    def lambs(self):
        return self.data_table["EFF_WAVE"].data
    @property
    def dlambs(self):
        return self.data_table["EFF_BAND"].data
    @property
    def nus(self):
        return (cst_c/(self.lambs*u.m)).value
    @property
    def dnus(self):
        return np.abs(np.gradient(self.nus))

from dataclasses import field
from typing import List


@dataclass
class OI_TARGET(NI_EXTENSION):
    """
    ``OI_TARGET`` definition.
    """
    # target: List[str] = field(default_factory=list)
    # raep0: float = 0.
    # decep0: float = 0.
    name="OI_TARGET"
    @classmethod
    def from_scratch(cls, ):
        """
        Creates the OI_TARGET object with an empty table.

        **Returns:        
            OI_TARGET: object with an empty table.

        Use ``add_target()`` to finish the job.
        """
        data_table = Table(names=["TARGET_ID", "TARGET", "RAEP0", "DECEP0",
                                    "EQUINOX", "RA_ERR", "DEC_ERR",
                                    "SYSVEL", "VELTYP", "VELDEF",
                                    "PMRA", "PMDEC", "PMRA_ERR", "PMDEC_ERR", 
                                    "PARALLAX", "PARA_ERR", "SPECTYP", "CATEGORY" ],
                            dtype=[int, str, float, float,
                                    float, float, float,
                                    float, str, str,
                                    float, float, float, float, 
                                    float, float, str, str ],)
        return cls(data_table=data_table)
    def add_target(self, target_id=0, target="MyTarget", raep0=0., decep0=0.,
                        equinox=0., ra_err=0., dec_err=0.,
                        sysvel=0., veltyp="", veldef="",
                        pmra=0., pmdec=0., pmra_err=0., pmdec_err=0., 
                        parallax=0., para_err=0., spectyp="", category="" ):
        """
            Use this method to add a row to the table of targets
        Args:
            param target_id  : (default: 0)
            target     : (default: "MyTarget")
            raep0      : (default: 0.)
            decep0     : (default: 0.)
            equinox    : (default: 0.)
            ra_err     : (default: 0.)
            dec_err    : (default: 0.)
            sysvel     : (default: 0.)
            veltyp     : (default: "")
            veldef     : (default: "")
            pmra       : (default: 0.)
            pmdec      : (default: 0.)
            pmra_err   : (default: 0.)
            pmdec_err  : (default: 0.)
            parallax   : (default: 0.)
            para_err   : (default: 0.)
            spectyp    : (default: "")
            category   : (default: "")
            
        """
        self.data_table.add_row(vals=[target_id, target, raep0, decep0,
                                    equinox, ra_err, dec_err,
                                    sysvel, veltyp, veldef,
                                    pmra, pmdec, pmra_err, pmdec_err, 
                                    parallax, para_err, spectyp, category ])




@dataclass
class NI_CATM(NI_EXTENSION_CPX_ARRAY):
    __doc__ = """
    The complex amplitude transfre function
    """ + NI_EXTENSION_CPX_ARRAY.__doc__
    name = "NI_CATM"
    @property
    def M(self):
        return self.data_array


class NI_IOUT(NI_EXTENSION):
    __doc__ = """
    ``NI_IOUT`` : a recording of the output values, given in intensity,
    flux, counts or arbitrary units. Providing unit is mandatory since 0.2.
        
    """ + NI_EXTENSION.__doc__
    # TODO: resume here proper IO for units
    name = "NI_IOUT"
    @property
    def iout(self):
        return self.data_table["value"].data
    def set_unit(self, new_unit, comment=None):
        if comment is None:
            comment = "The unit of the raw output flux."
        self.header["NIFITS IUNIT"] = (new_unit.to_string(), comment)


class NI_KIOUT(NI_EXTENSION):
    __doc__ = """
    ``NI_KIOUT`` : a recording of the processed output values using the
    the post-processing matrix given by ``NI_KMAT``. Typically differential
    null or kernel-null. Providing unit is mandatory since 0.2. The unit should
    match NI_IOUT.
    """ + NI_EXTENSION_ARRAY.__doc__
    unit: u.Unit = u.photon/u.s
    name = "NI_KIOUT"
    @property
    def kiout(self):
        return self.data_table["value"].data
    @property
    def shape(self):
        return self.data_table["value"].data.shape
    def set_unit(self, new_unit, comment=None):
        if comment is None:
            comment = "The unit of the processed flux."
        self.header["NIFITS IUNIT"] = (new_unit.to_string(), comment)


class NI_KCOV(NI_EXTENSION_ARRAY):
    __doc__ = """
    The covariance matrix for the processed data contained in KIOUT.
     Providing unit is mandatory since 0.2. The unit should be
    the unit of NI_KIOUT ^2.
    """ + NI_EXTENSION_ARRAY.__doc__
    unit: u.Unit = (u.photon/u.s)**2
    name = "NI_KCOV"
    @property
    def kcov(self):
        return self.data_array
    def set_unit(self, new_unit, comment=None):
        if comment is None:
            comment = "The unit of the covariance matrix."
        self.header["NIFITS IUNIT"] = (new_unit.to_string(), comment)


@dataclass
class NI_KMAT(NI_EXTENSION_ARRAY):
    __doc__ = """
    The kernel matrix that defines the post-processing operation between outputs.
    The linear combination is defined by a real-valued matrix.
    """ + NI_EXTENSION_ARRAY.__doc__
    name = "NI_KMAT"
    @property
    def K(self):
        return self.data_array.astype(float)

@dataclass
class NI_DSAMP(NI_EXTENSION_ARRAY):
    __doc__ = """
    The matrix that defines linear combinations of output wavelengths. It is
    meant to be used to down-sample the wavelengths of the forward model. The
    number of columns should match the number of channels described by
    ``NI_OSWAVELENGTH`` and the number of rows should match the number of channels
    described in OI_WAVELENGTH.

    If ``NI_DSAMP`` or ``NI_OSWAVELENGTH`` are missing, then assume the identity matrix
    and possibly skip the computation step.
        
    The linear combination is defined by a real-valued matrix. It is recommended
    that the matrix be semi-unitary on the left, so that a flux conservation is
    observed, and both input and outputs can be described in the same units.
    """ + NI_EXTENSION_ARRAY.__doc__
    name = "NI_DSAMP"
    @property
    def D(self):
        return self.data_array.astype(float)
        

@dataclass
class NI_IOTAGS(NI_EXTENSION):
    r"""
    Contains information on the inputs and outputs.
    """
    data_table: Table = field(default_factory=Table)
    header: fits.Header = field(default_factory=fits.Header)
    name = "NI_IOTAGS"

    @classmethod
    def from_arrays(cls, outbright, outdark, outphot=None,
                        inpola=None, outpola=None, header=None):
        from astropy.table import Column, Table
        outbrightcol = Column(data=outbright,
                       name="BRIGHT", unit=None, dtype=bool)
        outphotcol = Column(data=outphot,
                           name="PHOT", unit=None, dtype=bool)
        outdarkcol = Column(data=outdark,
                           name="DARK", unit=None, dtype=bool)
        inpolcol = Column(data=outpola,
                           name="OUTPOLA", unit=None, dtype=str)
        outpolcol = Column(data=inpola,
                           name="INPOLA", unit=None, dtype=str)
        iotags_table = Table()
        iotags_table.add_columns((outbrightcol, outphotcol, outdarkcol, inpolcol, outpolcol))
        return cls(data_table=iotags_table, header=header)

        
    @property
    def outbright(self):
        """
        The flags of bright outputs
        """
        return self.data_table["BRIGHT"].data
    @property
    def outdark(self):
        """
        The flags of dark outputs
        """
        return self.data_table["DARK"].data
    @property
    def outphot(self):
        """
        The flags of photometric outputs
        """
        return self.data_table["PHOT"].data
    @property
    def outpola(self):
        """
        The polarization of outputs.
        """
        return self.data_table["OUTPOLA"].data
    @property
    def inpola(self):
        """
        The polarization of inputs.
        """
        return self.data_table["INPOLA"].data

    def output_type(self, index, frame_index=0):
        """
        Args:
            index : The index of an output

        Returns:
            output_type : (str) characterization of an output type
        """
        output_type = ""
        if self.outdark[frame_index, index]:
            output_type += f"dark ({self.outpola[frame_index, index]})"
        elif self.outbright[frame_index, index]:
            output_type += f"bright ({self.outpola[frame_index, index]})"
        elif self.outphot[frame_index, index]:
            output_type += f"photometric ({self.outpola[frame_index, index]})"
        else:
            output_type += "undefined"
        return output_type

@dataclass
class NI_MOD(NI_EXTENSION):
    r"""
    Contains input modulation vector for the given observation. The format
    is a complex phasor representing the alteration applied by the instrument
    to the light at its inputs. Either an intended modulation, or an estimated
    instrumental error. the dimenstions are (n_ch, n_in)
    
    The effects modeled in NI_MOD must cumulate with some that may be modeled
    in NI_CATM. It is recommended to include in CATM the static effects and in
    NI_MOD any affect that may vary throughout the observµng run.


    :math:`n_a \times \lambda`

    
    .. table:: ``NI_MOD``: The table of time-dependent collectorwise
    information.

       +---------------+----------------------------+------------------+-------------------+
       | Item          | format                     | unit             | comment           |
       +===============+============================+==================+===================+
       | ``APP_INDEX`` |  ``n_a`` ``int``           | NA               | Indices of        |
       |               |                            |                  | subaperture       |
       |               |                            |                  | (starts at 0)     |
       +---------------+----------------------------+------------------+-------------------+
       | ``TARGET_ID`` |  ``int``                   | d                | Index of target   |
       |               |                            |                  | in ``OI_TARGET``  |
       +---------------+----------------------------+------------------+-------------------+
       | ``TIME``      | ``float``                  | s                | Backwards         |
       |               |                            |                  | compatibility     |
       +---------------+----------------------------+------------------+-------------------+
       | ``MJD``       | ``float``                  | day              |                   |
       +---------------+----------------------------+------------------+-------------------+
       | ``INT_TIME``  | ``float``                  | s                | Exposure time     |
       +---------------+----------------------------+------------------+-------------------+
       | ``MOD_PHAS``  | ``n_{wl},n_a`` ``float``   |                  | Complex phasor of |
       |               |                            |                  | modulation for    |
       |               |                            |                  | all collectors    |
       +---------------+----------------------------+------------------+-------------------+
       | ``APPXY``     | ``n_a, 2`` ``float``       | m                | Projected         |
       |               |                            |                  | location of       |
       |               |                            |                  | subapertures in   |
       |               |                            |                  | the plane         |
       |               |                            |                  | orthogonal to the |
       |               |                            |                  | line of sight and |
       |               |                            |                  | oriented as       |
       |               |                            |                  | ``(               |
       |               |                            |                  | \alpha, \delta)`` |
       +---------------+----------------------------+------------------+-------------------+
       | ``ARRCOL``    | ``n_a`` ``float``          | ``\mathrm{m}^2`` | Collecting area   |
       |               |                            |                  | of the            |
       |               |                            |                  | subaperture       |
       +---------------+----------------------------+------------------+-------------------+
       | ``FOV_INDEX`` | ``n_a`` ``int``            | NA               | The entry of the  |
       |               |                            |                  | ``NI_FOV`` to use |
       |               |                            |                  | for this          |
       |               |                            |                  | subaperture.      |
       +---------------+----------------------------+------------------+-------------------+

    """
    data_table: Table = field(default_factory=Table)
    header: fits.Header = field(default_factory=fits.Header)
    name = "NI_MOD"

    @property
    def n_series(self):
        return len(self.data_table)

    @property
    def all_phasors(self):
        return self.data_table["MOD_PHAS"].data

    @property
    def appxy(self):
        """Shape n_frames x n_a x 2"""
        return self.data_table["APPXY"].data.astype(float)

    @property
    def dateobs(self):
        """
        Get the dateobs from the weighted mean of the observation time
        from each of the observation times given in the rows of ``NI_MOD``
        table.
        """
        raise NotImplementedError("self.dateobs")
        return None

    @property
    def arrcol(self):
        """
        The collecting area of the telescopes
        """
        return self.data_table["ARRCOL"].data

    @property
    def int_time(self):
        """
        Conveniently retrieve the integration time.
        """
        return self.data_table["INT_TIME"].data
        
        
        
def create_basic_fov_data(D, offset, lamb, n):
    """
    A convenience function to help define the FOV function and data model
    """
    r_0 = (lamb/D)*u.rad.to(u.mas)
    def xy2phasor(x,y, offset):
        """
            Returns the complex phasor corresponding to a point in the FOV

        Args:
            ``x`` : ``ArrayLike`` [rad] Position in the FoV along the alpha direction
            ``y`` : ``ArrayLike`` [rad] Position in the FoV along the delta direction
            ``offset`` : ``ArrayLike`` [rad] Chromatic offset of the injection function
                        shape: (``n_{wl}``, ``n_{points}``).
        """
        r = np.hypot(x[None,:]-offset[:,0], y[None,:]-offset[:,1])
        phasor = np.exp(-(r/r_0)**2)
        return phasor.astype(complex)
    all_offsets = np.zeros((n, lamb.shape[0], 2))
    indices = np.arange(n)
    mytable = Table(names=["INDEX", "offsets"],
                    data=[indices, all_offsets])
    return mytable, xy2phasor


class NI_FOV(NI_EXTENSION):
    __doc__ = r"""
    The ``NI_FOV`` data containing information of the field of view (vigneting)
    function as a function of wavelength.

    This can be interpreted in different ways depending on the value of the
    header keyword ``NIFITS FOV_MODE``.

    * ``diameter_gaussian_radial``   : A simple gaussian radial falloff function
      based on a size of :math:`\lambda/D` and a chromatic offset defined for each
      spectral bin. The ``simple_from_header()`` constructor can help create a simple
      extension with 0 offset.
    * More options will come.
    """ + NI_EXTENSION.__doc__
    name = "NI_FOV"
    @classmethod
    def simple_from_header(cls, header=None, lamb=None, n=0):
        r"""
        Constructor for a simple ``NIFITS NI_FOV`` object with chromatic gaussian profile and 
        no offset.

        Args:
            header : (astropy.io.fits.Header) Header containing the required information
                      such as ``NIFITS FOV_TELDIAM`` and ``NIFITS FOV_TELDIAM_UNIT`` which are used to
                      create the gaussian profiles of radius :math:`\lambda/D`
        """
        offset = np.zeros((n,2))
        telescope_diameter_q = header["NIFITS FOV_TELDIAM"]*u.Unit(header["NIFITS FOV_TELDIAM_UNIT"])
        telescope_diameter_m = telescope_diameter_q.to(u.m).value
        mytable, xh2phasor = create_basic_fov_data(telescope_diameter_m,
                                    offset=offset,
                                    lamb=lamb, n=n)
        return cls(data_table=mytable, header=header)



    def __info__(self):
        mode = self.header["HIERARCH NIFITS FOV_MODE"] 
        if mode == "diameter_gaussian_radial":
            mydiam = self.header["HIERARCH NIFITS FOV_TELDIAM"]
            mydiam_unit = self.header["NIFITS FOV_TELDIAM_UNIT"]
            myinfostring = f"""
            ### NI_FOV : definition of the FOV function.
            
            * Mode: {mode}
            * Telescope diameter {mydiam} {mydiam_unit}
            * offsets : {self.data_table["offsets"]}

            """
            return myinfostring


# class NI_MOD(object):
#     """Contains input modulation vector for the given observation. The format
#     is a complex phasor representing the alteration applied by the instrument
#     to the light at its inputs. Either an intended modulation, or an estimated
#     instrumental error. the dimenstions are (n_ch, n_in)
#     
#     The effects modeled in NI_MOD must cumulate with some that may be modeled
#     in NI_CATM. It is recommended to include in CATM the static effects and in
#     NI_MOD any affect that may vary throughout the observing run."""
#     def __init__(self, app_index, target_id, time, mjd,
#                 int_time, mod_phas, app_xy, arrcol,
#                 fov_index):
#         self.app_index = app_index
#         self.target_id = target_id
#         self.time = time
#         self.mjd = mjd
#         self.int_time = int_time
#         self.app_xy = app_xy
#         self.arrcol = arrcol
#         self.fov_index = fov_index
#         self.mod_phas = mod_phas

TEST_BACKUP = True


NIFITS_EXTENSIONS = np.array(["OI_ARRAY",
                    "OI_WAVELENGTH",
                    "NI_CATM",
                    "NI_FOV",
                    "NI_KMAT",
                    "NI_MOD",
                    "NI_IOUT",
                    "NI_KIOUT",
                    "NI_KCOV",
                    "NI_DSAMP",
                    "NI_OSWAVELENGTH",
                    "NI_IOTAGS"])

NIFITS_KEYWORDS = []

STATIC_EXTENSIONS = [True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    False]

def getclass(classname):
    return getattr(sys.modules[__name__], classname)

H_PREFIX = "HIERARCH NIFITS "

@dataclass
class nifits(object):
    """Class representation of the nifits object."""
    header: fits.Header = None
    oi_array: OI_ARRAY = None
    ni_catm: NI_CATM = None
    ni_fov: NI_FOV = None
    ni_kmat: NI_KMAT = None
    oi_wavelength: OI_WAVELENGTH = None
    oi_target: OI_TARGET = None
    ni_mod: NI_MOD = None
    ni_iout: NI_IOUT = None
    ni_kiout: NI_KIOUT = None
    ni_kcov: NI_KCOV = None
    ni_dsamp: NI_DSAMP = None
    ni_oswavelength: NI_OSWAVELENGTH = None
    ni_iotags: NI_IOTAGS = None

    

    @classmethod
    def from_nifits(cls, filename: str):
        """
        Create the nifits object from the HDU extension of an opened fits file.
        """
        if isinstance(filename, fits.hdu.hdulist.HDUList):
            hdulist = filename
        else:
            hdulist = fits.open(filename)
            
        obj_dict = {}
        header = hdulist["PRIMARY"].header
        obj_dict["header"] = header
        for anext in NIFITS_EXTENSIONS:
            if hdulist.__contains__(anext):
                theclass = getclass(anext)
                theobj = theclass.from_hdu(hdulist[anext])
                obj_dict[anext.lower()] = theobj
            else:
                print(f"Missing {anext}")
        print("Checking header", isinstance(header, fits.Header))
        print("contains_header:", obj_dict.__contains__("header"))
        return cls(**obj_dict)

    def to_nifits(self, filename:str = "",
                        static_only: bool = False,
                        dynamic_only: bool = False,
                        static_hash: str = "",
                        writefile: bool = True,
                        overwrite: bool = False):
        """
        Write the extension objects to a nifits file.

        Args: 
            static_only :  (bool) only save the extensions corresponding
                          to static parameters of the model (NI_CATM and NI_FOV). 
                          Default: False
            dynamic_only : (bool) only save the dynamic extensions. If true,
                          the hash of the static file should be passed as `static_hash`.
                          Defaultult: False
            static_hash : (str) The hash of the static file.
                        Default: ""

        """
        # TODO: Possibly, the static_hash should be a dictionary with
        # a hash for each extension
        self.header["HIERARCH NIFITS VERSION"] = (__standard_version__,
                            f"Writen with rlaugier/nifits v{__version__}")
        
        hdulist = fits.HDUList()
        hdu = fits.PrimaryHDU()
        if static_only:
            extension_list = NIFITS_EXTENSIONS[STATIC_EXTENSIONS]
        elif dynamic_only:
            extension_list = NIFITS_EXTENSIONS[np.logical_not(STATIC_EXTENSIONS)]
        else:
            extension_list = NIFITS_EXTENSIONS
        hdulist.append(hdu)
        for anext in extension_list:
            print(anext, hasattr(self,anext.lower()))
            if hasattr(self, anext.lower()):
                print(anext.lower(), flush=True)
                theobj = getattr(self, anext.lower())
                if theobj is not None:
                    thehdu = theobj.to_hdu()
                    hdulist.append(thehdu)
                    hdu.header[H_PREFIX + anext] = "Included"
                    # TODO Possibly we need to do this differently:
                    # TODO Maybe pass the header to the `to_hdu` method?
                else:
                    hdu.header[H_PREFIX + anext] = "Not included (None)"
                    print(f"Warning: {anext} was present but empty")
            else:
                hdu.header[H_PREFIX + anext] = "Not included"
                print(f"Warning: Could not find the {anext} object")
        print(hdu.header)
        if writefile:
            hdulist.writeto(filename, overwrite=overwrite)
            return hdulist
        else:
            return hdulist

    def check_unit_coherence(self):
        """
            Check the coherence of the units of and prints the result
        NI_IOUT, NI_KCOV, and NI_KIOUT if they exist.

        Otherwise, does nothing.
        """
        if hasattr(self, "ni_iout"):
            print("NI_IOUT", self.ni_iout.unit)
        else:
            print("No NI_IOUT")
        if hasattr(self, "ni_kiout"):
            print("NI_KIOUT", self.ni_kiout.unit)
        else:
            print("No NI_KIOUT")
        if hasattr(self, "ni_kcov"):
            print("NI_KCOV", self.ni_kcov.unit)
        else:
            print("No NI_KCOV")
            
        if hasattr(self, "ni_iout") and hasattr(self, "ni_kiout"):
            print(self.ni_iout.unit.is_equivalent(self.ni_kiout.unit))
        if hasattr(self, "ni_kcov") and hasattr(self, "ni_kiout"):
            print(np.sqrt(self.ni_kcov.unit).is_equivalent(self.ni_kiout.unit))
        if hasattr(self, "ni_kcov") and hasattr(self, "ni_iout"):
            print(np.sqrt(self.ni_kcov.unit).is_equivalent(self.ni_iout.unit))










