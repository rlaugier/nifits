"""
NIFITS Extra
-------------

The NIFITS extra contains some additional tools to go somewhat **beyond** the
a simple implementation of the NIFITS standard, and facilitate the usage of 
common powerful techniques of nulling interferometry data reduction and
interpretation.

This includes:

* statistical whitening of observables (Ceau et al. 2019, Laugier et al. 2023)
* Correlation maps (coming soon)
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
                                md: types.ModuleType = np):
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
        if self.nifits.ni_kcov.header["SHAPE"] != "frame (wavelength output)":
            raise NotImplementedError("Covariance shape expected: frame (wavelength output)")

        Ws = []
        for amat in self.nifits.ni_kcov.data_array:
            Ws.append(np.linalg.inv(sqrtm(amat)))
        self.Ws = md.array(Ws)
        if hasattr(self.nifits, "ni_kiout"):
            self.datashape = (self.nifits.ni_kiout.shape) 
        self.flatshape = (len(self.nifits.ni_mod), self.nifits.ni_kcov.shape[0])

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


    def whitened_outputs(self, func):
        """
        A decorator methods that transform a forward model method into 
        """
        def inner(*args, **kwargs):
            output = func(*args, **kwargs)
            full_shape = output.shape
            flat_full_shape = (*self.flatshape, full_shape[-1])
            flat_out = rearrange(output, "frame wavelength output source -> frame (wavelength output) source")
            wout = self.md.einsum("f o i , f i m -> f o m", self.Ws, flat_out)
            wout_full = wout.reshape((full_shape))
            return wout_full
        inner.__doc__ = """**Modified to whiten the observables**\n\n"""\
                    + func.__doc__
        return inner


def whitened_kiout(self):
    data = self.data_table["value"].data
    full_shape = data.shape
    flat_full_shape = (full_shape[0], full_shape[1]*full_shape[2])
    flat_out = rearrange(data, "frame wavelength output -> frame (wavelength output)")
    wout = self.md.einsum("f o i , f i -> f o", self.Ws, flat_out)
    wout_full = wout.reshape((full_shape))
    return wout_full




        
