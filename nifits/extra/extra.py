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
    This variant of the backend class offers the same interface but with directly whitened observables

    Use `get_all_outs` and `get_moving_outs` in the same way, but they return whitened observables
    """
    def create_whitening_matrix(self,
                                replace: bool = False,
                                md: types.ModuleType = np):
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


def decorate_print(func):
    def inner(*args, **kwargs):
        output = func(*args, **kwargs)
        print(output)
    return inner


# def whitened_outputs(func):
#     def inner(self, *args, **kwargs):
#         output = func(self, *args, **kwargs)
#         full_shape = output.shape
#         flat_full_shape = (*self.flatshape, full_shape[-1])
#         flat_out = rearrange(output, "frame wavelength output source -> frame (wavelength output) source")
#         wout = self.md.einsum("f o i , f o m -> f w", self.Ws, flat_out)
#         wout_full = wout.reshape((full_shape))
#         return wout_full
#     inner.__doc__ = """**Modified to whiten the observables**\n\n"""\
#                 + func.__doc__
#     return inner


        
