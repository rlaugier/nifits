from tests.base_nifits_test_case import BaseNIFITSTestCase
from unittest import TestCase

import nifits.io.niio as io

from nifits import __version__ as lib_version
from nifits import __standard_version__ as std_version

import numpy as np

class Test_Version(TestCase):
    def test_lib_version(self):
        self.assertEqual(io.__version__, lib_version)
        io.__version__ == f"{io.__version_int__()[0]}.{io.__version_int__()[1]}.{io.__version_int__()[2]}"
        from nifits.backend import __version__
        self.assertEqual(io.__version__, __version__)

    def test_standard_version(self):
        self.assertEqual(io.__standard_version__, std_version)
        io.__standard_version__ == f"{io.__standard_version_int__()[0]}.{io.__standard_version_int__()[1]}"
        from nifits.backend import __standard_version__
        self.assertEqual(io.__standard_version__, __standard_version__)

ANY_FLOAT = (np.float64, np.float32, float)
ANY_COMPLEX = (np.complex64, np.complex128, complex)
ANY_INT = (np.int16, np.int32, int)

reference_header = {
    "SIMPLE":( (bool,), True),
    "BITPIX":( (ANY_INT), True),
    "NAXIS":( ANY_INT, True),
    "EXTEND":( (bool,), True),
    "INSTRUME":( (str,), True),
    "HIERARCH NIFITS NI_RMAJ":( (np.int16,), True),
    "HIERARCH NIFITS NI_RMIN":( (np.int16,), True),
    "HIERARCH NIFITS LIB_NAME":( (str,), True),
    "HIERARCH NIFITS LIB_REV":( (str,), True),
    "ORIGIN":( (str,), True),
    "DATE":( (str,), True),
    "DATE-OBS":( (str,), True),
    "CONTENT":( (str,), True),
    "AUTHOR":( (str,), False),
    "DATASUM":( (str,), False),
    "CHECKSUM":( (str,), False),
    "TELESCOP":( (str,), True),
    "INSTRUME":( (str,), True),
    "OBSERVER":( (str,), True),
    "OBJECT":( (str,), True),
    "INSMODE":( (str,), True),
    "REFERENC":( (str,), False),
    "PROG_ID":( (str,), False),
    "PROCSOFT":( (str,), False),
    "OBSTECH":( (str,), False),
    "RA":( ANY_FLOAT, False),
    "DEC":( ANY_FLOAT, False),
    "EQUINOX":( ANY_FLOAT, False),
    "RADECSYS":( (str,), False),
    "SPECSYS":( (str,), False),
    "TEXPTIME":( ANY_FLOAT, False),
    "MJD-OBS":( ANY_FLOAT, False),
    "MJD-END":( ANY_FLOAT, False),
    "BASE_MIN":( ANY_FLOAT, False),
    "BASE_MAX":( ANY_FLOAT, False),
    "WAVELMIN":( ANY_FLOAT, False),
    "WAVELMAX":( ANY_FLOAT, False),
    "NUM_CHAN":( (str,), False),
    "SPEC_RES":( (str,), False),
    "HIERARCH NIFITS NULLERR":( (np.float64,), False),
}


class Test_Header(BaseNIFITSTestCase):
    def test_primary_header(self):
        self.assertTrue("HIERARCH NIFITS NI_RMAJ" in self.nifits.header)
        self.assertEqual(io.__standard_version_int__()[0], self.nifits.header["HIERARCH NIFITS NI_RMAJ"])
        self.assertTrue("HIERARCH NIFITS NI_RMIN" in self.nifits.header)
        self.assertEqual(io.__standard_version_int__()[1], self.nifits.header["HIERARCH NIFITS NI_RMIN"])
        self.assertEqual(len(self.nifits.get_version()), 2)# Checking versioning with 2 numbers
        self.assertEqual(self.nifits.get_version(), io.__standard_version_int__())
        print(self.nifits.header)
        for akey, aref in reference_header.items():
            print(f"Checking {akey}")
            self.header_type_check(self.nifits.header, akey, aref)

    def test_OI_WAVELEGTH_header(self):
        self.assertTrue("OI_REVN" in self.nifits.oi_wavelength.header)
        self.assertEqual(self.nifits.oi_wavelength.header["OI_REVN"], 2)

    def header_type_check(self, myheader, key, content):
        if content[-1]:
            self.assertTrue(key in myheader, msg=f"Mandatory key not found {key}")
            self.assertTrue(type(myheader[key]) in content[0], msg="Wrong type {type(myheader[key])} for keyword {key}")
        elif key in myheader:
            self.assertTrue(type(myheader[key]) in content[0], msg="Wrong type {type(myheader[key])} for keyword {key}")



