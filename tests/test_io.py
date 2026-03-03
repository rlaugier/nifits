from tests.base_nifits_test_case import BaseNIFITSTestCase
from unittest import TestCase

import nifits.io.niio as io

from nifits import __version__ as lib_version
from nifits import __standard_version__ as std_version

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

class Test_Header(BaseNIFITSTestCase):
    def test_primary_header(self):
        self.assertTrue("HIERARCH NIFITS NI_RMAJ" in self.nifits.header)
        self.assertEqual(io.__standard_version_int__()[0], self.nifits.header["HIERARCH NIFITS NI_RMAJ"])
        self.assertTrue("HIERARCH NIFITS NI_RMIN" in self.nifits.header)
        self.assertEqual(io.__standard_version_int__()[1], self.nifits.header["HIERARCH NIFITS NI_RMIN"])
        self.assertEqual(len(self.nifits.get_version()), 2)# Checking versioning with 2 numbers
        self.assertEqual(self.nifits.get_version(), io.__standard_version_int__())

    def test_OI_WAVELEGTH_header(self):
        self.assertTrue("OI_REVN" in self.nifits.oi_wavelength.header)
        self.assertEqual(self.nifits.oi_wavelength.header["OI_REVN"], 2)



