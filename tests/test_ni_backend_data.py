import unittest
from copy import copy

import numpy as np

from nifits.backend.backend import NI_Backend
from nifits.io.oifits import NIFITS_EXTENSIONS, STATIC_EXTENSIONS
from tests.base_ni_backend_test_case import BaseNIBackendTestCase


class TestNIBackendData(BaseNIBackendTestCase):
    def setUp(self):
        super().setUp()
        self.backend = NI_Backend()

    def test_add_observation_data_when_nifits_is_none(self):
        self.backend.add_observation_data(nifits_data=self.nifits)
        self.assertEqual(self.backend.nifits, self.nifits)

    def test_add_observation_data_with_force_true(self):
        new_nifits = copy(self.nifits)
        self.backend.nifits = self.nifits

        self.backend.add_observation_data(nifits_data=new_nifits, force=True)

        for ext in NIFITS_EXTENSIONS[np.logical_not(STATIC_EXTENSIONS)]:
            self.assertEqual(getattr(self.backend.nifits, ext.lower()), getattr(new_nifits, ext.lower()))

    def test_add_observation_data_with_force_false(self):
        new_nifits = copy(self.nifits)
        self.backend.nifits = self.nifits

        self.backend.add_observation_data(nifits_data=new_nifits, force=False)

        for ext in NIFITS_EXTENSIONS[np.logical_not(STATIC_EXTENSIONS)]:
            self.assertEqual(getattr(self.backend.nifits, ext.lower()), getattr(new_nifits, ext.lower()))

    def test_add_observation_data_with_new_attribute(self):
        new_nifits = copy(self.nifits)
        self.backend.nifits = self.nifits
        missing_extension = NIFITS_EXTENSIONS[STATIC_EXTENSIONS][0]

        self.backend.add_observation_data(nifits_data=new_nifits, force=False)

        for ext in NIFITS_EXTENSIONS[np.logical_not(STATIC_EXTENSIONS)]:
            if ext.lower() != missing_extension:
                self.assertEqual(getattr(self.backend.nifits, ext.lower()), getattr(self.nifits, ext.lower()))
            else:
                self.assertFalse(hasattr(self.backend.nifits, ext.lower()))


if __name__ == "__main__":
    unittest.main()
