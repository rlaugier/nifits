import numpy as np

from nifits.backend import NI_Backend
from tests.base_nifits_test_case import BaseNIFITSTestCase


class TestNIFITSIntensity(BaseNIFITSTestCase):

    def setUp(self):
        super().setUp()
        self.backend = NI_Backend()
        self.backend.nifits = self.nifits

    def test_get_Is_valid_input(self):
        xs = np.ones((1, 1, 1, 1))
        result = self.backend.get_Is(xs)

        expected = np.abs(np.einsum("w o i , t w i m -> t w o m", self.nifits.ni_catm.M, xs)) ** 2

        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, expected.shape)

    def test_get_KIs_valid_input(self):
        i_array = np.ones((1, 1, 1, 1))  # Example input

        # Call the method
        result = self.backend.get_KIs(i_array)

        # Expected result
        expected = np.einsum("k o, t w o m -> t w k m", self.nifits.ni_kmat.K[:, :], i_array)

        # Assertions
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, expected.shape)
