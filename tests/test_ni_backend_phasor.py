from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from nifits.backend import NI_Backend


class TestNI_Backend(TestCase):

    def setUp(self):
        self.backend = NI_Backend()
        self.mock_nifits = MagicMock()
        self.backend.nifits = self.mock_nifits

    def test_xy2phasor(self):
        # Mock header and data_table
        self.mock_nifits.ni_fov.header = {
            "NIFITS FOV_MODE": "diameter_gaussian_radial",
            "NIFITS FOV_TELDIAM": 10,
            "NIFITS FOV_TELDIAM_UNIT": "m"
        }
        self.mock_nifits.oi_wavelength.lambs = np.array([1.0, 2.0, 3.0])
        self.mock_nifits.ni_fov.data_table = {"offsets": np.ones((3, 1, 2))}

        # Call the method
        self.backend.create_fov_function_all()

        # Test xy2phasor with sample inputs
        x = np.array([0.1, 0.2])
        y = np.array([0.3, 0.4])

        phasor = self.backend.nifits.ni_fov.xy2phasor(x, y)
        self.assertEqual(phasor.dtype, complex)
        self.assertEqual(phasor.shape, (3, 3, 2))

    def test_xy2phasor_moving_function(self):
        # Mock header and data_table
        self.mock_nifits.ni_fov.header = {
            "NIFITS FOV_MODE": "diameter_gaussian_radial",
            "NIFITS FOV_TELDIAM": 10,
            "NIFITS FOV_TELDIAM_UNIT": "m"
        }
        self.mock_nifits.oi_wavelength.lambs = np.array([1.0, 2.0, 3.0])
        self.mock_nifits.ni_fov.data_table = {"offsets": np.ones((2, 1, 2))}

        # Call the method
        self.backend.create_fov_function_all()

        # Test xy2phasor_moving with sample inputs
        x = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([[0.5, 0.6], [0.7, 0.8]])

        phasor_moving = self.backend.nifits.ni_fov.xy2phasor_moving(x, y)
        self.assertEqual(phasor_moving.dtype, complex)
        self.assertEqual(phasor_moving.shape, (2, 3, 2))

    def test_get_modulation_phasor_valid_data(self):
        # Mock valid data
        self.mock_nifits.ni_mod.all_phasors = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        self.mock_nifits.ni_mod.arrcol = np.ones((2, 2))

        # Call the method
        result = self.backend.get_modulation_phasor()
        print(result)

        # Expected result
        expected = np.array([[[1. + 1.j, 2. + 2.j],
                              [3. + 3.j, 4. + 4.j]],
                             [[1. + 1.j, 2. + 2.j],
                              [3. + 3.j, 4. + 4.j]]])

        # Assertions
        np.testing.assert_array_equal(result, expected)

    def test_get_modulation_phasor_invalid_data(self):
        # Mock invalid data
        self.mock_nifits.ni_mod.all_phasors = None
        self.mock_nifits.ni_mod.arrcol = None

        # Assertions
        with self.assertRaises(TypeError):
            self.backend.get_modulation_phasor()
