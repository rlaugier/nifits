import numpy as np

from nifits.backend import NI_Backend
from tests.base_test_case import BaseTestCase


class TestNI_BackendPhasor(BaseTestCase):

    def setUp(self):
        super().setUp()
        self.backend = NI_Backend()
        self.backend.nifits = self.nifits

    def test_xy2phasor(self):
        # Call the method
        self.backend.create_fov_function_all()

        # Test xy2phasor with sample inputs
        x = np.array([0.1, 0.2])
        y = np.array([0.3, 0.4])

        phasor = self.backend.nifits.ni_fov.xy2phasor(x, y)
        self.assertEqual(phasor.dtype, complex)
        self.assertEqual(phasor.shape, (100, 5, 2))

    def test_xy2phasor_moving(self):
        # Call the method
        self.backend.create_fov_function_all()

        # Test xy2phasor_moving with sample inputs
        x = np.ones((100, 5, 1))
        y = np.ones((100, 5, 1))

        phasor_moving = self.backend.nifits.ni_fov.xy2phasor_moving(x, y)
        self.assertEqual(phasor_moving.dtype, complex)
        self.assertEqual(phasor_moving.shape, (100, 100, 5, 1))

    def test_get_modulation_phasor(self):
        # Call the method
        result = self.backend.get_modulation_phasor()

        # Expected result
        expected = 2.658680776358274 + 0j * np.zeros((100, 5, 3), dtype=complex)

        # Assertions
        np.testing.assert_array_equal(result, expected)

    def test_geometric_phasor_with_modulation(self):
        alpha = np.ones((2))
        beta = np.ones((2))

        # Call the method with include_mod=True
        result = self.backend.geometric_phasor(alpha, beta, include_mod=True)

        # Assertions
        self.assertEqual(result.dtype, complex)
        self.assertEqual(result.shape, (100, 5, 3, 2))
