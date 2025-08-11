import numpy as np

from nifits.backend import NI_Backend
from tests.base_nifits_test_case import BaseNIFITSTestCase


class TestNIFITSPhasor(BaseNIFITSTestCase):

    def setUp(self):
        super().setUp()
        self.backend = NI_Backend()
        self.backend.nifits = self.nifits

    def test_xy2phasor_valid_input(self):
        self.backend.create_fov_function_all()
        x = np.array([0.1, 0.2])
        y = np.array([0.3, 0.4])

        phasor = self.backend.nifits.ni_fov.xy2phasor(x, y)
        self.assertEqual(phasor.dtype, complex)
        self.assertEqual(phasor.shape, (100, 5, 2))

    def test_xy2phasor_moving_valid_input(self):
        self.backend.create_fov_function_all()
        x = np.ones((100, 5, 1))
        y = np.ones((100, 5, 1))

        phasor_moving = self.backend.nifits.ni_fov.xy2phasor_moving(x, y)
        self.assertEqual(phasor_moving.dtype, complex)
        self.assertEqual(phasor_moving.shape, (100, 100, 5, 1))

    def test_get_modulation_phasor_valid_input(self):
        result = self.backend.get_modulation_phasor()

        expected = 2.658680776358274 + 0j * np.zeros((100, 5, 3), dtype=complex)
        np.testing.assert_array_equal(result, expected)

    def test_geometric_phasor_with_modulation(self):
        alpha = np.ones((2))
        beta = np.ones((2))

        result = self.backend.geometric_phasor(alpha, beta, include_mod=True)
        self.assertEqual(result.dtype, complex)
        self.assertEqual(result.shape, (100, 5, 3, 2))

    def test_geometric_phasor_without_modulation(self):
        alpha = np.ones((2))
        beta = np.ones((2))

        result = self.backend.geometric_phasor(alpha, beta, include_mod=False)
        self.assertEqual(result.dtype, complex)
        self.assertEqual(result.shape, (100, 5, 3, 2))

    def test_moving_geometric_phasor_with_modulation(self):
        self.backend.create_fov_function_all()
        alpha = np.ones((100, 2))
        beta = np.ones((100, 2))

        result = self.backend.moving_geometric_phasor(alpha, beta, include_mod=True)
        self.assertEqual(result.shape, (100, 5, 3, 2))

    def test_moving_geometric_phasor_without_modulation(self):
        self.backend.create_fov_function_all()
        alpha = np.ones((100, 2))
        beta = np.ones((100, 2))

        result = self.backend.moving_geometric_phasor(alpha, beta, include_mod=True)
        self.assertEqual(result.dtype, complex)
        self.assertEqual(result.shape, (100, 5, 3, 2))
