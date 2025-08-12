import numpy as np

from nifits.backend import NI_Backend
from tests.base_nifits_test_case import BaseNIFITSTestCase


class TestNIFITSPhasor(BaseNIFITSTestCase):

    def setUp(self):
        super().setUp()
        self.backend = NI_Backend()
        self.backend.nifits = self.nifits
        self.n_dim = 2

    def test_xy2phasor_valid_input(self):
        self.backend.create_fov_function_all()
        x = np.ones(self.n_dim)
        y = np.ones(self.n_dim)

        phasor = self.backend.nifits.ni_fov.xy2phasor(x, y)
        self.assertEqual(phasor.dtype, complex)
        self.assertEqual(phasor.shape, (self.n_time, self.n_wl_bin, self.n_dim))

    def test_xy2phasor_moving_valid_input(self):
        self.backend.create_fov_function_all()
        x = np.ones((self.n_time, self.n_wl_bin, self.n_dim))
        y = np.ones((self.n_time, self.n_wl_bin, self.n_dim))

        phasor_moving = self.backend.nifits.ni_fov.xy2phasor_moving(x, y)
        self.assertEqual(phasor_moving.dtype, complex)
        self.assertEqual(phasor_moving.shape, (self.n_time, self.n_time, self.n_wl_bin, self.n_dim))

    def test_get_modulation_phasor_valid_input(self):
        result = self.backend.get_modulation_phasor()

        self.assertEqual(result.dtype, complex)
        self.assertEqual(result.shape, (self.n_time, self.n_wl_bin, self.n_diff_out))

    def test_geometric_phasor_with_modulation(self):
        alpha = np.ones(self.n_dim)
        beta = np.ones(self.n_dim)

        result = self.backend.geometric_phasor(alpha, beta, include_mod=True)
        self.assertEqual(result.dtype, complex)
        self.assertEqual(result.shape, (self.n_time, self.n_wl_bin, self.n_in, self.n_dim))

    def test_geometric_phasor_without_modulation(self):
        alpha = np.ones(self.n_dim)
        beta = np.ones(self.n_dim)

        result = self.backend.geometric_phasor(alpha, beta, include_mod=False)
        self.assertEqual(result.dtype, complex)
        self.assertEqual(result.shape, (self.n_time, self.n_wl_bin, self.n_in, self.n_dim))

    def test_moving_geometric_phasor_with_modulation(self):
        self.backend.create_fov_function_all()
        alpha = np.ones((self.n_time, self.n_dim))
        beta = np.ones((self.n_time, self.n_dim))

        result = self.backend.moving_geometric_phasor(alpha, beta, include_mod=True)
        self.assertEqual(result.shape, (self.n_time, self.n_wl_bin, self.n_in, self.n_dim))

    def test_moving_geometric_phasor_without_modulation(self):
        self.backend.create_fov_function_all()
        alpha = np.ones((self.n_time, self.n_dim))
        beta = np.ones((self.n_time, self.n_dim))

        result = self.backend.moving_geometric_phasor(alpha, beta, include_mod=True)
        self.assertEqual(result.dtype, complex)
        self.assertEqual(result.shape, (self.n_time, self.n_wl_bin, self.n_in, self.n_dim))
