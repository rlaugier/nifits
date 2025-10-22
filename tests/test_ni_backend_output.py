import numpy as np

from nifits.backend import NI_Backend
from tests.base_nifits_test_case import BaseNIFITSTestCase


class TestNIFITSOutput(BaseNIFITSTestCase):

    def setUp(self):
        super().setUp()
        self.backend = NI_Backend()
        self.backend.nifits = self.nifits
        self.n_dim = 2
        self.n_dim2 = 6

    def test_get_all_outs_without_kernels(self):
        self.backend.create_fov_function_all()
        alphas = np.ones(self.n_dim)
        betas = np.ones(self.n_dim)

        result = self.backend.get_all_outs(alphas, betas, kernels=False)
        self.assertEqual(result.shape, (self.n_time, self.n_wl_bin, self.n_out, self.n_dim))

    def test_get_all_outs_with_kernels(self):
        self.backend.create_fov_function_all()
        alphas = np.ones(self.n_dim)
        betas = np.ones(self.n_dim)

        result = self.backend.get_all_outs(alphas, betas, kernels=True)
        self.assertEqual(result.shape, (self.n_time, self.n_wl_bin, self.n_diff_out, self.n_dim))

    def test_get_moving_outs_without_kernels(self):
        self.backend.create_fov_function_all()
        alphas = np.ones((self.n_time, self.n_dim2))
        betas = np.ones((self.n_time, self.n_dim2))

        result = self.backend.get_moving_outs(alphas, betas, kernels=False)
        self.assertEqual(result.shape, (self.n_time, self.n_wl_bin, self.n_out, self.n_dim2))

    def test_get_moving_outs_with_kernels(self):
        self.backend.create_fov_function_all()
        alphas = np.ones((self.n_time, self.n_dim2))
        betas = np.ones((self.n_time, self.n_dim2))

        result = self.backend.get_moving_outs(alphas, betas, kernels=True)
        self.assertEqual(result.shape, (self.n_time, self.n_wl_bin, self.n_diff_out, self.n_dim2))

    def test_downsample_valid_input(self):
        self.backend.create_fov_function_all()
        Is = np.ones((self.n_time, self.n_wl_bin, self.n_dim, self.n_dim2))

        result = self.backend.downsample(Is)
        self.assertEqual(result.shape, (self.n_time, self.n_wl_bin, self.n_dim, self.n_dim2))
