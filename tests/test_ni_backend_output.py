import numpy as np

from nifits.backend import NI_Backend
from tests.base_ni_backend_test_case import BaseNIBackendTestCase


class TestNIBackendOutput(BaseNIBackendTestCase):

    def setUp(self):
        super().setUp()
        self.backend = NI_Backend()
        self.backend.nifits = self.nifits

    def test_get_all_outs_without_kernels(self):
        self.backend.create_fov_function_all()
        alphas = np.array([0.1, 0.2])
        betas = np.array([0.3, 0.4])

        result = self.backend.get_all_outs(alphas, betas, kernels=False)
        self.assertEqual(result.shape, (100, 5, 3, 2))

    def test_get_all_outs_with_kernels(self):
        self.backend.create_fov_function_all()
        alphas = np.array([0.1, 0.2])
        betas = np.array([0.3, 0.4])

        result = self.backend.get_all_outs(alphas, betas, kernels=True)
        self.assertEqual(result.shape, (100, 5, 1, 2))

    def test_get_moving_outs_without_kernels(self):
        self.backend.create_fov_function_all()
        alphas = np.ones((100, 4))
        betas = np.ones((100, 4))

        result = self.backend.get_moving_outs(alphas, betas, kernels=False)
        self.assertEqual(result.shape, (100, 5, 3, 4))

    def test_get_moving_outs_with_kernels(self):
        self.backend.create_fov_function_all()
        alphas = np.ones((100, 4))
        betas = np.ones((100, 4))

        result = self.backend.get_moving_outs(alphas, betas, kernels=True)
        self.assertEqual(result.shape, (100, 5, 1, 4))

    def test_downsample_valid_input(self):
        self.backend.create_fov_function_all()
        Is = np.ones((100, 5, 2, 4))

        result = self.backend.downsample(Is)
        self.assertEqual(result.shape, (100, 5, 2, 4))
