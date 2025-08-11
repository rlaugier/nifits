import numpy as np

from nifits.backend import NI_Backend
from tests.base_test_case import BaseTestCase


class TestNI_BackendIntensity(BaseTestCase):

    def setUp(self):
        super().setUp()
        self.backend = NI_Backend()
        self.backend.nifits = self.nifits

    def test_get_all_outs_without_kernels(self):
        alphas = np.array([0.1, 0.2])
        betas = np.array([0.3, 0.4])

        # Call the method with kernels=False
        result = self.backend.get_all_outs(alphas, betas, kernels=False)

        # Assertions
        self.assertIsNotNone(result)

    def test_get_all_outs_with_kernels(self):
        alphas = np.array([0.1, 0.2])
        betas = np.array([0.3, 0.4])

        # Call the method with kernels=False
        result = self.backend.get_all_outs(alphas, betas, kernels=True)

        # Assertions
        self.assertIsNotNone(result)

    def test_get_moving_outs_valid_input(self):
        alphas = np.ones(4)
        betas = np.ones(4)

        # Call the method
        result = self.backend.get_moving_outs(alphas, betas)

        # Assertions
        self.assertIsNotNone(result)

    def test_get_moving_outs_invalid_input(self):
        alphas = None
        betas = None

        # Assertions
        with self.assertRaises(TypeError):
            self.backend.get_moving_outs(alphas, betas)

    def test_downsample_valid_input(self):
        Is = np.ones((2, 2, 2, 4))  # Example input

        # Call the method
        result = self.backend.downsample(Is)

        # Assertions
        self.assertIsNotNone(result)

    def test_downsample_empty_input(self):
        Is = np.array([])  # Empty input

        # Call the method
        result = self.backend.downsample(Is)

        # Assertions
        self.assertEqual(result.size, 0)

    def test_downsample_invalid_input(self):
        Is = None  # Invalid input

        # Assertions
        with self.assertRaises(TypeError):
            self.backend.downsample(Is)
