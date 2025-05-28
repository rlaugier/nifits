import unittest

import numpy as np

from nifits.backend.backend import PointCollection


class TestPointCollection(unittest.TestCase):
    def test_creates_uniform_disk_with_correct_number_of_points(self):
        radius = 10
        n_points = 100
        point_collection = PointCollection.from_uniform_disk(radius, n=n_points)
        self.assertEqual(point_collection.aa.shape[0], n_points)
        self.assertEqual(point_collection.bb.shape[0], n_points)

    def test_handles_zero_radius_for_uniform_disk_creation(self):
        radius = 0
        n_points = 100
        self.assertRaises(ValueError, lambda: PointCollection.from_uniform_disk(radius, n=n_points))

    def test_handles_negative_radius_for_uniform_disk_creation(self):
        radius = -10
        n_points = 100
        self.assertRaises(ValueError, lambda: PointCollection.from_uniform_disk(radius, n=n_points))

    def test_handles_empty_grid_creation(self):
        a_coords = np.array([])
        b_coords = np.array([])
        self.assertRaises(ValueError, lambda: PointCollection.from_grid(a_coords, b_coords))

    def test_creates_segment_with_correct_number_of_samples(self):
        start_coords = np.array([0, 0])
        end_coords = np.array([10, 10])
        n_samples = 50
        point_collection = PointCollection.from_segment(start_coords, end_coords, n_samples)
        self.assertEqual(point_collection.aa.shape[0], n_samples)
        self.assertEqual(point_collection.bb.shape[0], n_samples)

    def test_adds_two_point_collections_correctly(self):
        pc1 = PointCollection.from_uniform_disk(10, n=50)
        pc2 = PointCollection.from_uniform_disk(5, n=30)
        combined = pc1 + pc2
        self.assertEqual(combined.aa.shape[0], 80)
        self.assertEqual(combined.bb.shape[0], 80)

    def test_transforms_coordinates_correctly(self):
        pc = PointCollection.from_uniform_disk(10, n=50)
        matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pc.transform(matrix)
        self.assertTrue(np.allclose(pc.aa, pc.aa))
        self.assertTrue(np.allclose(pc.bb, pc.bb))


if __name__ == '__main__':
    unittest.main()
