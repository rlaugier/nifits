from nifits.backend import col_row_numbers
from tests.base_test_case import BaseTestCase


class TestBackend(BaseTestCase):

    def test_small_number_of_items(self):
        nrows, ncols = col_row_numbers(6)
        self.assertEqual(nrows, 3)
        self.assertEqual(ncols, 2)

    def test_exceeding_col_ceiling(self):
        nrows, ncols = col_row_numbers(10, col_ceiling=3)
        self.assertEqual(nrows, 4)
        self.assertEqual(ncols, 3)

    def test_single_item(self):
        nrows, ncols = col_row_numbers(1)
        self.assertEqual(nrows, 1)
        self.assertEqual(ncols, 1)

    def test_custom_col_ceiling(self):
        nrows, ncols = col_row_numbers(12, col_ceiling=5)
        self.assertEqual(nrows, 4)
        self.assertEqual(ncols, 3)
