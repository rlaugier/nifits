import unittest
from unittest.mock import MagicMock

import numpy as np

from nifits.backend.backend import NI_Backend
from nifits.io.oifits import NIFITS_EXTENSIONS, STATIC_EXTENSIONS


class TestNI_BackendData(unittest.TestCase):
    def setUp(self):
        self.backend = NI_Backend()

    def test_add_observation_data_when_nifits_is_none(self):
        mock_nifits = MagicMock()
        self.backend.add_observation_data(nifits_data=mock_nifits)
        self.assertEqual(self.backend.nifits, mock_nifits)

    def test_add_observation_data_with_force_true(self):
        mock_existing_nifits = MagicMock()
        mock_new_nifits = MagicMock()
        self.backend.nifits = mock_existing_nifits

        # Simulate extensions in the new nifits object
        for ext in NIFITS_EXTENSIONS[np.logical_not(STATIC_EXTENSIONS)]:
            setattr(mock_existing_nifits, ext.lower(), MagicMock())
            setattr(mock_new_nifits, ext.lower(), MagicMock())

        self.backend.add_observation_data(nifits_data=mock_new_nifits, force=True)

        for ext in NIFITS_EXTENSIONS[np.logical_not(STATIC_EXTENSIONS)]:
            self.assertEqual(getattr(self.backend.nifits, ext.lower()), getattr(mock_new_nifits, ext.lower()))

    def test_add_observation_data_with_force_false(self):
        mock_existing_nifits = MagicMock()
        mock_new_nifits = MagicMock()
        self.backend.nifits = mock_existing_nifits

        # Simulate extensions in both existing and new nifits objects
        for ext in NIFITS_EXTENSIONS[np.logical_not(STATIC_EXTENSIONS)]:
            setattr(mock_existing_nifits, ext.lower(), MagicMock())
            setattr(mock_new_nifits, ext.lower(), MagicMock())

        self.backend.add_observation_data(nifits_data=mock_new_nifits, force=False)

        for ext in NIFITS_EXTENSIONS[np.logical_not(STATIC_EXTENSIONS)]:
            self.assertEqual(getattr(self.backend.nifits, ext.lower()), getattr(mock_existing_nifits, ext.lower()))

    def test_add_observation_data_with_new_attribute(self):
        mock_existing_nifits = MagicMock()
        mock_new_nifits = MagicMock()
        self.backend.nifits = mock_existing_nifits
        missing_extension = NIFITS_EXTENSIONS[STATIC_EXTENSIONS][0]

        # Simulate some extensions missing in the new nifits object
        for ext in NIFITS_EXTENSIONS[np.logical_not(STATIC_EXTENSIONS)]:
            if ext.lower() != missing_extension:
                setattr(mock_new_nifits, ext.lower(), MagicMock())
            setattr(mock_existing_nifits, ext.lower(), MagicMock())

        self.backend.add_observation_data(nifits_data=mock_new_nifits, force=False)

        for ext in NIFITS_EXTENSIONS[np.logical_not(STATIC_EXTENSIONS)]:
            if ext.lower() != missing_extension:
                self.assertEqual(getattr(self.backend.nifits, ext.lower()), getattr(mock_existing_nifits, ext.lower()))
            else:
                self.assertFalse(hasattr(self.backend.nifits, ext.lower()))


if __name__ == "__main__":
    unittest.main()
