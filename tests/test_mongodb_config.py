"""Tests for selecting the configured MongoDB database."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from database.mongodb import (
    DATABASE_NAME,
    MongoConfigurationError,
    _get_database_name,
    get_mongo_database,
)


class MongoDatabaseConfigurationTests(unittest.TestCase):
    @patch.dict(os.environ, {"DISEASE_AKINATOR_MONGODB_DATABASE": "disease_akinator_staging"})
    def test_environment_variable_selects_database(self):
        with patch.dict(
            sys.modules,
            {"streamlit": SimpleNamespace(secrets={"mongodb": {"database": "ignored"}})},
        ):
            self.assertEqual(_get_database_name(), "disease_akinator_staging")

    @patch.dict(os.environ, {}, clear=True)
    def test_streamlit_secret_selects_database(self):
        fake_streamlit = SimpleNamespace(
            secrets={"mongodb": {"uri": "mongodb+srv://example.invalid", "database": "disease_akinator_staging"}}
        )
        with patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            self.assertEqual(_get_database_name(), "disease_akinator_staging")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_database_override_preserves_production_default(self):
        fake_streamlit = SimpleNamespace(secrets={"mongodb": {"uri": "mongodb+srv://example.invalid"}})
        with patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            self.assertEqual(_get_database_name(), DATABASE_NAME)
            self.assertEqual(DATABASE_NAME, "disease_akinator")

    @patch.dict(os.environ, {}, clear=True)
    def test_empty_configured_database_name_is_rejected(self):
        fake_streamlit = SimpleNamespace(secrets={"mongodb": {"database": "   "}})
        with patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            with self.assertRaises(MongoConfigurationError):
                _get_database_name()

    @patch("database.mongodb.get_mongo_client")
    @patch.dict(os.environ, {"DISEASE_AKINATOR_MONGODB_DATABASE": "disease_akinator_staging"})
    def test_get_mongo_database_uses_selected_database(self, mock_get_client):
        client = mock_get_client.return_value
        expected_database = object()
        client.__getitem__.return_value = expected_database

        result = get_mongo_database("mongodb+srv://example.invalid")

        self.assertIs(result, expected_database)
        client.__getitem__.assert_called_once_with("disease_akinator_staging")


if __name__ == "__main__":
    unittest.main(verbosity=2)
