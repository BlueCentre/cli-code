import logging
import unittest
from unittest.mock import MagicMock, patch

from cli_code.utils.log_config import get_logger


class TestLogConfig(unittest.TestCase):
    """Tests for the log_config module"""

    @patch("logging.getLogger")
    @patch("logging.basicConfig")
    def test_get_logger(self, mock_basic_config, mock_get_logger):
        """Test that get_logger returns a configured logger"""
        mock_logger = MagicMock()
        # Configure the mock logger to have no handlers
        mock_logger.hasHandlers.return_value = False
        mock_get_logger.return_value = mock_logger

        # Call the function
        logger = get_logger("test_logger")

        # Verify that logging.getLogger was called with the right name
        mock_get_logger.assert_called_once_with("test_logger")

        # Verify that basicConfig was called
        mock_basic_config.assert_called_once()

        # Verify the returned logger is what we expect
        self.assertEqual(logger, mock_logger)


if __name__ == "__main__":
    unittest.main()
