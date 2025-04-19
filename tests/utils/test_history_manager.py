"""
Tests for the HistoryManager utility class.
"""

import unittest
from unittest.mock import call, patch

from cli_code.utils.history_manager import MAX_HISTORY_TURNS, HistoryManager


class TestHistoryManager(unittest.TestCase):
    """Test cases for the HistoryManager class."""

    def setUp(self):
        """Set up a fresh HistoryManager before each test."""
        self.history_manager = HistoryManager()

    def test_init_default(self):
        """Test initialization with default values."""
        manager = HistoryManager()
        self.assertEqual(manager.max_turns, MAX_HISTORY_TURNS)
        self.assertEqual(manager.history, [])

    def test_init_custom_max_turns(self):
        """Test initialization with custom max_turns value."""
        custom_max = 10
        manager = HistoryManager(max_turns=custom_max)
        self.assertEqual(manager.max_turns, custom_max)
        self.assertEqual(manager.history, [])

    def test_add_entry(self):
        """Test adding entries to the history."""
        entry = {"role": "user", "content": "Test message"}
        self.history_manager.add_entry(entry)
        self.assertEqual(len(self.history_manager.history), 1)
        self.assertEqual(self.history_manager.history[0], entry)

    def test_get_history(self):
        """Test retrieving the history."""
        entry1 = {"role": "user", "content": "Test message 1"}
        entry2 = {"role": "assistant", "content": "Test message 2"}
        self.history_manager.add_entry(entry1)
        self.history_manager.add_entry(entry2)

        history = self.history_manager.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], entry1)
        self.assertEqual(history[1], entry2)

    def test_truncate(self):
        """Test history truncation when it exceeds max_turns."""
        # Set a small max_turns for testing
        self.history_manager.max_turns = 2

        # Add more entries than the limit (2 * max_turns = 4)
        for i in range(10):
            self.history_manager.add_entry({"role": "user", "content": f"Message {i}"})

        # History should be truncated to last 4 entries
        self.assertEqual(len(self.history_manager.history), 4)
        self.assertEqual(self.history_manager.history[0]["content"], "Message 6")
        self.assertEqual(self.history_manager.history[3]["content"], "Message 9")

    def test_clear(self):
        """Test clearing the history."""
        # Add some entries
        for i in range(3):
            self.history_manager.add_entry({"role": "user", "content": f"Message {i}"})

        # Verify entries were added
        self.assertEqual(len(self.history_manager.history), 3)

        # Clear the history
        self.history_manager.clear()

        # Verify history is empty
        self.assertEqual(len(self.history_manager.history), 0)

    @patch("builtins.print")
    def test_debug_prints(self, mock_print):
        """Test debug print statements are called."""
        # Test init print
        HistoryManager(max_turns=15)
        mock_print.assert_called_with("[Debug] HistoryManager initialized with max_turns=15")

        # Test add_entry print
        self.history_manager.add_entry({"role": "user", "content": "Test"})
        mock_print.assert_called_with("[Debug] Added entry, history length: 1")

        # Test clear print
        self.history_manager.clear()
        mock_print.assert_called_with("[Debug] History cleared.")

        # Test truncate print
        # Set small max_turns and add more entries to trigger truncation
        self.history_manager.max_turns = 1
        # We need to add enough entries to trigger truncation, which happens at > max_turns * 2
        # Add 3 entries to ensure we go over the threshold of 2 (max_turns=1 * 2 = 2)
        for i in range(3):
            self.history_manager.add_entry({"role": "user", "content": f"Message {i}"})

        # Get all the calls to mock_print
        truncation_calls = [
            entry for entry in mock_print.call_args_list if entry[0][0].startswith("[Debug] Truncating history")
        ]

        # Ensure there was at least one truncation message
        self.assertTrue(
            any("Truncating history" in entry[0][0] for entry in mock_print.call_args_list),
            "No truncation debug message was logged",
        )


if __name__ == "__main__":
    unittest.main()
