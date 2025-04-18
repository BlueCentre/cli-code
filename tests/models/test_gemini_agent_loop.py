"""
Tests for the refactored agent loop functionality in the GeminiModel class.
This file focuses on testing the _execute_agent_loop, _process_agent_iteration, and _process_candidate_response methods.
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from google.api_core.exceptions import ResourceExhausted

from src.cli_code.models.gemini import GeminiModel

# Test constants
FAKE_API_KEY = "test-api-key"
TEST_MODEL_NAME = "test-model"
FALLBACK_MODEL = "gemini-2.0-flash"


@pytest.fixture
def mock_console():
    """Provides a mocked Console object with a functional status context manager."""
    mock_console = MagicMock()
    mock_status_obj = MagicMock()
    mock_status_obj.update = MagicMock()
    mock_console.status.return_value.__enter__.return_value = mock_status_obj
    mock_console.status.return_value.__exit__.return_value = None
    return mock_console


@pytest.fixture
def gemini_instance(mock_console):
    """Provides a GeminiModel instance with mocked dependencies."""
    with (
        patch("src.cli_code.models.gemini.genai.configure"),
        patch("src.cli_code.models.gemini.genai.GenerativeModel"),
        patch("src.cli_code.models.gemini.GeminiModel._create_tool_definitions", return_value=None),
        patch("src.cli_code.models.gemini.GeminiModel._create_system_prompt", return_value="Test Prompt"),
    ):
        # Create the instance
        model = GeminiModel(api_key=FAKE_API_KEY, console=mock_console, model_name=TEST_MODEL_NAME)

        # Mock the add_to_history method
        model.add_to_history = MagicMock()

        # Mock _manage_context_window
        model._manage_context_window = MagicMock()

        # Replace history list with a mock that behaves like a list
        model.history = MagicMock()
        model.history.__bool__.return_value = True  # history is not empty
        model.history.__getitem__.side_effect = (
            lambda i: {"role": "model", "parts": [{"text": "test response"}]} if i == -1 else None
        )

        # Set up the model property
        model.model = MagicMock()
        model.current_model_name = TEST_MODEL_NAME

        # Mock status object for tests
        model.status = MagicMock()

        return model


# Tests for _execute_agent_loop


def test_execute_agent_loop_normal_completion(gemini_instance):
    """Test agent loop with normal task completion."""
    # Set up mocks for method calls
    with patch.object(gemini_instance, "_process_agent_iteration") as mock_process:
        mock_process.return_value = ("task_completed", None)

        # Set up necessary values for the method call
        iteration_count = 0
        task_completed = False
        final_summary = "Final summary"
        last_text_response = ""

        # Mock _handle_loop_completion
        with patch.object(gemini_instance, "_handle_loop_completion") as mock_handle_completion:
            mock_handle_completion.return_value = "Success: Task completed"

            # Execute the loop
            result = gemini_instance._execute_agent_loop(
                iteration_count, task_completed, final_summary, last_text_response
            )

            # Verify expectations
            mock_process.assert_called_once()
            mock_handle_completion.assert_called_once_with(True, None, 1)
            assert result == "Success: Task completed"


def test_execute_agent_loop_error_result(gemini_instance):
    """Test agent loop with an error result from iteration."""
    # Set up mocks for method calls
    with patch.object(gemini_instance, "_process_agent_iteration") as mock_process:
        mock_process.return_value = ("error", "Error: something went wrong")

        # Set up necessary values for the method call
        iteration_count = 0
        task_completed = False
        final_summary = "Final summary"
        last_text_response = ""

        # Execute the loop
        result = gemini_instance._execute_agent_loop(iteration_count, task_completed, final_summary, last_text_response)

        # Verify expectations
        mock_process.assert_called_once()
        assert result == "Error: something went wrong"


def test_execute_agent_loop_complete_result(gemini_instance):
    """Test agent loop with a complete result from iteration."""
    # Set up mocks for method calls
    with patch.object(gemini_instance, "_process_agent_iteration") as mock_process:
        mock_process.return_value = ("complete", "Task completed successfully")

        # Set up necessary values for the method call
        iteration_count = 0
        task_completed = False
        final_summary = "Final summary"
        last_text_response = ""

        # Execute the loop
        result = gemini_instance._execute_agent_loop(iteration_count, task_completed, final_summary, last_text_response)

        # Verify expectations
        mock_process.assert_called_once()
        assert result == "Task completed successfully"


def test_execute_agent_loop_continue_result(gemini_instance):
    """Test agent loop with a continue result that should loop once and then complete."""
    # Set up mocks for method calls
    with patch.object(gemini_instance, "_process_agent_iteration") as mock_process:
        # First call returns continue, second call returns complete
        mock_process.side_effect = [
            ("continue", "User rejected operation"),
            ("complete", "Task completed with different approach"),
        ]

        # Set up necessary values for the method call
        iteration_count = 0
        task_completed = False
        final_summary = "Final summary"
        last_text_response = ""

        # Execute the loop
        result = gemini_instance._execute_agent_loop(iteration_count, task_completed, final_summary, last_text_response)

        # Verify expectations
        assert mock_process.call_count == 2
        assert result == "Task completed with different approach"


def test_execute_agent_loop_with_user_rejection(gemini_instance):
    """Test agent loop with user rejection as the final outcome."""
    # Set up mocks for method calls
    with patch.object(gemini_instance, "_process_agent_iteration") as mock_process:
        mock_process.return_value = ("task_completed", None)

        # Set up necessary values for the method call
        iteration_count = 0
        task_completed = False
        final_summary = "Final summary"
        last_text_response = "User rejected the proposed operation on file.txt"

        # Mock _handle_loop_completion (shouldn't be called due to user rejection)
        with patch.object(gemini_instance, "_handle_loop_completion") as mock_handle_completion:
            # Execute the loop
            result = gemini_instance._execute_agent_loop(
                iteration_count, task_completed, final_summary, last_text_response
            )

            # Verify expectations
            mock_process.assert_called_once()
            assert result == "User rejected the proposed operation on file.txt"
            mock_handle_completion.assert_not_called()


# Tests for _process_agent_iteration


def test_process_agent_iteration_with_empty_history(gemini_instance):
    """Test handling empty history during agent iteration."""
    # Make history empty
    gemini_instance.history.__bool__.return_value = False

    # Set up a mock status
    mock_status = MagicMock()

    # Call the method
    result = gemini_instance._process_agent_iteration(mock_status, "")

    # Verify expectations
    assert result[0] == "error"
    assert "Error: Agent history is empty" in result[1]


def test_process_agent_iteration_llm_response_empty_candidates(gemini_instance):
    """Test handling empty candidates in LLM response."""
    # Mock _get_llm_response to return response with empty candidates
    with patch.object(gemini_instance, "_get_llm_response") as mock_get_response:
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_get_response.return_value = mock_response

        # Mock _handle_empty_response
        with patch.object(gemini_instance, "_handle_empty_response") as mock_handle_empty:
            mock_handle_empty.return_value = "Error: No candidates in response"

            # Call the method
            mock_status = MagicMock()
            result = gemini_instance._process_agent_iteration(mock_status, "")

            # Verify expectations
            mock_get_response.assert_called_once()
            mock_handle_empty.assert_called_once_with(mock_response)
            assert result == ("error", "Error: No candidates in response")


def test_process_agent_iteration_valid_response(gemini_instance):
    """Test processing a valid LLM response."""
    # Mock _get_llm_response to return valid response
    with patch.object(gemini_instance, "_get_llm_response") as mock_get_response:
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_get_response.return_value = mock_response

        # Mock _process_candidate_response
        with patch.object(gemini_instance, "_process_candidate_response") as mock_process_candidate:
            mock_process_candidate.return_value = ("complete", "Task completed successfully")

            # Call the method
            mock_status = MagicMock()
            result = gemini_instance._process_agent_iteration(mock_status, "")

            # Verify expectations
            mock_get_response.assert_called_once()
            mock_process_candidate.assert_called_once_with(mock_candidate, mock_status)
            assert result == ("complete", "Task completed successfully")


def test_process_agent_iteration_exception(gemini_instance):
    """Test handling exceptions during agent iteration."""
    # Mock _get_llm_response to raise an exception
    with patch.object(gemini_instance, "_get_llm_response") as mock_get_response:
        mock_get_response.side_effect = Exception("Test exception")

        # Mock _handle_agent_loop_exception
        with patch.object(gemini_instance, "_handle_agent_loop_exception") as mock_handle_exception:
            mock_handle_exception.return_value = "Error: Test exception handled"

            # Call the method
            mock_status = MagicMock()
            result = gemini_instance._process_agent_iteration(mock_status, "last response")

            # Verify expectations
            mock_get_response.assert_called_once()
            assert result == ("error", "Error: Test exception handled")


def test_process_agent_iteration_exception_no_result(gemini_instance):
    """Test handling exceptions during agent iteration with no result from handler."""
    # Mock _get_llm_response to raise an exception
    with patch.object(gemini_instance, "_get_llm_response") as mock_get_response:
        mock_get_response.side_effect = Exception("Test exception")

        # Mock _handle_agent_loop_exception to return None
        with patch.object(gemini_instance, "_handle_agent_loop_exception") as mock_handle_exception:
            mock_handle_exception.return_value = None

            # Call the method
            mock_status = MagicMock()
            last_text = "last response"
            result = gemini_instance._process_agent_iteration(mock_status, last_text)

            # Verify expectations
            mock_get_response.assert_called_once()
            assert result == ("continue", last_text)


# Tests for _process_candidate_response


def test_process_candidate_response_stop_reason_with_text(gemini_instance):
    """Test processing a response candidate with STOP reason and text."""
    # Mock _check_for_stop_reason to return True
    with patch.object(gemini_instance, "_check_for_stop_reason") as mock_check_stop:
        mock_check_stop.return_value = True

        # Mock _extract_final_text to return text
        with patch.object(gemini_instance, "_extract_final_text") as mock_extract:
            mock_extract.return_value = "Final result text"

            # Create test candidate
            mock_candidate = MagicMock()
            mock_status = MagicMock()

            # Call the method
            result = gemini_instance._process_candidate_response(mock_candidate, mock_status)

            # Verify expectations
            mock_check_stop.assert_called_once_with(mock_candidate, mock_status)
            mock_extract.assert_called_once_with(mock_candidate)
            assert result == ("complete", "Final result text")


def test_process_candidate_response_stop_reason_no_text(gemini_instance):
    """Test processing a response candidate with STOP reason but no text."""
    # Mock _check_for_stop_reason to return True
    with patch.object(gemini_instance, "_check_for_stop_reason") as mock_check_stop:
        mock_check_stop.return_value = True

        # Mock _extract_final_text to return empty text
        with patch.object(gemini_instance, "_extract_final_text") as mock_extract:
            mock_extract.return_value = "   "  # Empty after strip()

            # Mock _process_response_content
            with patch.object(gemini_instance, "_process_response_content") as mock_process:
                mock_process.return_value = "Processed content result"

                # Create test candidate
                mock_candidate = MagicMock()
                mock_status = MagicMock()

                # Call the method
                result = gemini_instance._process_candidate_response(mock_candidate, mock_status)

                # Verify expectations
                mock_check_stop.assert_called_once_with(mock_candidate, mock_status)
                mock_extract.assert_called_once_with(mock_candidate)
                mock_process.assert_called_once_with(mock_candidate, mock_status)
                assert result == ("complete", "Processed content result")


def test_process_candidate_response_with_rejection(gemini_instance):
    """Test processing a response with user rejection."""
    # Mock _check_for_stop_reason to return False
    with patch.object(gemini_instance, "_check_for_stop_reason") as mock_check_stop:
        mock_check_stop.return_value = False

        # Mock _process_response_content to return a rejection message
        with patch.object(gemini_instance, "_process_response_content") as mock_process:
            rejection_message = "User rejected the proposed operation on file.txt"
            mock_process.return_value = rejection_message

            # Create test candidate
            mock_candidate = MagicMock()
            mock_status = MagicMock()

            # Call the method
            result = gemini_instance._process_candidate_response(mock_candidate, mock_status)

            # Verify expectations
            mock_check_stop.assert_called_once_with(mock_candidate, mock_status)
            mock_process.assert_called_once_with(mock_candidate, mock_status)
            assert result == ("continue", rejection_message)


def test_process_candidate_response_with_completion(gemini_instance):
    """Test processing a response with a completion result."""
    # Mock _check_for_stop_reason to return False
    with patch.object(gemini_instance, "_check_for_stop_reason") as mock_check_stop:
        mock_check_stop.return_value = False

        # Mock _process_response_content to return a completion message
        with patch.object(gemini_instance, "_process_response_content") as mock_process:
            completion_message = "Task completed successfully"
            mock_process.return_value = completion_message

            # Create test candidate
            mock_candidate = MagicMock()
            mock_status = MagicMock()

            # Call the method
            result = gemini_instance._process_candidate_response(mock_candidate, mock_status)

            # Verify expectations
            mock_check_stop.assert_called_once_with(mock_candidate, mock_status)
            mock_process.assert_called_once_with(mock_candidate, mock_status)
            assert result == ("complete", completion_message)


def test_process_candidate_response_no_result(gemini_instance):
    """Test processing a response with no immediate result."""
    # Mock _check_for_stop_reason to return False
    with patch.object(gemini_instance, "_check_for_stop_reason") as mock_check_stop:
        mock_check_stop.return_value = False

        # Mock _process_response_content to return None
        with patch.object(gemini_instance, "_process_response_content") as mock_process:
            mock_process.return_value = None

            # Create test candidate
            mock_candidate = MagicMock()
            mock_status = MagicMock()

            # Call the method
            result = gemini_instance._process_candidate_response(mock_candidate, mock_status)

            # Verify expectations
            mock_check_stop.assert_called_once_with(mock_candidate, mock_status)
            mock_process.assert_called_once_with(mock_candidate, mock_status)
            assert result[0] == "continue"


def test_execute_agent_loop_max_iterations(gemini_instance):
    """Test agent loop with maximum iterations."""
    # Set up mocks for method calls
    with patch.object(gemini_instance, "_process_agent_iteration") as mock_process:
        mock_process.return_value = ("task_completed", None)

        # Set up necessary values for the method call
        iteration_count = 0
        task_completed = False
        final_summary = "Final summary"
        last_text_response = ""

        # Mock _handle_loop_completion
        with patch.object(gemini_instance, "_handle_loop_completion") as mock_handle_completion:
            mock_handle_completion.return_value = "Success: Task completed"

            # Execute the loop
            result = gemini_instance._execute_agent_loop(
                iteration_count, task_completed, final_summary, last_text_response
            )

            # Verify expectations
            mock_process.assert_called_once()
            mock_handle_completion.assert_called_once_with(True, None, 1)
            assert result == "Success: Task completed"


def test_process_candidate_response_user_rejected(gemini_instance):
    """Test processing a response containing a user rejection message."""
    # Mock _check_for_stop_reason to return False
    with patch.object(gemini_instance, "_check_for_stop_reason") as mock_check_stop:
        mock_check_stop.return_value = False

        # Mock _process_response_content to return a rejection message
        with patch.object(gemini_instance, "_process_response_content") as mock_process:
            rejection_message = "User rejected the proposed operation on file.txt"
            mock_process.return_value = rejection_message

            # Create test candidate
            mock_candidate = MagicMock()
            mock_status = MagicMock()

            # Call the method
            result = gemini_instance._process_candidate_response(mock_candidate, mock_status)

            # Verify expectations
            mock_check_stop.assert_called_once_with(mock_candidate, mock_status)
            mock_process.assert_called_once_with(mock_candidate, mock_status)
            assert result == ("continue", rejection_message)
