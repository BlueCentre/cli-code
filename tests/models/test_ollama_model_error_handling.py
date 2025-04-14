import pytest
import json
from unittest.mock import MagicMock, patch, call
import sys
from pathlib import Path

# Ensure src is in the path for imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from cli_code.models.ollama import OllamaModel, MAX_OLLAMA_ITERATIONS


class TestOllamaModelErrorHandling:
    """Tests for error handling in the OllamaModel class."""
    
    @pytest.fixture
    def mock_console(self):
        console = MagicMock()
        console.print = MagicMock()
        console.status = MagicMock()
        # Make status return a context manager
        status_cm = MagicMock()
        console.status.return_value = status_cm
        status_cm.__enter__ = MagicMock(return_value=None)
        status_cm.__exit__ = MagicMock(return_value=None)
        return console
    
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.chat.completions.create = MagicMock()
        client.models.list = MagicMock()
        return client
    
    @pytest.fixture
    def mock_questionary(self):
        questionary = MagicMock()
        confirm = MagicMock()
        questionary.confirm.return_value = confirm
        confirm.ask = MagicMock(return_value=True)
        return questionary
    
    def test_generate_without_client(self, mock_console):
        """Test generate method when the client is not initialized."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model.client = None  # Explicitly set client to None
        
        # Execute
        result = model.generate("test prompt")
        
        # Assert
        assert "Error: Ollama client not initialized" in result
        mock_console.print.assert_not_called()
    
    def test_generate_without_model_name(self, mock_console):
        """Test generate method when no model name is specified."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console)
        model.model_name = None  # Explicitly set model_name to None
        model.client = MagicMock()  # Add a mock client
        
        # Execute
        result = model.generate("test prompt")
        
        # Assert
        assert "Error: No Ollama model name configured" in result
        mock_console.print.assert_not_called()
    
    @patch('cli_code.models.ollama.get_tool')
    def test_generate_with_invalid_tool_call(self, mock_get_tool, mock_console, mock_client):
        """Test generate method with invalid JSON in tool arguments."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model.client = mock_client
        model.add_to_history = MagicMock()  # Mock history management
        
        # Create mock response with tool call that has invalid JSON
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [
            MagicMock(
                function=MagicMock(
                    name="test_tool",
                    arguments='invalid json'
                ),
                id="test_id"
            )
        ]
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=mock_message,
            finish_reason="tool_calls"
        )]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Execute
        with patch('cli_code.models.ollama.json.loads', side_effect=json.JSONDecodeError("Expecting value", "", 0)):
            result = model.generate("test prompt")
        
        # Assert
        assert "reached maximum iterations" in result
        # Verify the log message was recorded (we'd need to patch logging.error and check call args)
    
    @patch('cli_code.models.ollama.get_tool')
    @patch('cli_code.models.ollama.SENSITIVE_TOOLS', ['edit'])
    @patch('cli_code.models.ollama.questionary')
    def test_generate_with_user_rejection(self, mock_questionary, mock_get_tool, mock_console, mock_client):
        """Test generate method when user rejects a sensitive tool execution."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model.client = mock_client
        
        # Create mock response with a sensitive tool call
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [
            MagicMock(
                function=MagicMock(
                    name="edit",
                    arguments='{"file_path": "test.txt", "content": "test content"}'
                ),
                id="test_id"
            )
        ]
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=mock_message,
            finish_reason="tool_calls"
        )]
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Make user reject the confirmation
        confirm_mock = MagicMock()
        confirm_mock.ask.return_value = False
        mock_questionary.confirm.return_value = confirm_mock
        
        # Mock the tool function
        mock_tool = MagicMock()
        mock_get_tool.return_value = mock_tool
        
        # Execute
        result = model.generate("test prompt")
        
        # Assert
        assert "rejected" in result or "maximum iterations" in result
    
    def test_list_models_error(self, mock_console, mock_client):
        """Test list_models method when an error occurs."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model.client = mock_client
        
        # Make client.models.list raise an exception
        mock_client.models.list.side_effect = Exception("Test error")
        
        # Execute
        result = model.list_models()
        
        # Assert
        assert result is None
        mock_console.print.assert_called()
        assert any("Error contacting Ollama endpoint" in str(call_args) for call_args in mock_console.print.call_args_list)
    
    def test_add_to_history_invalid_message(self, mock_console):
        """Test add_to_history with an invalid message."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model._manage_ollama_context = MagicMock()  # Mock to avoid side effects
        original_history_len = len(model.history)
        
        # Add invalid message (not a dict)
        model.add_to_history("not a dict")
        
        # Assert
        # System message will be there, but invalid message should not be added
        assert len(model.history) == original_history_len
        model._manage_ollama_context.assert_not_called()
    
    def test_manage_ollama_context_empty_history(self, mock_console):
        """Test _manage_ollama_context with empty history."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        original_history = model.history.copy()  # Save the original which includes system prompt
        
        # Execute
        model._manage_ollama_context()
        
        # Assert
        assert model.history == original_history  # Should remain the same with system prompt
    
    @patch('cli_code.models.ollama.count_tokens')
    def test_manage_ollama_context_serialization_error(self, mock_count_tokens, mock_console):
        """Test _manage_ollama_context when serialization fails."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        # Add a message that will cause serialization error (contains an unserializable object)
        model.history = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": MagicMock()}  # Unserializable
        ]
        
        # Make count_tokens return a low value to avoid truncation
        mock_count_tokens.return_value = 10
        
        # Execute
        with patch('cli_code.models.ollama.json.dumps', side_effect=TypeError("Object is not JSON serializable")):
            model._manage_ollama_context()
        
        # Assert - history should remain unchanged
        assert len(model.history) == 3
    
    def test_generate_max_iterations(self, mock_console, mock_client):
        """Test generate method when max iterations is reached."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model.client = mock_client
        model._prepare_openai_tools = MagicMock(return_value=[{"type": "function", "function": {"name": "test_tool"}}])
        
        # Create mock response with tool call
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [
            MagicMock(
                function=MagicMock(
                    name="test_tool",
                    arguments='{"param1": "value1"}'
                ),
                id="test_id"
            )
        ]
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=mock_message,
            finish_reason="tool_calls"
        )]
        
        # Mock the client to always return a tool call (which would lead to an infinite loop without max iterations)
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock get_tool to return a tool that always succeeds
        tool_mock = MagicMock()
        tool_mock.execute.return_value = "Tool result"
        
        # Execute - this should hit the max iterations
        with patch('cli_code.models.ollama.get_tool', return_value=tool_mock):
            with patch('cli_code.models.ollama.MAX_OLLAMA_ITERATIONS', 2):  # Lower max iterations for test
                result = model.generate("test prompt")
        
        # Assert
        assert "(Agent reached maximum iterations)" in result
    
    def test_prepare_openai_tools_without_available_tools(self, mock_console):
        """Test _prepare_openai_tools when AVAILABLE_TOOLS is empty."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        
        # Execute
        with patch('cli_code.models.ollama.AVAILABLE_TOOLS', {}):
            result = model._prepare_openai_tools()
        
        # Assert
        assert result is None
    
    def test_prepare_openai_tools_conversion_error(self, mock_console):
        """Test _prepare_openai_tools when conversion fails."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        
        # Mock tool instance
        tool_mock = MagicMock()
        tool_declaration = MagicMock()
        tool_declaration.name = "test_tool"
        tool_declaration.description = "Test tool"
        tool_declaration.parameters = MagicMock()
        tool_declaration.parameters._pb = MagicMock()
        tool_mock.get_function_declaration.return_value = tool_declaration
        
        # Execute - with a mocked error during conversion
        with patch('cli_code.models.ollama.AVAILABLE_TOOLS', {"test_tool": tool_mock}):
            with patch('cli_code.models.ollama.MessageToDict', side_effect=Exception("Conversion error")):
                result = model._prepare_openai_tools()
        
        # Assert
        assert result is None or len(result) == 0  # Should be empty list or None 
    
    @patch('cli_code.models.ollama.log') # Patch log
    def test_generate_with_connection_error(self, mock_log, mock_console, mock_client):
        """Test generate method when a connection error occurs during API call."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model.client = mock_client
        
        # Simulate a connection error (e.g., RequestError from httpx)
        # Assuming the ollama client might raise something like requests.exceptions.ConnectionError or httpx.RequestError
        # We'll use a generic Exception and check the message for now.
        # If a specific exception class is known, use it instead.
        connection_err = Exception("Failed to connect") 
        mock_client.chat.completions.create.side_effect = connection_err
        
        # Execute
        result = model.generate("test prompt")
        
        # Assert
        assert "Error connecting to Ollama" in result or "Failed to connect" in result
        mock_log.error.assert_called() # Check that an error was logged
        # Check specific log message if needed
        log_call_args, _ = mock_log.error.call_args
        assert "Error during Ollama agent iteration" in log_call_args[0]
        
    @patch('cli_code.models.ollama.log') # Patch log
    def test_generate_with_timeout_error(self, mock_log, mock_console, mock_client):
        """Test generate method when a timeout error occurs during API call."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model.client = mock_client
        
        # Simulate a timeout error 
        # Use a generic Exception, check message. Replace if specific exception is known (e.g., httpx.TimeoutException)
        timeout_err = Exception("Request timed out") 
        mock_client.chat.completions.create.side_effect = timeout_err
        
        # Execute
        result = model.generate("test prompt")
        
        # Assert
        assert "Error connecting to Ollama" in result or "timed out" in result
        mock_log.error.assert_called()
        log_call_args, _ = mock_log.error.call_args
        assert "Error during Ollama agent iteration" in log_call_args[0] 
    
    @patch('cli_code.models.ollama.log') # Patch log
    def test_generate_with_server_error(self, mock_log, mock_console, mock_client):
        """Test generate method when a server error occurs during API call."""
        # Setup
        model = OllamaModel("http://localhost:11434", mock_console, "llama3")
        model.client = mock_client
        
        # Simulate a server error (e.g., HTTP 500)
        # Use a generic Exception, check message. Replace if specific exception is known (e.g., ollama.APIError?)
        server_err = Exception("Internal Server Error") 
        mock_client.chat.completions.create.side_effect = server_err
        
        # Execute
        result = model.generate("test prompt")
        
        # Assert
        # Check for a generic error message indicating an unexpected issue
        assert "Error interacting with Ollama" in result # Check for the actual prefix
        assert "Internal Server Error" in result # Check the specific error message is included
        mock_log.error.assert_called()
        log_call_args, _ = mock_log.error.call_args
        assert "Error during Ollama agent iteration" in log_call_args[0] 