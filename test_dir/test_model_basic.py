"""
Tests for basic model functionality that doesn't require API access.
These tests focus on increasing coverage for the model classes.
"""

from unittest import TestCase, skipIf, mock
from unittest.mock import MagicMock, patch
import os
import sys
import json

# Check if running in CI
IN_CI = os.environ.get('CI', 'false').lower() == 'true'

# Import necessary modules safely with better error handling
IMPORTS_AVAILABLE = False
IMPORT_ERROR = None

try:
    # Set up mocks for external dependencies before importing model classes
    if 'google' not in sys.modules:
        mock_google = MagicMock()
        mock_google.generativeai = MagicMock()
        sys.modules['google'] = mock_google
        sys.modules['google.generativeai'] = mock_google.generativeai
    
    # Mock requests before importing
    if 'requests' not in sys.modules:
        mock_requests = MagicMock()
        sys.modules['requests'] = mock_requests
    
    # Now try to import the model classes
    from cli_code.models.base import AbstractModelAgent
    from cli_code.models.gemini import GeminiModelAgent
    from cli_code.models.ollama import OllamaModelAgent
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR = str(e)
    # Create dummy classes for type checking
    class AbstractModelAgent: pass
    class GeminiModelAgent: pass
    class OllamaModelAgent: pass

# Check if we should skip all tests - only skip if imports truly failed
# But in CI, we can still run tests with mocked modules
SHOULD_SKIP_TESTS = not IMPORTS_AVAILABLE and not IN_CI
SKIP_REASON = f"Required model imports not available: {IMPORT_ERROR}" if IMPORT_ERROR else "Required model imports not available"

@skipIf(SHOULD_SKIP_TESTS, SKIP_REASON)
class TestGeminiModelBasics(TestCase):
    """Test basic GeminiModelAgent functionality that doesn't require API calls."""
    
    def setUp(self):
        """Set up test environment."""
        # Create patches for external dependencies
        self.patch_configure = patch('google.generativeai.configure')
        self.patch_get_model = patch('google.generativeai.get_model')
        
        # Start patches
        self.mock_configure = self.patch_configure.start()
        self.mock_get_model = self.patch_get_model.start()
        
        # Set up default mock model
        self.mock_model = MagicMock()
        self.mock_get_model.return_value = self.mock_model
    
    def tearDown(self):
        """Clean up test environment."""
        # Stop patches
        self.patch_configure.stop()
        self.patch_get_model.stop()
    
    def test_gemini_init(self):
        """Test initialization of GeminiModelAgent."""
        agent = GeminiModelAgent("fake-api-key", "gemini-pro")
            
        # Verify API key was passed to configure
        self.mock_configure.assert_called_once_with(api_key="fake-api-key")
        
        # Check agent properties
        self.assertEqual(agent.model_name, "gemini-pro")
        self.assertEqual(agent.api_key, "fake-api-key")
        self.assertEqual(agent.history, [])
    
    def test_gemini_clear_history(self):
        """Test history clearing functionality."""
        agent = GeminiModelAgent("fake-api-key", "gemini-pro")
        
        # Add some fake history
        agent.history = [{"role": "user", "parts": ["test message"]}]
        
        # Clear history
        agent.clear_history()
        
        # Verify history is cleared
        self.assertEqual(agent.history, [])

    def test_gemini_add_system_prompt(self):
        """Test adding system prompt to history."""
        agent = GeminiModelAgent("fake-api-key", "gemini-pro")
        
        # Add system prompt
        agent.add_system_prompt("I am a helpful AI assistant")
        
        # Verify system prompt was added to history
        self.assertEqual(len(agent.history), 1)
        self.assertEqual(agent.history[0]["role"], "model")
        self.assertEqual(agent.history[0]["parts"][0]["text"], "I am a helpful AI assistant")
    
    def test_gemini_append_history(self):
        """Test appending to history."""
        agent = GeminiModelAgent("fake-api-key", "gemini-pro")
        
        # Append to history
        agent.append_to_history(role="user", content="Hello")
        agent.append_to_history(role="model", content="Hi there!")
        
        # Verify history entries
        self.assertEqual(len(agent.history), 2)
        self.assertEqual(agent.history[0]["role"], "user")
        self.assertEqual(agent.history[0]["parts"][0]["text"], "Hello")
        self.assertEqual(agent.history[1]["role"], "model")
        self.assertEqual(agent.history[1]["parts"][0]["text"], "Hi there!")
    
    def test_gemini_chat_generation_parameters(self):
        """Test chat generation parameters are properly set."""
        agent = GeminiModelAgent("fake-api-key", "gemini-pro")
        
        # Setup the mock model's generate_content to return a valid response
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Generated response"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = mock_content
        self.mock_model.generate_content.return_value = mock_response
        
        # Add some history before chat
        agent.add_system_prompt("System prompt")
        agent.append_to_history(role="user", content="Hello")
        
        # Call chat method with custom parameters
        response = agent.chat("What can you help me with?", temperature=0.2, max_tokens=1000)
        
        # Verify the model was called with correct parameters
        self.mock_model.generate_content.assert_called_once()
        args, kwargs = self.mock_model.generate_content.call_args
        
        # Check that history was included
        self.assertEqual(len(args[0]), 3)  # System prompt + user message + new query
        
        # Check generation parameters 
        self.assertIn('generation_config', kwargs)
        
        # Check response handling
        self.assertEqual(response, "Generated response")
    
    def test_gemini_parse_response(self):
        """Test parsing different response formats from the Gemini API."""
        agent = GeminiModelAgent("fake-api-key", "gemini-pro")
        
        # Mock normal response
        normal_response = MagicMock()
        normal_content = MagicMock()
        normal_content.text = "Normal response"
        normal_response.candidates = [MagicMock()]
        normal_response.candidates[0].content = normal_content
        
        # Mock empty response
        empty_response = MagicMock()
        empty_response.candidates = []
        
        # Mock response with finish reason not STOP
        blocked_response = MagicMock()
        blocked_response.candidates = [MagicMock()]
        blocked_candidate = blocked_response.candidates[0]
        blocked_candidate.content.text = "Blocked content"
        blocked_candidate.finish_reason = MagicMock()
        blocked_candidate.finish_reason.name = "SAFETY"
        
        # Test normal response parsing
        result = agent._parse_response(normal_response)
        self.assertEqual(result, "Normal response")
        
        # Test empty response parsing
        result = agent._parse_response(empty_response)
        self.assertEqual(result, "No response generated. Please try again.")
        
        # Test blocked response parsing
        result = agent._parse_response(blocked_response)
        self.assertEqual(result, "The response was blocked due to: SAFETY")
    
    def test_gemini_content_handling(self):
        """Test content handling for different input types."""
        agent = GeminiModelAgent("fake-api-key", "gemini-pro")
        
        # Test string content
        parts = agent._prepare_content("Hello world")
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["text"], "Hello world")
        
        # Test list content
        parts = agent._prepare_content(["Hello", "world"])
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["text"], "Hello")
        self.assertEqual(parts[1]["text"], "world")
        
        # Test already formatted content
        parts = agent._prepare_content([{"text": "Already formatted"}])
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["text"], "Already formatted")
        
        # Test empty content
        parts = agent._prepare_content("")
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["text"], "")


@skipIf(SHOULD_SKIP_TESTS, SKIP_REASON)
class TestOllamaModelBasics(TestCase):
    """Test basic OllamaModelAgent functionality that doesn't require API calls."""
    
    def setUp(self):
        """Set up test environment."""
        # Create patches for external dependencies
        self.patch_requests_post = patch('requests.post')
        
        # Start patches
        self.mock_post = self.patch_requests_post.start()
        
        # Setup default response
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Response from model"}}
        self.mock_post.return_value = mock_response
    
    def tearDown(self):
        """Clean up test environment."""
        # Stop patches
        self.patch_requests_post.stop()
    
    def test_ollama_init(self):
        """Test initialization of OllamaModelAgent."""
        agent = OllamaModelAgent("http://localhost:11434", "llama2")
        
        # Check agent properties
        self.assertEqual(agent.model_name, "llama2")
        self.assertEqual(agent.api_url, "http://localhost:11434")
        self.assertEqual(agent.history, [])
    
    def test_ollama_clear_history(self):
        """Test history clearing functionality."""
        agent = OllamaModelAgent("http://localhost:11434", "llama2")
        
        # Add some fake history
        agent.history = [{"role": "user", "content": "test message"}]
        
        # Clear history
        agent.clear_history()
        
        # Verify history is cleared
        self.assertEqual(agent.history, [])
    
    def test_ollama_add_system_prompt(self):
        """Test adding system prompt to history."""
        agent = OllamaModelAgent("http://localhost:11434", "llama2")
        
        # Add system prompt
        agent.add_system_prompt("I am a helpful AI assistant")
        
        # Verify system prompt was added to history
        self.assertEqual(len(agent.history), 1)
        self.assertEqual(agent.history[0]["role"], "system")
        self.assertEqual(agent.history[0]["content"], "I am a helpful AI assistant")
    
    def test_ollama_append_history(self):
        """Test appending to history."""
        agent = OllamaModelAgent("http://localhost:11434", "llama2")
        
        # Append to history
        agent.append_to_history(role="user", content="Hello")
        agent.append_to_history(role="assistant", content="Hi there!")
        
        # Verify history entries
        self.assertEqual(len(agent.history), 2)
        self.assertEqual(agent.history[0]["role"], "user")
        self.assertEqual(agent.history[0]["content"], "Hello")
        self.assertEqual(agent.history[1]["role"], "assistant")
        self.assertEqual(agent.history[1]["content"], "Hi there!")
    
    def test_ollama_prepare_chat_params(self):
        """Test preparing parameters for chat request."""
        agent = OllamaModelAgent("http://localhost:11434", "llama2")
        
        # Add history entries
        agent.add_system_prompt("System instructions")
        agent.append_to_history(role="user", content="Hello")
        
        # Prepare chat params and verify structure
        params = agent._prepare_chat_params()
        
        self.assertEqual(params["model"], "llama2")
        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][0]["role"], "system")
        self.assertEqual(params["messages"][0]["content"], "System instructions")
        self.assertEqual(params["messages"][1]["role"], "user")
        self.assertEqual(params["messages"][1]["content"], "Hello")
    
    def test_ollama_chat_with_parameters(self):
        """Test chat method with various parameters."""
        agent = OllamaModelAgent("http://localhost:11434", "llama2")
        
        # Add a system prompt
        agent.add_system_prompt("Be helpful")
        
        # Call chat with different parameters
        result = agent.chat("Hello", temperature=0.3, max_tokens=2000)
        
        # Verify the post request was called with correct parameters
        self.mock_post.assert_called_once()
        args, kwargs = self.mock_post.call_args
        
        # Check URL
        self.assertEqual(args[0], "http://localhost:11434/api/chat")
        
        # Check JSON payload
        json_data = kwargs.get('json', {})
        self.assertEqual(json_data["model"], "llama2")
        self.assertEqual(len(json_data["messages"]), 3)  # System + history + new message
        self.assertEqual(json_data["temperature"], 0.3)
        self.assertEqual(json_data["max_tokens"], 2000)
        
        # Verify the response was correctly processed
        self.assertEqual(result, "Response from model")
    
    def test_ollama_error_handling(self):
        """Test handling of various error cases."""
        agent = OllamaModelAgent("http://localhost:11434", "llama2")
        
        # Test connection error
        self.mock_post.side_effect = Exception("Connection failed")
        result = agent.chat("Hello")
        self.assertTrue("Error communicating with Ollama API" in result)
        
        # Test bad response
        self.mock_post.side_effect = None
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "Model not found"}
        self.mock_post.return_value = mock_response
        result = agent.chat("Hello")
        self.assertTrue("Error" in result)
        
        # Test missing content in response
        mock_response.json.return_value = {"message": {}}  # Missing content
        result = agent.chat("Hello")
        self.assertTrue("Unexpected response format" in result)
    
    def test_ollama_url_handling(self):
        """Test handling of different URL formats."""
        # Test with trailing slash
        agent = OllamaModelAgent("http://localhost:11434/", "llama2")
        self.assertEqual(agent.api_url, "http://localhost:11434")
        
        # Test without protocol
        agent = OllamaModelAgent("localhost:11434", "llama2")
        self.assertEqual(agent.api_url, "http://localhost:11434")
        
        # Test with https
        agent = OllamaModelAgent("https://ollama.example.com", "llama2")
        self.assertEqual(agent.api_url, "https://ollama.example.com") 