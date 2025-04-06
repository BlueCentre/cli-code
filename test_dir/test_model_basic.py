"""
Tests for basic model functionality that doesn't require API access.
These tests focus on increasing coverage for the model classes.
"""

from unittest import TestCase, skipIf
from unittest.mock import MagicMock, patch

# Import necessary modules safely
try:
    from cli_code.models.base import AbstractModelAgent
    from cli_code.models.gemini import GeminiModelAgent
    from cli_code.models.ollama import OllamaModelAgent
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    # Create dummy classes for type checking
    class AbstractModelAgent: pass
    class GeminiModelAgent: pass
    class OllamaModelAgent: pass


# Skip all tests if imports aren't available
@skipIf(not IMPORTS_AVAILABLE, "Required model imports not available")
class TestGeminiModelBasics(TestCase):
    """Test basic GeminiModelAgent functionality that doesn't require API calls."""
    
    def test_gemini_init(self):
        """Test initialization of GeminiModelAgent."""
        with patch('google.generativeai.configure') as mock_configure:
            agent = GeminiModelAgent("fake-api-key", "gemini-pro")
            
            # Verify API key was passed to configure
            mock_configure.assert_called_once_with(api_key="fake-api-key")
            
            # Check agent properties
            self.assertEqual(agent.model_name, "gemini-pro")
            self.assertEqual(agent.api_key, "fake-api-key")
            self.assertEqual(agent.history, [])
    
    def test_gemini_clear_history(self):
        """Test history clearing functionality."""
        with patch('google.generativeai.configure'):
            agent = GeminiModelAgent("fake-api-key", "gemini-pro")
            
            # Add some fake history
            agent.history = [{"role": "user", "parts": ["test message"]}]
            
            # Clear history
            agent.clear_history()
            
            # Verify history is cleared
            self.assertEqual(agent.history, [])

    @patch('google.generativeai.configure')
    @patch('google.generativeai.get_model')
    def test_gemini_add_system_prompt(self, mock_get_model, mock_configure):
        """Test adding system prompt to history."""
        agent = GeminiModelAgent("fake-api-key", "gemini-pro")
        
        # Add system prompt
        agent.add_system_prompt("I am a helpful AI assistant")
        
        # Verify system prompt was added to history
        self.assertEqual(len(agent.history), 1)
        self.assertEqual(agent.history[0]["role"], "model")
        self.assertEqual(agent.history[0]["parts"][0]["text"], "I am a helpful AI assistant")
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.get_model')
    def test_gemini_append_history(self, mock_get_model, mock_configure):
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
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.get_model')
    def test_gemini_chat_generation_parameters(self, mock_get_model, mock_configure):
        """Test chat generation parameters are properly set."""
        agent = GeminiModelAgent("fake-api-key", "gemini-pro")
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        
        # Setup the mock model's generate_content to return a valid response
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Generated response"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = mock_content
        mock_model.generate_content.return_value = mock_response
        
        # Add some history before chat
        agent.add_system_prompt("System prompt")
        agent.append_to_history(role="user", content="Hello")
        
        # Call chat method with custom parameters
        response = agent.chat("What can you help me with?", temperature=0.2, max_tokens=1000)
        
        # Verify the model was called with correct parameters
        mock_model.generate_content.assert_called_once()
        args, kwargs = mock_model.generate_content.call_args
        
        # Check that history was included
        self.assertEqual(len(args[0]), 3)  # System prompt + user message + new query
        
        # Check that generation parameters were passed correctly
        self.assertEqual(kwargs.get('generation_config').temperature, 0.2)
        self.assertEqual(kwargs.get('generation_config').max_output_tokens, 1000)
        
        # Check response handling
        self.assertEqual(response, "Generated response")
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.get_model')
    def test_gemini_parse_response(self, mock_get_model, mock_configure):
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
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.get_model')
    def test_gemini_content_handling(self, mock_get_model, mock_configure):
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


@skipIf(not IMPORTS_AVAILABLE, "Required model imports not available")
class TestOllamaModelBasics(TestCase):
    """Test basic OllamaModelAgent functionality that doesn't require API calls."""
    
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
    
    @patch('requests.post')
    def test_ollama_prepare_chat_params(self, mock_post):
        """Test preparing parameters for chat request."""
        agent = OllamaModelAgent("http://localhost:11434", "llama2")
        
        # Add history entries
        agent.add_system_prompt("System instructions")
        agent.append_to_history(role="user", content="Hello")
        
        # Mock the response for the post request
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Response from model"}}
        mock_post.return_value = mock_response
        
        # Prepare chat params and verify structure
        params = agent._prepare_chat_params()
        
        self.assertEqual(params["model"], "llama2")
        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][0]["role"], "system")
        self.assertEqual(params["messages"][0]["content"], "System instructions")
        self.assertEqual(params["messages"][1]["role"], "user")
        self.assertEqual(params["messages"][1]["content"], "Hello")
    
    @patch('requests.post')
    def test_ollama_chat_with_parameters(self, mock_post):
        """Test chat method with various parameters."""
        agent = OllamaModelAgent("http://localhost:11434", "llama2")
        
        # Mock the response for the post request
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Response from model"}}
        mock_post.return_value = mock_response
        
        # Add a system prompt
        agent.add_system_prompt("Be helpful")
        
        # Call chat with different parameters
        result = agent.chat("Hello", temperature=0.3, max_tokens=2000)
        
        # Verify the post request was called with correct parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
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
    
    @patch('requests.post')
    def test_ollama_error_handling(self, mock_post):
        """Test handling of various error cases."""
        agent = OllamaModelAgent("http://localhost:11434", "llama2")
        
        # Test connection error
        mock_post.side_effect = Exception("Connection failed")
        result = agent.chat("Hello")
        self.assertTrue("Error communicating with Ollama API" in result)
        
        # Test bad response
        mock_post.side_effect = None
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "Model not found"}
        mock_post.return_value = mock_response
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