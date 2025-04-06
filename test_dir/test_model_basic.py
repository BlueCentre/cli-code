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