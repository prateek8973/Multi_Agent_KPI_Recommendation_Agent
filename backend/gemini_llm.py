# backend/gemini_llm.py - Enhanced Production Version
import os
import time
import logging
from typing import Any, Optional, List, Dict
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiLLM:
    """
    Enhanced Gemini LLM wrapper fully compatible with CrewAI's LLM interface.
    
    Features:
    - Automatic retry with exponential backoff
    - Token usage tracking
    - Error handling and logging
    - Rate limiting protection
    - Response validation
    - Multiple model support
    """
    
    # Available Gemini models
    MODELS = {
        "flash": "gemini-2.5-flash",
        "flash-thinking": "gemini-2.5-flash",
        "pro": "gemini-2.5-flash",
        "legacy": "gemini-2.5-flash"
    }
    
    def __init__(
        self, 
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.7,
        max_retries: int = 3,
        timeout: int = 60,
        max_output_tokens: int = 8192
    ):
        """
        Initialize Gemini LLM wrapper.
        
        Args:
            model: Model identifier (use MODELS dict or custom model name)
            temperature: Sampling temperature (0.0 - 2.0)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            max_output_tokens: Maximum tokens in response
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        
        # Initialize client
        self.client = genai.Client(api_key=api_key)
        
        # Model configuration
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        
        # Track usage
        self.total_tokens_used = 0
        self.total_requests = 0
        self.failed_requests = 0
        
        logger.info(f"Initialized GeminiLLM with model: {model}, temperature: {temperature}")
    
    def call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        Main method CrewAI uses to invoke the LLM.
        
        Args:
            prompt: The input prompt/message
            stop: Stop sequences (not fully supported by Gemini)
            **kwargs: Additional parameters
            
        Returns:
            str: Generated text response
        """
        return self._generate_with_retry(prompt, **kwargs)
    
    def _generate_with_retry(self, prompt: str, **kwargs) -> str:
        """
        Generate content with automatic retry and exponential backoff.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated response text
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self._generate(prompt, **kwargs)
                self.total_requests += 1
                return response
                
            except Exception as e:
                last_error = e
                self.failed_requests += 1
                
                # Calculate backoff delay
                if attempt < self.max_retries - 1:
                    delay = (2 ** attempt) + (time.time() % 1)  # Exponential backoff with jitter
                    logger.warning(
                        f"Generation failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed: {str(e)}")
        
        # All retries exhausted
        error_msg = f"Failed after {self.max_retries} attempts. Last error: {str(last_error)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    def _generate(self, prompt: str, **kwargs) -> str:
        """
        Core generation method.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            str: Generated text
        """
        # Extract parameters from kwargs or use defaults
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_output_tokens)
        top_p = kwargs.get('top_p', 0.95)
        top_k = kwargs.get('top_k', 40)
        
        # Build generation config
        config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            response_modalities=["TEXT"],
        )
        
        # Add system instruction if provided
        system_instruction = kwargs.get('system_instruction')
        if system_instruction:
            config.system_instruction = system_instruction
        
        logger.debug(f"Generating with model={self.model}, temp={temperature}, max_tokens={max_tokens}")
        
        # Generate content
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        
        # Extract text from response
        if hasattr(response, 'text'):
            result_text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            result_text = response.candidates[0].content.parts[0].text
        else:
            raise ValueError("Unable to extract text from response")
        
        # Track token usage if available
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            tokens_used = getattr(usage, 'total_token_count', 0)
            self.total_tokens_used += tokens_used
            logger.debug(f"Tokens used: {tokens_used} (total: {self.total_tokens_used})")
        
        return result_text
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Make the object callable for compatibility.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        return self.call(prompt, **kwargs)
    
    @property
    def model_name(self) -> str:
        """Return the model name for CrewAI compatibility."""
        return self.model
    
    @property
    def supports_function_calling(self) -> bool:
        """Indicate if this model supports function calling."""
        # Gemini models support function calling, but not implemented here
        return False
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Batch generation support.
        
        Args:
            prompts: List of prompts to generate from
            **kwargs: Generation parameters
            
        Returns:
            List[str]: List of generated responses
        """
        logger.info(f"Batch generating {len(prompts)} prompts")
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.debug(f"Processing batch item {i + 1}/{len(prompts)}")
            result = self.call(prompt, **kwargs)
            results.append(result)
        
        return results
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dict with usage metrics
        """
        success_rate = (
            (self.total_requests - self.failed_requests) / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(success_rate, 2),
            "total_tokens_used": self.total_tokens_used,
            "model": self.model,
            "temperature": self.temperature
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.total_requests = 0
        self.failed_requests = 0
        logger.info("Usage statistics reset")
    
    @classmethod
    def create_for_kpi_system(cls, agent_type: str = "general") -> 'GeminiLLM':
        """
        Factory method to create optimized LLM instances for different agent types.
        
        Args:
            agent_type: Type of agent ('data', 'strategy', 'kpi', 'compute', 'insight', 'general')
            
        Returns:
            GeminiLLM: Configured instance
        """
        configs = {
            "data": {
                "temperature": 0.1,  # Very deterministic for data profiling
                "max_output_tokens": 2048,
            },
            "strategy": {
                "temperature": 0.3,  # Somewhat creative for strategy
                "max_output_tokens": 2048,
            },
            "kpi": {
                "temperature": 0.2,  # Low for accurate formula generation
                "max_output_tokens": 4096,
            },
            "compute": {
                "temperature": 0.0,  # Completely deterministic for computation
                "max_output_tokens": 2048,
            },
            "insight": {
                "temperature": 0.5,  # More creative for insights
                "max_output_tokens": 4096,
            },
            "general": {
                "temperature": 0.3,
                "max_output_tokens": 4096,
            }
        }
        
        config = configs.get(agent_type, configs["general"])
        logger.info(f"Creating LLM for agent type: {agent_type} with config: {config}")
        
        return cls(**config)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def test_gemini_connection() -> bool:
    """
    Test if Gemini API is accessible and working.
    
    Returns:
        bool: True if connection successful
    """
    try:
        llm = GeminiLLM(temperature=0.1)
        response = llm.call("Say 'Hello' if you can read this.")
        
        if "hello" in response.lower():
            logger.info("✅ Gemini API connection successful")
            print(f"✅ Gemini API connection successful")
            print(f"Response: {response}")
            return True
        else:
            logger.warning(f"⚠️ Unexpected response: {response}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Gemini API connection failed: {str(e)}")
        print(f"❌ Gemini API connection failed: {str(e)}")
        return False


def get_available_models() -> List[str]:
    """
    Get list of available Gemini models.
    
    Returns:
        List[str]: Model identifiers
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        
        # List available models
        models = client.models.list()
        model_names = [model.name for model in models if 'gemini' in model.name.lower()]
        
        logger.info(f"Found {len(model_names)} Gemini models")
        return model_names
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        return list(GeminiLLM.MODELS.values())


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Gemini LLM Wrapper - Test Suite")
    print("=" * 60)
    
    # Test 1: Basic connection
    print("\n[Test 1] Testing API connection...")
    test_gemini_connection()
    
    # Test 2: Create LLM and generate
    print("\n[Test 2] Testing basic generation...")
    try:
        llm = GeminiLLM(temperature=0.7)
        response = llm.call("Explain what a KPI is in one sentence.")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Batch generation
    print("\n[Test 3] Testing batch generation...")
    try:
        llm = GeminiLLM(temperature=0.5)
        prompts = [
            "What is revenue?",
            "What is profit margin?",
            "What is customer lifetime value?"
        ]
        responses = llm.generate(prompts)
        for i, resp in enumerate(responses, 1):
            print(f"{i}. {resp[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Agent-specific configuration
    print("\n[Test 4] Testing agent-specific configs...")
    try:
        data_llm = GeminiLLM.create_for_kpi_system("data")
        kpi_llm = GeminiLLM.create_for_kpi_system("kpi")
        insight_llm = GeminiLLM.create_for_kpi_system("insight")
        
        print(f"Data Agent - Temperature: {data_llm.temperature}")
        print(f"KPI Agent - Temperature: {kpi_llm.temperature}")
        print(f"Insight Agent - Temperature: {insight_llm.temperature}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Usage statistics
    print("\n[Test 5] Testing usage statistics...")
    try:
        llm = GeminiLLM(temperature=0.3)
        llm.call("Test prompt 1")
        llm.call("Test prompt 2")
        
        stats = llm.get_usage_stats()
        print(f"Usage Stats: {stats}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 6: Error handling and retry
    print("\n[Test 6] Testing error handling...")
    try:
        llm = GeminiLLM(temperature=0.5, max_retries=2)
        # This should work normally
        response = llm.call("Say 'test'")
        print(f"Response: {response}")
        print(f"Failed requests: {llm.failed_requests}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 7: List available models
    print("\n[Test 7] Listing available models...")
    try:
        models = get_available_models()
        print(f"Found {len(models)} models")
        for model in models[:5]:  # Show first 5
            print(f"  - {model}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)