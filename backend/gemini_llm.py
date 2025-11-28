# backend/gemini_llm.py
from google import genai
import os
from dotenv import load_dotenv
from typing import Any, Optional, List

load_dotenv()

class GeminiLLM:
    """
    Enhanced Gemini wrapper fully compatible with CrewAI's LLM interface.
    """
    
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.7):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self._model_name = model
    
    def call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        Main method CrewAI uses to invoke the LLM.
        """
        try:
            # Extract temperature if provided in kwargs
            temp = kwargs.get('temperature', self.temperature)
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    'temperature': temp,
                    'max_output_tokens': kwargs.get('max_tokens', 2048),
                }
            )
            return response.text
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return f"Error: {str(e)}"
    
    def __call__(self, prompt: str, **kwargs) -> str:
        return self.call(prompt, **kwargs)
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    # CrewAI may check these attributes
    @property
    def supports_function_calling(self) -> bool:
        return False  # Set True if using Gemini function calling
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Batch generation support."""
        return [self.call(prompt, **kwargs) for prompt in prompts]