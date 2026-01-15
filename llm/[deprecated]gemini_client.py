import os
import time
import re
from config import GEMINI_API_KEY


class GeminiClient:
    """
    A robust client for the Gemini API that handles Rate Limits (429) automatically.
    """
    def __init__(self, model_name="gemini-2.0-flash-exp"):
        if genai is None:
            raise ImportError("The 'google-genai' package is missing. Run: pip install google-genai")
        
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = model_name

    def generate_text(self, prompt: str) -> str:
        """
        Generates text and automatically waits if the API says we are rate limited.
        """
        max_retries = 5
        base_delay = 2

        for i in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                return response.text
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for Rate Limit (429)
                if "429" in error_str or "resource_exhausted" in error_str:
                    print(f"\n[AI] Rate limit hit. Cooling down...")
                    
                    # Try to extract the requested wait time from the error message
                    # Google often sends: "please retry in 49.68s"
                    wait_match = re.search(r"retry in (\d+(\.\d+)?)s", error_str)
                    
                    if wait_match:
                        wait_time = float(wait_match.group(1)) + 1 # Add 1s buffer
                        print(f"[AI] Google asked to wait {wait_time:.1f}s. Sleeping...")
                        time.sleep(wait_time)
                    else:
                        # Fallback exponential backoff
                        wait_time = base_delay * (2 ** i)
                        print(f"[AI] Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    
                    # Continue to next loop iteration to retry
                    continue
                
                # If it's another error, fail gracefully
                return f"AI Error: {e}"

        return "Failed to generate response after multiple retries."