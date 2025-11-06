"""
Cloud provider wrappers for LLM and Vision APIs.

Supports:
- Google Gemini (free tier: 60 requests/minute)
- Easy toggle between local (Ollama) and cloud providers
"""

from typing import Dict, Any, Optional, List
import base64
from pathlib import Path

from .config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    GEMINI_VISION_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
)


class GeminiProvider:
    """Google Gemini API wrapper for text and vision tasks."""

    def __init__(self):
        """Initialize Gemini API client."""
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not set in .env file. "
                "Get your free API key from: https://makersuite.google.com/app/apikey"
            )

        try:
            import google.generativeai as genai
            self.genai = genai
            self.genai.configure(api_key=GOOGLE_API_KEY)
            self._text_model = None
            self._vision_model = None
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )

    def _get_text_model(self):
        """Get or initialize text model (lazy loading)."""
        if self._text_model is None:
            self._text_model = self.genai.GenerativeModel(GEMINI_MODEL)
        return self._text_model

    def _get_vision_model(self):
        """Get or initialize vision model (lazy loading)."""
        if self._vision_model is None:
            self._vision_model = self.genai.GenerativeModel(GEMINI_VISION_MODEL)
        return self._vision_model

    def generate_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Generate text response from prompt.

        Args:
            prompt: Input prompt
            temperature: Temperature for generation (0-1)

        Returns:
            Generated text
        """
        try:
            model = self._get_text_model()

            generation_config = {
                "temperature": temperature if temperature is not None else TEMPERATURE,
                "max_output_tokens": MAX_TOKENS,
            }

            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Check if response has valid candidates
            if not response.candidates:
                raise RuntimeError(
                    "Response was blocked by safety filters. "
                    "This may happen with certain types of content. "
                    "Try rephrasing your question or adjusting the request."
                )

            # Check finish reason
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason

            # finish_reason values:
            # 0 = FINISH_REASON_UNSPECIFIED
            # 1 = STOP (successful completion)
            # 2 = MAX_TOKENS
            # 3 = SAFETY
            # 4 = RECITATION
            # 5 = OTHER

            if finish_reason == 3:  # SAFETY
                raise RuntimeError(
                    "Response was blocked due to safety concerns. "
                    "Please rephrase your request."
                )
            elif finish_reason == 2:  # MAX_TOKENS
                # Still return partial response if available
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    return candidate.content.parts[0].text
                raise RuntimeError(
                    "Response was truncated due to token limit. "
                    "Try asking for a shorter response."
                )
            elif finish_reason == 4:  # RECITATION
                raise RuntimeError(
                    "Response was blocked due to recitation concerns. "
                    "Please rephrase your request."
                )
            elif finish_reason not in [0, 1]:  # Not UNSPECIFIED or STOP
                raise RuntimeError(
                    f"Response generation stopped unexpectedly (reason: {finish_reason}). "
                    "Please try again or rephrase your request."
                )

            # Check if response has parts
            if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                raise RuntimeError(
                    "Response does not contain valid content. "
                    "Please try rephrasing your question."
                )

            return response.text

        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")

    def generate_with_image(
        self,
        image_path: str,
        prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text response from image and prompt.

        Args:
            image_path: Path to image file
            prompt: Input prompt
            temperature: Temperature for generation

        Returns:
            Generated text description
        """
        try:
            from PIL import Image as PILImage

            model = self._get_vision_model()

            # Load image
            image = PILImage.open(image_path)

            generation_config = {
                "temperature": temperature if temperature is not None else TEMPERATURE,
                "max_output_tokens": MAX_TOKENS,
            }

            # Generate response
            response = model.generate_content(
                [prompt, image],
                generation_config=generation_config
            )

            # Check if response has valid candidates
            if not response.candidates:
                raise RuntimeError(
                    "Response was blocked by safety filters. "
                    "This may happen with certain images or prompts. "
                    "Try with a different image or rephrase your request."
                )

            # Check finish reason
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason

            if finish_reason == 3:  # SAFETY
                raise RuntimeError(
                    "Response was blocked due to safety concerns. "
                    "Please try a different image or prompt."
                )
            elif finish_reason == 2:  # MAX_TOKENS
                # Still return partial response if available
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    return candidate.content.parts[0].text
                raise RuntimeError(
                    "Response was truncated due to token limit. "
                    "Try asking for a shorter response."
                )
            elif finish_reason == 4:  # RECITATION
                raise RuntimeError(
                    "Response was blocked due to recitation concerns. "
                    "Please rephrase your request."
                )
            elif finish_reason not in [0, 1]:  # Not UNSPECIFIED or STOP
                raise RuntimeError(
                    f"Response generation stopped unexpectedly (reason: {finish_reason}). "
                    "Please try again or rephrase your request."
                )

            # Check if response has parts
            if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                raise RuntimeError(
                    "Response does not contain valid content. "
                    "Please try rephrasing your question."
                )

            return response.text

        except Exception as e:
            raise RuntimeError(f"Gemini Vision API error: {str(e)}")

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using Gemini's OCR capabilities.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text
        """
        prompt = """Extract all text from this image.
        Return only the extracted text, without any additional commentary or description.
        If there is no text in the image, return an empty string."""

        return self.generate_with_image(image_path, prompt)

    def describe_image(self, image_path: str) -> str:
        """
        Generate detailed description of image.

        Args:
            image_path: Path to image file

        Returns:
            Image description
        """
        prompt = """Describe this image in detail. Include:
1. What objects, people, or text you see
2. The overall scene or context
3. Any notable details or features
4. If there are diagrams, charts, or technical content, describe their purpose

Be specific and detailed."""

        return self.generate_with_image(image_path, prompt)

    def process_image_complete(self, image_path: str) -> Dict[str, str]:
        """
        Process image to extract both text and description.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with extracted_text and description
        """
        try:
            # Use a single prompt to get both text extraction and description
            prompt = """Analyze this image and provide:

1. **Extracted Text**: Any text visible in the image (OCR). If no text, write "No text found."

2. **Visual Description**: Describe what you see - objects, people, scenes, diagrams, charts, or any visual elements.

Format your response exactly as:
EXTRACTED TEXT:
[text here or "No text found"]

VISUAL DESCRIPTION:
[description here]"""

            response = self.generate_with_image(image_path, prompt)

            # Parse response
            extracted_text = ""
            description = ""

            if "EXTRACTED TEXT:" in response and "VISUAL DESCRIPTION:" in response:
                parts = response.split("VISUAL DESCRIPTION:")
                text_part = parts[0].replace("EXTRACTED TEXT:", "").strip()
                desc_part = parts[1].strip() if len(parts) > 1 else ""

                extracted_text = text_part if text_part and text_part != "No text found" else ""
                description = desc_part
            else:
                # Fallback: treat entire response as description
                description = response

            return {
                "extracted_text": extracted_text,
                "description": description
            }

        except Exception as e:
            print(f"Gemini image processing error: {e}")
            return {
                "extracted_text": "",
                "description": f"Error processing image: {str(e)}"
            }


# Singleton instance
_gemini_instance = None


def get_gemini_provider() -> GeminiProvider:
    """Get or create Gemini provider instance."""
    global _gemini_instance
    if _gemini_instance is None:
        _gemini_instance = GeminiProvider()
    return _gemini_instance


# Convenience functions
def gemini_generate(prompt: str, temperature: Optional[float] = None) -> str:
    """Generate text using Gemini."""
    provider = get_gemini_provider()
    return provider.generate_text(prompt, temperature)


def gemini_describe_image(image_path: str) -> str:
    """Describe image using Gemini Vision."""
    provider = get_gemini_provider()
    return provider.describe_image(image_path)


def gemini_extract_text(image_path: str) -> str:
    """Extract text from image using Gemini."""
    provider = get_gemini_provider()
    return provider.extract_text_from_image(image_path)


def gemini_process_image(image_path: str) -> Dict[str, str]:
    """Process image completely with Gemini."""
    provider = get_gemini_provider()
    return provider.process_image_complete(image_path)
