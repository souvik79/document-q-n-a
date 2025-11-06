"""
Image processing module for extracting text and understanding images.

Supports:
- OCR text extraction (Tesseract, EasyOCR)
- Vision model analysis (LLaVA via Ollama)
- Multiple image formats (PNG, JPG, JPEG, BMP, TIFF, WEBP)
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import base64
import io
import os

from PIL import Image
import requests

from .config import OLLAMA_BASE_URL, VISION_PROVIDER, GOOGLE_API_KEY


class ImageProcessor:
    """Process images to extract text and generate descriptions."""

    def __init__(
        self,
        use_tesseract: bool = True,
        use_easyocr: bool = False,
        use_vision_model: bool = True,
        vision_model: str = "llava:7b",
        vision_provider: Optional[str] = None
    ):
        """
        Initialize image processor.

        Args:
            use_tesseract: Use Tesseract OCR for text extraction
            use_easyocr: Use EasyOCR for text extraction (slower but better for non-English)
            use_vision_model: Use vision model for image understanding
            vision_model: Vision model to use (llava:7b for local, ignored for gemini)
            vision_provider: 'local' or 'gemini' (default from config)
        """
        self.use_tesseract = use_tesseract
        self.use_easyocr = use_easyocr
        self.use_vision_model = use_vision_model
        self.vision_model = vision_model
        self.vision_provider = vision_provider or VISION_PROVIDER

        # Initialize OCR engines (lazy loading)
        self._tesseract_available = None
        self._easyocr_reader = None

        # Initialize cloud provider if needed
        self._gemini_provider = None
        if self.vision_provider == "gemini" and self.use_vision_model:
            if not GOOGLE_API_KEY:
                print("Warning: GOOGLE_API_KEY not set. Falling back to local vision model.")
                self.vision_provider = "local"
            else:
                from .cloud_providers import get_gemini_provider
                self._gemini_provider = get_gemini_provider()

    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        if self._tesseract_available is not None:
            return self._tesseract_available

        try:
            import pytesseract
            # Try to get version to verify it's installed
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            return True
        except Exception:
            self._tesseract_available = False
            return False

    def _get_easyocr_reader(self):
        """Get or initialize EasyOCR reader (lazy loading)."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(['en'])  # English only by default
            except Exception as e:
                print(f"Failed to initialize EasyOCR: {e}")
                return None
        return self._easyocr_reader

    def process_image(
        self,
        image_path: str,
        extract_text: bool = True,
        generate_description: bool = True
    ) -> Dict[str, Any]:
        """
        Process an image to extract text and/or generate description.

        Args:
            image_path: Path to image file
            extract_text: Whether to extract text using OCR
            generate_description: Whether to generate description using vision model

        Returns:
            Dictionary with extracted text, description, and metadata
        """
        try:
            # Load image
            image = Image.open(image_path)

            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            result = {
                "success": True,
                "image_path": str(image_path),
                "image_size": image.size,
                "image_mode": image.mode,
                "extracted_text": "",
                "description": "",
                "metadata": {}
            }

            # Extract text using OCR
            if extract_text:
                text_result = self._extract_text_ocr(image)
                result["extracted_text"] = text_result["text"]
                result["metadata"]["ocr_method"] = text_result["method"]
                result["metadata"]["ocr_confidence"] = text_result.get("confidence")

            # Generate description using vision model
            if generate_description and self.use_vision_model:
                description_result = self._generate_description(image_path)
                result["description"] = description_result["description"]
                result["metadata"]["vision_model"] = self.vision_model

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process image: {str(e)}",
                "image_path": str(image_path)
            }

    def _extract_text_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract text from image using OCR.

        Args:
            image: PIL Image object

        Returns:
            Dictionary with extracted text and method used
        """
        text = ""
        method = "none"
        confidence = None

        # Try Tesseract first (faster)
        if self.use_tesseract and self._check_tesseract():
            try:
                import pytesseract
                text = pytesseract.image_to_string(image)
                method = "tesseract"

                # Get confidence if possible
                try:
                    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                    if confidences:
                        confidence = sum(confidences) / len(confidences)
                except:
                    pass

            except Exception as e:
                print(f"Tesseract OCR failed: {e}")

        # Try EasyOCR if Tesseract failed or not available
        if not text and self.use_easyocr:
            try:
                reader = self._get_easyocr_reader()
                if reader:
                    # Convert PIL Image to numpy array
                    import numpy as np
                    image_array = np.array(image)

                    # Perform OCR
                    results = reader.readtext(image_array)

                    # Extract text
                    text = "\n".join([result[1] for result in results])
                    method = "easyocr"

                    # Calculate average confidence
                    if results:
                        confidence = sum([result[2] for result in results]) / len(results) * 100

            except Exception as e:
                print(f"EasyOCR failed: {e}")

        return {
            "text": text.strip(),
            "method": method,
            "confidence": confidence
        }

    def _generate_description(self, image_path: str) -> Dict[str, Any]:
        """
        Generate description of image using vision model.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with description
        """
        try:
            # Use Gemini if configured
            if self.vision_provider == "gemini" and self._gemini_provider:
                description = self._gemini_provider.describe_image(image_path)
                return {
                    "description": description.strip(),
                    "model": "gemini",
                    "provider": "gemini"
                }

            # Otherwise use local Ollama
            # Encode image to base64
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')

            # Prepare request for Ollama vision model
            url = f"{OLLAMA_BASE_URL}/api/generate"

            prompt = """Describe this image in detail. Include:
1. What objects, people, or text you see
2. The overall scene or context
3. Any notable details or features
4. If there are diagrams, charts, or technical content, describe their purpose

Be specific and detailed."""

            payload = {
                "model": self.vision_model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False
            }

            # Make request
            response = requests.post(url, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                description = result.get("response", "")
                return {
                    "description": description.strip(),
                    "model": self.vision_model,
                    "provider": "local"
                }
            else:
                return {
                    "description": "",
                    "error": f"Vision model request failed: {response.status_code}"
                }

        except Exception as e:
            print(f"Vision model description failed: {e}")
            return {
                "description": "",
                "error": str(e)
            }

    def process_and_create_document_text(
        self,
        image_path: str,
        title: Optional[str] = None
    ) -> str:
        """
        Process image and create formatted text for document storage.

        Args:
            image_path: Path to image file
            title: Optional title for the image

        Returns:
            Formatted text combining OCR and description
        """
        result = self.process_image(image_path)

        if not result["success"]:
            return f"[Image processing failed: {result.get('error', 'Unknown error')}]"

        # Build document text
        parts = []

        # Add title
        if title:
            parts.append(f"# Image: {title}")
        else:
            parts.append(f"# Image: {Path(image_path).name}")

        # Add description from vision model
        if result["description"]:
            parts.append("\n## Visual Description")
            parts.append(result["description"])

        # Add extracted text from OCR
        if result["extracted_text"]:
            parts.append("\n## Extracted Text (OCR)")
            parts.append(result["extracted_text"])

        # Add metadata
        parts.append("\n## Image Metadata")
        parts.append(f"- Size: {result['image_size'][0]}x{result['image_size'][1]}")
        parts.append(f"- Format: {result['image_mode']}")

        if result["metadata"].get("ocr_method"):
            parts.append(f"- OCR Method: {result['metadata']['ocr_method']}")

        if result["metadata"].get("vision_model"):
            parts.append(f"- Vision Model: {result['metadata']['vision_model']}")

        return "\n".join(parts)

    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """
        Check if file is a supported image format.

        Args:
            file_path: Path to file

        Returns:
            True if file is a supported image
        """
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
        return Path(file_path).suffix.lower() in image_extensions

    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported image formats."""
        return ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp', 'gif']


# Convenience functions
def process_image_file(
    image_path: str,
    use_vision: bool = True,
    vision_model: str = "llava:7b"
) -> Dict[str, Any]:
    """
    Process a single image file.

    Args:
        image_path: Path to image
        use_vision: Whether to use vision model
        vision_model: Which vision model to use

    Returns:
        Processing result dictionary
    """
    processor = ImageProcessor(
        use_tesseract=True,
        use_easyocr=False,  # Tesseract is faster
        use_vision_model=use_vision,
        vision_model=vision_model
    )

    return processor.process_image(image_path)


def create_image_document(
    image_path: str,
    title: Optional[str] = None,
    use_vision: bool = True
) -> str:
    """
    Create document text from image for vector storage.

    Args:
        image_path: Path to image
        title: Optional title
        use_vision: Whether to use vision model

    Returns:
        Formatted document text
    """
    processor = ImageProcessor(use_vision_model=use_vision)
    return processor.process_and_create_document_text(image_path, title)
