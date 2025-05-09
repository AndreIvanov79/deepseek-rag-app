import os
import pytesseract
import cv2
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CodeImageProcessor:
    """Utility class for extracting code from images."""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize the CodeImageProcessor.
        
        Args:
            tesseract_path: Path to tesseract executable. If None, will use system default.
        """
        # Configure tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Check if tesseract is installed and accessible
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.warning(
                f"Tesseract OCR not available or not properly configured: {e}. "
                f"Image-to-code functionality will be limited."
            )
    
    def extract_code_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract code from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing the extracted text and metadata
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "error": "Failed to read image", "text": ""}
            
            # Preprocess the image for better OCR results
            processed_image = self._preprocess_image(image)
            
            # Perform OCR
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(
                processed_image, 
                config=custom_config
            )
            
            # Try to determine language/code type from syntax
            lang_type = self._detect_code_language(text)
            
            return {
                "success": True,
                "text": text,
                "detected_language": lang_type,
                "char_count": len(text),
                "line_count": text.count('\n') + 1
            }
            
        except Exception as e:
            logger.error(f"Error extracting code from image: {e}")
            return {"success": False, "error": str(e), "text": ""}
    
    def extract_code_from_pil_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Extract code from a PIL Image object.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Dictionary containing the extracted text and metadata
        """
        try:
            # Convert PIL to OpenCV format
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR (OpenCV uses BGR)
            if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
                open_cv_image = open_cv_image[:, :, ::-1].copy()
            
            # Preprocess the image for better OCR results
            processed_image = self._preprocess_image(open_cv_image)
            
            # Perform OCR
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(
                processed_image, 
                config=custom_config
            )
            
            # Try to determine language/code type from syntax
            lang_type = self._detect_code_language(text)
            
            return {
                "success": True,
                "text": text,
                "detected_language": lang_type,
                "char_count": len(text),
                "line_count": text.count('\n') + 1
            }
            
        except Exception as e:
            logger.error(f"Error extracting code from PIL image: {e}")
            return {"success": False, "error": str(e), "text": ""}
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image for better OCR results.
        
        Args:
            image: OpenCV image in BGR format
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding - improves OCR for code screenshots
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean noise
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Edge enhancement
        edges = cv2.Canny(opening, 50, 150)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        result = cv2.bitwise_or(opening, dilated)
        
        return result
    
    def _detect_code_language(self, text: str) -> str:
        """Try to detect the programming language from code text.
        
        Args:
            text: Extracted text from OCR
            
        Returns:
            Detected language or 'Unknown'
        """
        text = text.lower()
        
        # Simple keyword-based detection
        if any(keyword in text for keyword in ['def ', 'import ', 'class ', 'if __name__', '->']):
            return 'python'
        elif any(keyword in text for keyword in ['func ', 'package ', 'import (', 'struct ']):
            return 'golang'
        elif any(keyword in text for keyword in ['public class', 'private ', 'protected ', '@Override', 'System.out']):
            return 'java'
        elif any(keyword in text for keyword in ['#include', 'int main(', 'std::', '->']):
            return 'c++'
        elif any(keyword in text for keyword in ['function(', 'const ', 'let ', 'var ', '=>', 'console.log']):
            return 'javascript'
        elif any(keyword in text for keyword in ['<?php', '->', 'function ', '$']):
            return 'php'
        elif any(keyword in text for keyword in ['<!DOCTYPE', '<html>', '<div', '<script']):
            return 'html'
        elif any(keyword in text for keyword in ['{', '}', ';', 'for(', 'if(', 'else']):
            return 'code' # Generic code
        
        return 'text' # Probably not code