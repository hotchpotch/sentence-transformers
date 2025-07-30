"""
Compatibility layer for sentence_transformers dependencies.
Contains minimal implementations of required utilities from sentence_transformers.
"""

import importlib
from typing import Any, List, Tuple, Optional, Dict
import logging
import re

logger = logging.getLogger(__name__)


def fullname(o) -> str:
    """
    Gives a full name (package_name.class_name) for a class / object in Python.
    Will be used to load the correct classes from JSON files.
    
    Args:
        o: The object for which to get the full name.
        
    Returns:
        str: The full name of the object.
    """
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + "." + o.__class__.__name__


def import_from_string(dotted_path: str) -> type:
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    
    Args:
        dotted_path (str): The dotted module path.
        
    Returns:
        Any: The attribute/class designated by the last name in the path.
        
    Raises:
        ImportError: If the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError:
        msg = f"{dotted_path} doesn't look like a module path"
        raise ImportError(msg)
    
    try:
        module = importlib.import_module(dotted_path)
    except Exception:
        module = importlib.import_module(module_path)
    
    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = f'Module "{module_path}" does not define a "{class_name}" attribute/class'
        raise ImportError(msg)


# Re-export text chunking utilities
try:
    from sentence_transformers.utils.text_chunking import MultilingualChunker
except ImportError:
    # If sentence_transformers is not available, provide a minimal implementation
    class MultilingualChunker:
        """
        Minimal multilingual text chunker implementation.
        Falls back to basic sentence splitting when sentence_transformers is not available.
        """
        
        def __init__(self):
            self.sentence_endings = re.compile(
                r'[.!?。！？।॥۔؟]\s*',
                re.UNICODE
            )
        
        def chunk_text(self, 
                      text: str, 
                      language: str = "auto",
                      max_length: Optional[int] = None,
                      min_length: Optional[int] = None,
                      **kwargs) -> List[Tuple[str, Tuple[int, int]]]:
            """
            Split text into sentences with position information.
            
            Args:
                text: Text to split
                language: Language code or "auto" for detection (ignored in minimal implementation)
                max_length: Maximum sentence length (characters)
                min_length: Minimum sentence length (characters)
                
            Returns:
                List of tuples (sentence, (start_pos, end_pos))
            """
            if not text or not text.strip():
                return []
            
            sentences = []
            last_end = 0
            
            for match in self.sentence_endings.finditer(text):
                end = match.end()
                sentence = text[last_end:end].strip()
                if sentence:
                    if min_length and len(sentence) < min_length:
                        continue
                    if max_length and len(sentence) > max_length:
                        sentence = sentence[:max_length]
                    sentences.append((sentence, (last_end, end)))
                last_end = end
            
            # Add remaining text if any
            if last_end < len(text):
                remaining = text[last_end:].strip()
                if remaining:
                    if not min_length or len(remaining) >= min_length:
                        if max_length and len(remaining) > max_length:
                            remaining = remaining[:max_length]
                        sentences.append((remaining, (last_end, len(text))))
            
            # If no sentences found, return the whole text
            if not sentences and text.strip():
                sentences = [(text.strip(), (0, len(text)))]
            
            return sentences


# CrossEncoder base class stub for compatibility
class CrossEncoder:
    """
    Minimal CrossEncoder stub for compatibility.
    Real CrossEncoder functionality should be imported from sentence_transformers.
    """
    def __init__(self):
        raise NotImplementedError(
            "This is a compatibility stub. Please install sentence_transformers "
            "for full CrossEncoder functionality."
        )