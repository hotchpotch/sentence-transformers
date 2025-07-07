"""
Text chunking utilities for Provence (Query-dependent Text Pruning).
Provides language-specific sentence segmentation with position tracking.
"""

import re
from typing import List, Tuple, Optional, Dict, Any
import langdetect
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Base class for text chunkers."""
    
    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Split text into chunks (sentences) with position information.
        
        Returns:
            List of tuples (chunk_text, (start_pos, end_pos))
        """
        pass


class DefaultChunker(BaseChunker):
    """Default regex-based chunker for most languages."""
    
    def __init__(self):
        # Common sentence-ending punctuation patterns
        self.sentence_endings = re.compile(
            r'[.!?。！？।॥۔؟]\s*',
            re.UNICODE
        )
    
    def chunk(self, text: str, **kwargs) -> List[Tuple[str, Tuple[int, int]]]:
        """Split text using regex patterns."""
        sentences = []
        last_end = 0
        
        for match in self.sentence_endings.finditer(text):
            end = match.end()
            sentence = text[last_end:end].strip()
            if sentence:
                sentences.append((sentence, (last_end, end)))
            last_end = end
        
        # Add remaining text if any
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sentences.append((remaining, (last_end, len(text))))
        
        # If no sentences found, return the whole text
        if not sentences and text.strip():
            sentences = [(text.strip(), (0, len(text)))]
        
        return sentences


class EnglishChunker(BaseChunker):
    """English-specific chunker using NLTK punkt tokenizer."""
    
    def __init__(self):
        self._tokenizer = None
        
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                import nltk
                try:
                    self._tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                except LookupError:
                    logger.info("Downloading NLTK punkt tokenizer...")
                    nltk.download('punkt', quiet=True)
                    self._tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            except Exception as e:
                logger.warning(f"Failed to load NLTK tokenizer: {e}. Using fallback.")
                return None
        return self._tokenizer
        
    def chunk(self, text: str, **kwargs) -> List[Tuple[str, Tuple[int, int]]]:
        """Use NLTK punkt tokenizer for English."""
        if self.tokenizer:
            try:
                spans = list(self.tokenizer.span_tokenize(text))
                return [(text[start:end], (start, end)) for start, end in spans]
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}. Using fallback.")
        
        # Fallback to default chunker
        return DefaultChunker().chunk(text)


class JapaneseChunker(BaseChunker):
    """Japanese-specific chunker using Bunkai."""
    
    def __init__(self):
        self._bunkai = None
        
    @property
    def bunkai(self):
        if self._bunkai is None:
            try:
                from bunkai import Bunkai
                self._bunkai = Bunkai()
            except ImportError:
                logger.warning("Bunkai not available. Using fallback for Japanese.")
                return None
        return self._bunkai
        
    def chunk(self, text: str, **kwargs) -> List[Tuple[str, Tuple[int, int]]]:
        """Use Bunkai for Japanese sentence segmentation."""
        if self.bunkai:
            try:
                sentences = list(self.bunkai(text))
                result = []
                pos = 0
                
                for sent in sentences:
                    start = text.find(sent, pos)
                    if start != -1:
                        end = start + len(sent)
                        result.append((sent, (start, end)))
                        pos = end
                    else:
                        # If exact match not found, append with approximate position
                        result.append((sent, (pos, pos + len(sent))))
                        pos += len(sent)
                
                return result
            except Exception as e:
                logger.warning(f"Bunkai chunking failed: {e}. Using fallback.")
        
        # Fallback: split by Japanese sentence endings
        pattern = r'([^。！？\n]+[。！？]?)'
        sentences = []
        for match in re.finditer(pattern, text):
            sent = match.group(0).strip()
            if sent:
                sentences.append((sent, match.span()))
        
        return sentences if sentences else [(text.strip(), (0, len(text)))]


class ChineseChunker(BaseChunker):
    """Chinese-specific chunker."""
    
    def chunk(self, text: str, **kwargs) -> List[Tuple[str, Tuple[int, int]]]:
        """Split Chinese text by punctuation."""
        # Chinese sentence endings
        pattern = r'[。！？；]\s*'
        sentences = []
        last_end = 0
        
        for match in re.finditer(pattern, text):
            end = match.end()
            sentence = text[last_end:end].strip()
            if sentence:
                sentences.append((sentence, (last_end, end)))
            last_end = end
        
        # Add remaining text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sentences.append((remaining, (last_end, len(text))))
        
        return sentences if sentences else [(text.strip(), (0, len(text)))]


class MultilingualChunker:
    """
    Multilingual text chunker with language detection and specialized chunkers.
    Based on multilingual_chunkers.py but adapted for Sentence Transformers.
    """
    
    def __init__(self):
        self._chunkers: Dict[str, BaseChunker] = {}
        self._default_chunker = DefaultChunker()
        
    def _get_chunker(self, language: str) -> BaseChunker:
        """Get or create chunker for a specific language."""
        if language not in self._chunkers:
            if language == "ja":
                self._chunkers[language] = JapaneseChunker()
            elif language == "en":
                self._chunkers[language] = EnglishChunker()
            elif language == "zh":
                self._chunkers[language] = ChineseChunker()
            else:
                # Use default chunker for other languages
                self._chunkers[language] = self._default_chunker
        
        return self._chunkers[language]
    
    def _detect_language(self, text: str) -> str:
        """Detect language of the text."""
        try:
            lang = langdetect.detect(text)
            return lang
        except Exception as e:
            logger.warning(f"Language detection failed: {e}. Using 'en' as default.")
            return "en"
    
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
            language: Language code or "auto" for detection
            max_length: Maximum sentence length (characters)
            min_length: Minimum sentence length (characters)
            
        Returns:
            List of tuples (sentence, (start_pos, end_pos))
        """
        if not text or not text.strip():
            return []
        
        # Detect language if needed
        if language == "auto":
            language = self._detect_language(text)
            logger.debug(f"Detected language: {language}")
        
        # Get appropriate chunker
        chunker = self._get_chunker(language)
        
        # Perform chunking
        sentences = chunker.chunk(text, **kwargs)
        
        # Post-process: filter by length if specified
        if min_length or max_length:
            filtered = []
            for sent, pos in sentences:
                sent_len = len(sent)
                if min_length and sent_len < min_length:
                    continue
                if max_length and sent_len > max_length:
                    # Split long sentences
                    # TODO: Implement smart splitting
                    filtered.append((sent[:max_length], pos))
                else:
                    filtered.append((sent, pos))
            sentences = filtered
        
        return sentences
    
    @staticmethod
    def reconstruct_text(sentences: List[str], 
                        masks: List[bool], 
                        positions: Optional[List[Tuple[int, int]]] = None,
                        original_text: Optional[str] = None) -> str:
        """
        Reconstruct text from sentences and masks.
        
        Args:
            sentences: List of sentences
            masks: Binary masks indicating which sentences to keep
            positions: Original positions of sentences
            original_text: Original text for exact reconstruction
            
        Returns:
            Reconstructed text with only kept sentences
        """
        if not sentences:
            return ""
        
        # Ensure masks match sentences length
        if len(masks) < len(sentences):
            masks = list(masks) + [False] * (len(sentences) - len(masks))
        
        # Convert masks to boolean list if needed
        masks = [bool(m) for m in masks]
        
        if original_text and positions:
            # Use original text and positions for exact reconstruction
            result = []
            last_end = 0
            
            for i, (sent, mask, (start, end)) in enumerate(zip(sentences, masks, positions)):
                if mask:
                    # Include any text between sentences (e.g., whitespace)
                    if start > last_end:
                        result.append(original_text[last_end:start])
                    result.append(original_text[start:end])
                    last_end = end
            
            return "".join(result)
        else:
            # Simple reconstruction with space joining
            kept_sentences = [sent for sent, mask in zip(sentences, masks) if mask]
            return " ".join(kept_sentences)
    
    @staticmethod
    def get_supported_languages() -> List[str]:
        """Get list of languages with specialized chunkers."""
        return ["ja", "en", "zh"]  # Languages with specialized support
    
    @staticmethod
    def get_chunker_info(language: str) -> str:
        """Get information about the chunker for a language."""
        info_map = {
            "ja": "Bunkai (High-precision Japanese sentence splitter)",
            "en": "NLTK Punkt Tokenizer (English-optimized)",
            "zh": "Regex-based (Chinese punctuation)",
        }
        return info_map.get(language, "Regex-based (Universal fallback)")


class RobustMultilingualChunker(MultilingualChunker):
    """
    Robust version with enhanced error handling and fallback strategies.
    """
    
    def chunk_text(self, text: str, **kwargs) -> List[Tuple[str, Tuple[int, int]]]:
        """Chunk with multiple fallback strategies."""
        try:
            return super().chunk_text(text, **kwargs)
        except Exception as e:
            logger.warning(f"Chunking failed: {e}. Using simple fallback.")
            
            # Fallback 1: Split by newlines and punctuation
            sentences = []
            for line in text.split('\n'):
                if line.strip():
                    # Split by common punctuation
                    parts = re.split(r'[。．.!?！？]\s*', line)
                    for part in parts:
                        part = part.strip()
                        if part:
                            # Find position in original text
                            start = text.find(part)
                            if start != -1:
                                sentences.append((part, (start, start + len(part))))
                            else:
                                sentences.append((part, (0, len(part))))
            
            # Fallback 2: Return whole text if no sentences found
            if not sentences and text.strip():
                sentences = [(text.strip(), (0, len(text)))]
            
            return sentences