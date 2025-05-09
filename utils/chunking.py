from typing import List, Dict, Any, Callable, Optional

class ResponseChunker:
    """
    Utility for chunking large responses to handle token limits.
    """
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        A rough approximation is 4 characters per token for English text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4  # Simple approximation
    
    @staticmethod
    def chunk_text(text: str, 
                  max_tokens_per_chunk: int = 2000, 
                  overlap_tokens: int = 50) -> List[str]:
        """
        Split a large text into chunks based on estimated token count.
        
        Args:
            text: Input text to chunk
            max_tokens_per_chunk: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        if ResponseChunker.estimate_tokens(text) <= max_tokens_per_chunk:
            return [text]
        
        # Approximate characters per chunk
        chars_per_chunk = max_tokens_per_chunk * 4
        overlap_chars = overlap_tokens * 4
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find a good breakpoint (at a newline if possible)
            end = min(start + chars_per_chunk, len(text))
            
            # Try to break at a paragraph or sentence
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + chars_per_chunk // 2:
                    end = paragraph_break + 2
                else:
                    # Try sentence break
                    sentence_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('.\n', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('? ', start, end)
                    )
                    
                    if sentence_break != -1 and sentence_break > start + chars_per_chunk // 2:
                        end = sentence_break + 2
                    else:
                        # Last resort: break at last space
                        last_space = text.rfind(' ', start, end)
                        if last_space != -1 and last_space > start + chars_per_chunk // 2:
                            end = last_space + 1
            
            chunks.append(text[start:end])
            
            # Move start position for next chunk, considering overlap
            if end == len(text):
                break
                
            start = max(start + chars_per_chunk - overlap_chars, end - overlap_chars)
        
        return chunks
    
    @staticmethod
    def chunk_response(response: Dict[str, Any], 
                      max_tokens: int = 2000) -> List[Dict[str, Any]]:
        """
        Chunk a response dictionary for manageable transmission.
        
        Args:
            response: Response dictionary with text content
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of chunked response dictionaries
        """
        if 'response' not in response:
            return [response]
        
        text = response['response']
        chunks = ResponseChunker.chunk_text(text, max_tokens)
        
        if len(chunks) == 1:
            return [response]
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_response = response.copy()
            chunk_response['response'] = chunk
            chunk_response['chunk_info'] = {
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            result.append(chunk_response)
            
        return result
    
    @staticmethod
    def process_with_chunking(processor_func: Callable, 
                             max_tokens: int = 2000, 
                             **kwargs) -> List[Dict[str, Any]]:
        """
        Process a query with automatic chunking of large responses.
        
        Args:
            processor_func: Function that processes the query and returns a response
            max_tokens: Maximum tokens per chunk
            **kwargs: Arguments to pass to processor_func
            
        Returns:
            List of response chunks
        """
        response = processor_func(**kwargs)
        return ResponseChunker.chunk_response(response, max_tokens)