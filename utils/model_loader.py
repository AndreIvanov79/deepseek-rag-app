import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Dict, Any

class ModelLoader:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-33b-instruct", 
                 cache_dir: str = "./models"):
        """
        Initialize the ModelLoader.
        
        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory to store the model weights
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_model(self) -> Tuple[Any, Any]:
        """
        Load the DeepSeek-Coder model and tokenizer.
        
        Returns:
            Tuple containing the model and tokenizer
        """
        print(f"Loading model {self.model_name} on {self.device}...")
        
        # Model loading parameters
        load_params = {
            "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "cache_dir": self.cache_dir,
        }
        
        # Load with optimizations if on CUDA
        if self.device == "cuda":
            if torch.cuda.get_device_properties(0).total_memory > 24 * 1024 * 1024 * 1024:  # More than 24GB VRAM
                # Load with 8-bit quantization for larger GPUs but still with memory optimization
                load_params["load_in_8bit"] = True
            else:
                # Load with 4-bit quantization for smaller GPUs
                load_params["load_in_4bit"] = True
                load_params["bnb_4bit_compute_dtype"] = torch.bfloat16
                load_params["bnb_4bit_quant_type"] = "nf4"
                load_params["bnb_4bit_use_double_quant"] = True
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_params
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        
        print(f"Model loaded successfully on {self.device}")
        return self.model, self.tokenizer
    
    def get_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """
        Get the loaded model and tokenizer, or load them if not already loaded.
        
        Returns:
            Tuple containing the model and tokenizer
        """
        if self.model is None or self.tokenizer is None:
            return self.load_model()
        return self.model, self.tokenizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        # Get model memory usage if on CUDA
        memory_info = {}
        if self.device == "cuda" and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
            memory_info = {
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
            }
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype),
            "cache_dir": self.cache_dir,
            **memory_info,
        }