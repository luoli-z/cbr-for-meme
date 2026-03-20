# -*- coding: utf-8 -*-
"""
Configuration for Retrieval-Augmented Multi-Tools Framework for Zero-shot Meme Detection
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ===================== API Configuration =====================
# Default API key (can be overridden by environment variable)
DEFAULT_API_KEY = "your-api-key"
API_BASE_URL = "https://your-api-endpoint/v1"

# Set environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY)

# ===================== Model Configuration =====================
@dataclass
class ModelConfig:
    """Configuration for different LLM models"""
    name: str
    api_name: str  # Name used in API calls
    supports_vision: bool = True
    max_tokens: int = 2048
    temperature: float = 0.0
    
# Available models for experiments
AVAILABLE_MODELS = {
    # Primary model (default)
    "gemini-flash": ModelConfig(
        name="Gemini 2.0 Flash",
        api_name="gemini-2.0-flash",
        supports_vision=True,
        max_tokens=2048,
        temperature=0.0
    ),
    # Alternative models for robustness experiments
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o Mini",
        api_name="gpt-4o-mini",
        supports_vision=True,
        max_tokens=2048,
        temperature=0.0
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        api_name="gpt-4o",
        supports_vision=True,
        max_tokens=4096,
        temperature=0.0
    ),
    "qwen-plus": ModelConfig(
        name="Qwen Plus",
        api_name="qwen-plus",
        supports_vision=True,
        max_tokens=2048,
        temperature=0.0
    ),
    "qwen-vl-max": ModelConfig(
        name="Qwen VL Max",
        api_name="qwen-vl-max",
        supports_vision=True,
        max_tokens=2048,
        temperature=0.0
    ),
}

# Default model selection
DEFAULT_MODEL = "gemini-flash"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# ===================== Framework Configuration =====================
@dataclass
class FrameworkConfig:
    """Main configuration for the meme detection framework"""
    # Retrieval settings
    top_k_retrieval: int = 5  # Number of similar samples to retrieve
    retrieval_weight_text: float = 0.2  # Weight for text similarity
    retrieval_weight_image: float = 0.8  # Weight for image similarity
    
    # Router settings
    max_tools_to_select: int = 3  # Maximum number of tools to select per sample
    min_tools_to_select: int = 3  # Minimum number of tools to select
    fixed_tool_count: Optional[int] = None  # If set, Router MUST select exactly this many tools (for experiments)
    
    # Model settings
    llm_model: str = DEFAULT_MODEL
    vision_model: str = DEFAULT_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    
    # Execution settings
    parallel_tool_execution: bool = True  # Enable parallel tool execution
    max_workers: int = 4  # Maximum parallel workers
    
    # Output settings
    verbose: bool = True  # Detailed logging
    save_intermediate: bool = True  # Save intermediate results
    

# ===================== Tool Configuration =====================
@dataclass
class ToolConfig:
    """Configuration for cognitive tools"""
    name: str
    description: str
    enabled: bool = True
    requires_vision: bool = False
    priority: int = 5  # 1-10, higher is more important

# Define all 8 cognitive tools
COGNITIVE_TOOLS = {
    "sentiment_reversal": ToolConfig(
        name="Sentiment Reversal Detector",
        description="Analyzes sentiment polarity contrast between text and image",
        requires_vision=True,
        priority=8
    ),
    "image_text_aligner": ToolConfig(
        name="Fine-grained Image-Text Aligner",
        description="Checks entity and attribute consistency between text and image",
        requires_vision=True,
        priority=9
    ),
    "visual_rhetoric": ToolConfig(
        name="Visual Rhetoric Decoder",
        description="Identifies visual rhetorical devices like exaggeration and juxtaposition",
        requires_vision=True,
        priority=6
    ),
    "micro_expression": ToolConfig(
        name="Micro-Expression Analyzer",
        description="Analyzes facial expressions in relation to textual context",
        requires_vision=True,
        priority=7
    ),
    "culture_retriever": ToolConfig(
        name="Culture Knowledge Retriever",
        description="Identifies cultural references, celebrities, and contextual knowledge",
        requires_vision=True,
        priority=8
    ),
    "pragmatic_irony": ToolConfig(
        name="Pragmatic Irony Identifier",
        description="Detects linguistic irony markers like rhetorical questions and sarcasm",
        requires_vision=False,
        priority=7
    ),
    "scene_text_ocr": ToolConfig(
        name="Scene Text OCR Integrator",
        description="Extracts and analyzes embedded text within images",
        requires_vision=True,
        priority=6
    ),
    "target_identifier": ToolConfig(
        name="Target Identification Probe",
        description="Identifies the target of the meme (self-deprecation, individual, social phenomenon)",
        requires_vision=True,
        priority=5
    ),
}


# ===================== Dataset Configuration =====================
DATASET_CONFIGS = {
    "FHM": {
        "image_key": "img",
        "text_key": "text",
        "label_key": "label",
        "label_mapping": None,
        "positive_label": "harmful",
        "negative_label": "harmless"
    },
    "HarM": {
        "image_key": "image",
        "text_key": "text",
        "label_key": "labels",
        "label_mapping": {"not harmful": 0, "default_harmful": 1},
        "positive_label": "harmful",
        "negative_label": "harmless"
    },
    "MAMI": {
        "image_key": "image",
        "text_key": "text",
        "label_key": "label",
        "label_mapping": None,
        "positive_label": "misogynistic",
        "negative_label": "not misogynistic"
    }
}


# ===================== Path Configuration =====================
@dataclass
class PathConfig:
    """Path configuration for data and results"""
    base_dir: str = "."
    data_dir: str = "data"
    results_dir: str = "results"
    cache_dir: str = "cache"
    embeddings_dir: str = "embeddings"
    knowledge_base_dir: str = "knowledge_base"
    
    def get_dataset_path(self, dataset_name: str) -> str:
        return os.path.join(self.base_dir, self.data_dir, dataset_name)
    
    def get_image_path(self, dataset_name: str) -> str:
        return os.path.join(self.get_dataset_path(dataset_name), "images")
    
    def get_results_path(self, dataset_name: str) -> str:
        path = os.path.join(self.base_dir, self.results_dir, dataset_name)
        os.makedirs(path, exist_ok=True)
        return path


# Create default instances
DEFAULT_FRAMEWORK_CONFIG = FrameworkConfig()
DEFAULT_PATH_CONFIG = PathConfig()
