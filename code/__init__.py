# -*- coding: utf-8 -*-
"""
Retrieval-Augmented Multi-Tools Framework for Zero-shot Meme Detection

This framework implements a comprehensive approach to meme detection with:
1. Contextual Anchor Retrieval - Knowledge base for similar meme retrieval
2. Multi-View Cognitive Tools - 8 specialized analysis tools
3. Cognitive Router - Intelligent tool selection
4. Dialectical Adjudicator - Evidence synthesis and final decision

Usage:
    from framework import MemeDetectionPipeline
    
    pipeline = MemeDetectionPipeline(
        dataset_name="FHM",
        model="gemini-flash",
        use_knowledge_base=True
    )
    
    result = pipeline.process_single(image_path, text)
    print(f"Prediction: {result.predicted_label}")
    print(f"Reasoning: {result.reasoning}")
"""

from framework.config import (
    FrameworkConfig,
    ModelConfig,
    ToolConfig,
    PathConfig,
    AVAILABLE_MODELS,
    COGNITIVE_TOOLS,
    DATASET_CONFIGS,
    DEFAULT_MODEL,
    DEFAULT_FRAMEWORK_CONFIG,
    DEFAULT_PATH_CONFIG
)

from framework.knowledge_base import (
    KnowledgeBase,
    TrainSample,
    RetrievalResult,
    SSRResult,
    verify_knowledge_base
)

from framework.tools import (
    ToolType,
    ToolObservation,
    BaseCognitiveTool,
    CognitiveToolManager,
    SentimentReversalDetector,
    ImageTextAligner,
    VisualRhetoricDecoder,
    MicroExpressionAnalyzer,
    CultureRetriever,
    PragmaticIronyIdentifier,
    SceneTextOCR,
    TargetIdentifier
)

from framework.router import (
    CognitiveRouter,
    RoutingPlan,
    AdaptiveRouter
)

from framework.adjudicator import (
    DialecticalAdjudicator,
    AdjudicationResult,
    EnsembleAdjudicator
)

from framework.pipeline import (
    MemeDetectionPipeline,
    AblationPipeline,
    PipelineResult
)

__version__ = "1.0.0"
__author__ = "MIND Research Team"

__all__ = [
    # Config
    "FrameworkConfig",
    "ModelConfig",
    "ToolConfig",
    "PathConfig",
    "AVAILABLE_MODELS",
    "COGNITIVE_TOOLS",
    "DATASET_CONFIGS",
    "DEFAULT_MODEL",
    
    # Knowledge Base
    "KnowledgeBase",
    "TrainSample",
    "RetrievalResult",
    "SSRResult",
    "verify_knowledge_base",
    
    # Tools
    "ToolType",
    "ToolObservation",
    "BaseCognitiveTool",
    "CognitiveToolManager",
    "SentimentReversalDetector",
    "ImageTextAligner",
    "VisualRhetoricDecoder",
    "MicroExpressionAnalyzer",
    "CultureRetriever",
    "PragmaticIronyIdentifier",
    "SceneTextOCR",
    "TargetIdentifier",
    
    # Router
    "CognitiveRouter",
    "RoutingPlan",
    "AdaptiveRouter",
    
    # Adjudicator
    "DialecticalAdjudicator",
    "AdjudicationResult",
    "EnsembleAdjudicator",
    
    # Pipeline
    "MemeDetectionPipeline",
    "AblationPipeline",
    "PipelineResult"
]
