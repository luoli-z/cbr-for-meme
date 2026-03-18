# -*- coding: utf-8 -*-
"""

This module implements the "Retrieval-as-Experience" mechanism that:
1. Loads pre-computed SSR (Similar Sample Retrieval) results
2. Loads training data with AI-generated explanations
3. Provides context for the Router and Adjudicator

Key Design Decisions:
- Uses existing SSR results (SSR/XXX_SSR.jsonl) instead of re-computing embeddings
- Expects training data with explanations (data/XXX/train_with_explanations.jsonl)
- k (number of similar samples) is a configurable hyperparameter
"""
import os
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.config import DATASET_CONFIGS, DEFAULT_FRAMEWORK_CONFIG


@dataclass
class TrainSample:
    """Represents a training sample with explanation"""
    index: int
    image_filename: str
    text: str
    label: int
    explanation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "image_filename": self.image_filename,
            "text": self.text,
            "label": self.label,
            "explanation": self.explanation
        }


@dataclass
class SSRResult:
    """SSR (Similar Sample Retrieval) result for a test sample"""
    test_index: int
    similar_indices: List[int]  # Indices into training set
    similarity_scores: List[float]


@dataclass
class RetrievalResult:
    """Result from knowledge base retrieval"""
    query_index: int
    retrieved_samples: List[TrainSample]
    similarity_scores: List[float]
    
    @property
    def retrieved_examples(self) -> List[TrainSample]:
        """Alias for retrieved_samples (for backward compatibility)"""
        return self.retrieved_samples
    
    def get_context_string(self, top_k: Optional[int] = None) -> str:
        """
        Generate context string for in-context learning
        
        Args:
            top_k: Number of samples to include (None = all)
        """
        samples = self.retrieved_samples[:top_k] if top_k else self.retrieved_samples
        scores = self.similarity_scores[:top_k] if top_k else self.similarity_scores
        
        context_parts = []
        for i, (sample, score) in enumerate(zip(samples, scores)):
            label_str = "harmful" if sample.label == 1 else "harmless"
            
            part = f"\n=== Reference Case {i+1} (Similarity: {score:.3f}) ===\n"
            part += f"Text: \"{sample.text}\"\n"
            part += f"Label: {label_str}\n"
            
            if sample.explanation:
                part += f"Analysis: {sample.explanation}\n"
            
            context_parts.append(part)
        
        return "\n".join(context_parts)
    
    def get_explanations_only(self, top_k: Optional[int] = None) -> List[str]:
        """Get only the explanations from retrieved samples"""
        samples = self.retrieved_samples[:top_k] if top_k else self.retrieved_samples
        return [s.explanation for s in samples if s.explanation]
    
    def to_dict(self) -> Dict:
        return {
            "query_index": self.query_index,
            "retrieved_samples": [s.to_dict() for s in self.retrieved_samples],
            "similarity_scores": self.similarity_scores
        }


class KnowledgeBase:
    """
    Knowledge Base for Contextual Anchor Retrieval
    
    This class manages:
    1. Loading pre-computed SSR results
    2. Loading training data with explanations
    3. Providing retrieval results for test samples
    
    Usage:
        kb = KnowledgeBase("FHM")
        kb.load()
        
        # For test sample index 5, get top-3 similar training samples
        result = kb.retrieve_by_test_index(5, top_k=3)
        print(result.get_context_string())
    """
    
    def __init__(
        self,
        dataset_name: str,
        config: Optional[Any] = None,
        ssr_path: Optional[str] = None,
        train_path: Optional[str] = None
    ):
        """
        Initialize Knowledge Base
        
        Args:
            dataset_name: Dataset name (FHM, HarM, MAMI)
            config: Framework configuration
            ssr_path: Path to SSR results (default: SSR/{dataset}_SSR.jsonl)
            train_path: Path to training data with explanations 
                       (default: data/{dataset}/train_with_explanations.jsonl)
        """
        self.dataset_name = dataset_name
        self.config = config or DEFAULT_FRAMEWORK_CONFIG
        
        # Default paths
        self.ssr_path = ssr_path or f"SSR/{dataset_name}_SSR.jsonl"
        self.train_path = train_path or f"data/{dataset_name}/train_with_explanations.jsonl"
        self.train_original_path = f"data/{dataset_name}/train.jsonl"
        
        # Dataset configuration
        self.dataset_config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["FHM"])
        
        # Storage
        self.train_samples: Dict[int, TrainSample] = {}  # index -> TrainSample
        self.ssr_results: Dict[int, SSRResult] = {}  # test_index -> SSRResult
        
        # State
        self.is_loaded = False
        self.has_explanations = False
    
    def _get_item_data(self, item: Dict, index: int) -> TrainSample:
        """Extract training sample from data item"""
        config = self.dataset_config
        
        image_filename = item.get(config["image_key"])
        text = item.get(config["text_key"])
        raw_label = item.get(config["label_key"])
        
        # Process label
        if self.dataset_name == "HarM":
            if isinstance(raw_label, list) and raw_label:
                label = 0 if raw_label[0].lower() == "not harmful" else 1
            else:
                label = 1
        else:
            label = raw_label if raw_label is not None else 0
        
        # Get explanation if available
        explanation = item.get("explanation")
        
        return TrainSample(
            index=index,
            image_filename=image_filename,
            text=text,
            label=label,
            explanation=explanation
        )
    
    def load(self, require_explanations: bool = False) -> bool:
        """
        Load SSR results and training data
        
        Args:
            require_explanations: If True, fail if explanations are not available
            
        Returns:
            True if loaded successfully
        """
        # Step 1: Load training data (prefer with explanations)
        if os.path.exists(self.train_path):
            print(f"Loading training data with explanations: {self.train_path}")
            with open(self.train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    idx = item.get("train_index", len(self.train_samples))
                    sample = self._get_item_data(item, idx)
                    self.train_samples[idx] = sample
            self.has_explanations = True
            print(f"  Loaded {len(self.train_samples)} samples with explanations")
            
        elif os.path.exists(self.train_original_path):
            if require_explanations:
                raise FileNotFoundError(
                    f"Training data with explanations not found: {self.train_path}\n"
                    f"Please run: python framework/generate_explanations.py --dataset {self.dataset_name}"
                )
            
            print(f"Warning: Explanations not found, loading original training data: {self.train_original_path}")
            with open(self.train_original_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    item = json.loads(line)
                    sample = self._get_item_data(item, idx)
                    self.train_samples[idx] = sample
            self.has_explanations = False
            print(f"  Loaded {len(self.train_samples)} samples (no explanations)")
            
        else:
            raise FileNotFoundError(f"Training data not found for {self.dataset_name}")
        
        # Step 2: Load SSR results
        if os.path.exists(self.ssr_path):
            print(f"Loading SSR results: {self.ssr_path}")
            with open(self.ssr_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    test_idx = item["index"]
                    self.ssr_results[test_idx] = SSRResult(
                        test_index=test_idx,
                        similar_indices=item["samples"],
                        similarity_scores=item["scores"]
                    )
            print(f"  Loaded SSR results for {len(self.ssr_results)} test samples")
        else:
            raise FileNotFoundError(
                f"SSR results not found: {self.ssr_path}\n"
                f"Please run SSR.py first to generate similarity results."
            )
        
        self.is_loaded = True
        return True
    
    def retrieve_by_test_index(
        self,
        test_index: int,
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve similar training samples for a test sample
        
        Args:
            test_index: Index of the test sample
            top_k: Number of similar samples to return (default from config)
            
        Returns:
            RetrievalResult with similar samples and scores
        """
        if not self.is_loaded:
            raise RuntimeError("Knowledge base not loaded. Call load() first.")
        
        if top_k is None:
            top_k = self.config.top_k_retrieval
        
        # Get SSR result for this test sample
        if test_index not in self.ssr_results:
            raise KeyError(f"Test index {test_index} not found in SSR results")
        
        ssr = self.ssr_results[test_index]
        
        # Get training samples
        retrieved_samples = []
        scores = []
        
        for train_idx, score in zip(ssr.similar_indices[:top_k], ssr.similarity_scores[:top_k]):
            if train_idx in self.train_samples:
                retrieved_samples.append(self.train_samples[train_idx])
                scores.append(score)
            else:
                print(f"Warning: Train index {train_idx} not found in training data")
        
        return RetrievalResult(
            query_index=test_index,
            retrieved_samples=retrieved_samples,
            similarity_scores=scores
        )
    
    def get_train_sample(self, train_index: int) -> Optional[TrainSample]:
        """Get a specific training sample by index"""
        return self.train_samples.get(train_index)
    
    def get_max_available_k(self, test_index: int) -> int:
        """Get maximum available k for a test sample"""
        if test_index in self.ssr_results:
            return len(self.ssr_results[test_index].similar_indices)
        return 0
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        stats = {
            "dataset": self.dataset_name,
            "is_loaded": self.is_loaded,
            "has_explanations": self.has_explanations,
            "num_train_samples": len(self.train_samples),
            "num_test_samples_with_ssr": len(self.ssr_results),
        }
        
        if self.ssr_results:
            # Get max k from first SSR result
            first_ssr = next(iter(self.ssr_results.values()))
            stats["max_k_available"] = len(first_ssr.similar_indices)
        
        # Count samples with explanations
        if self.has_explanations:
            with_exp = sum(1 for s in self.train_samples.values() if s.explanation)
            stats["samples_with_explanations"] = with_exp
        
        return stats


class KnowledgeBaseManager:
    """
    Manager for multiple knowledge bases (multi-dataset support)
    """
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config or DEFAULT_FRAMEWORK_CONFIG
        self.knowledge_bases: Dict[str, KnowledgeBase] = {}
    
    def get_kb(self, dataset_name: str) -> KnowledgeBase:
        """Get or create knowledge base for a dataset"""
        if dataset_name not in self.knowledge_bases:
            kb = KnowledgeBase(dataset_name, config=self.config)
            kb.load()
            self.knowledge_bases[dataset_name] = kb
        return self.knowledge_bases[dataset_name]
    
    def preload_all(self, datasets: List[str] = ["FHM", "HarM", "MAMI"]):
        """Preload all knowledge bases"""
        for dataset in datasets:
            try:
                self.get_kb(dataset)
            except Exception as e:
                print(f"Warning: Could not load KB for {dataset}: {e}")


def verify_knowledge_base(dataset_name: str) -> bool:
    """
    Verify that knowledge base is properly set up for a dataset
    
    Returns True if:
    1. SSR results exist
    2. Training data with explanations exists
    """
    ssr_path = f"SSR/{dataset_name}_SSR.jsonl"
    train_exp_path = f"data/{dataset_name}/train_with_explanations.jsonl"
    train_path = f"data/{dataset_name}/train.jsonl"
    
    print(f"\nVerifying Knowledge Base for {dataset_name}:")
    
    # Check SSR
    if os.path.exists(ssr_path):
        with open(ssr_path, 'r') as f:
            num_ssr = sum(1 for _ in f)
        print(f"  ✓ SSR results: {num_ssr} test samples")
    else:
        print(f"  ✗ SSR results not found: {ssr_path}")
        print(f"    → Run: python SSR.py")
        return False
    
    # Check training data with explanations
    if os.path.exists(train_exp_path):
        with open(train_exp_path, 'r') as f:
            num_train = sum(1 for _ in f)
        print(f"  ✓ Training data with explanations: {num_train} samples")
        return True
    else:
        # Check original training data
        if os.path.exists(train_path):
            with open(train_path, 'r') as f:
                num_train = sum(1 for _ in f)
            print(f"  ! Original training data found: {num_train} samples")
            print(f"  ✗ Explanations not generated")
            print(f"    → Run: python framework/generate_explanations.py --dataset {dataset_name}")
            return False
        else:
            print(f"  ✗ Training data not found")
            return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FHM", help="Dataset to verify/test")
    parser.add_argument("--verify", action="store_true", help="Verify KB setup")
    parser.add_argument("--test", action="store_true", help="Test retrieval")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k for test")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_knowledge_base(args.dataset)
    
    if args.test:
        print(f"\nTesting Knowledge Base for {args.dataset}:")
        
        try:
            kb = KnowledgeBase(args.dataset)
            kb.load(require_explanations=False)
            
            print(f"\nStatistics: {kb.get_statistics()}")
            
            # Test retrieval for first test sample
            print(f"\nRetrieving top-{args.top_k} for test sample 0:")
            result = kb.retrieve_by_test_index(0, top_k=args.top_k)
            print(result.get_context_string())
            
        except Exception as e:
            print(f"Error: {e}")
