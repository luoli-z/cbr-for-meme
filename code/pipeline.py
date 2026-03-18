# -*- coding: utf-8 -*-
"""
Main Pipeline: Retrieval-Augmented Multi-Tools Framework for Zero-shot Meme Detection

This module integrates all components:
1. Knowledge Base (Contextual Anchor Retrieval)
2. Cognitive Tools (Multi-View Analysis)
3. Router (Tool Selection)
4. Adjudicator (Final Decision)
"""
import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from tqdm import tqdm

from openai import OpenAI

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.config import (
    API_BASE_URL, DEFAULT_MODEL, DEFAULT_FRAMEWORK_CONFIG,
    DEFAULT_PATH_CONFIG, DATASET_CONFIGS, FrameworkConfig
)
from framework.knowledge_base import KnowledgeBase, RetrievalResult, TrainSample
from framework.tools import CognitiveToolManager, ToolObservation, ToolType
from framework.router import CognitiveRouter, RoutingPlan
from framework.adjudicator import DialecticalAdjudicator, AdjudicationResult


@dataclass
class PipelineResult:
    """Complete result from pipeline processing"""
    sample_index: int
    image_path: str
    text: str
    actual_label: Optional[int]
    
    # Component results
    retrieval_result: Optional[RetrievalResult]
    routing_plan: RoutingPlan
    tool_observations: List[ToolObservation]
    adjudication_result: AdjudicationResult
    
    # Final outputs
    predicted_label: int
    confidence: float
    reasoning: str
    
    # Timing
    processing_time: float
    
    def to_dict(self) -> Dict:
        return {
            "index": self.sample_index,
            "image_path": self.image_path,
            "text": self.text,
            "actual": self.actual_label,
            "predicted": self.predicted_label,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "selected_tools": [t.value for t in self.routing_plan.selected_tools],
            "routing_reasoning": self.routing_plan.reasoning,
            "tool_observations": [obs.to_dict() for obs in self.tool_observations],
            "key_evidence": self.adjudication_result.key_evidence,
            "core_contradiction": self.adjudication_result.core_contradiction,
            "processing_time": self.processing_time
        }
    
    def is_correct(self) -> Optional[bool]:
        if self.actual_label is None:
            return None
        return self.predicted_label == self.actual_label


class MemeDetectionPipeline:
    """
    Main pipeline for retrieval-augmented multi-tool meme detection
    """
    
    def __init__(
        self,
        dataset_name: str,
        config: Optional[FrameworkConfig] = None,
        model: str = DEFAULT_MODEL,
        use_knowledge_base: bool = True,
        preload_knowledge_base: bool = True
    ):
        self.dataset_name = dataset_name
        self.config = config or DEFAULT_FRAMEWORK_CONFIG
        self.model = model
        self.use_knowledge_base = use_knowledge_base
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=API_BASE_URL
        )
        
        # Dataset configuration
        self.dataset_config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["FHM"])
        
        # Initialize components
        self.knowledge_base: Optional[KnowledgeBase] = None
        if use_knowledge_base and preload_knowledge_base:
            self._load_knowledge_base()
        
        self.tool_manager = CognitiveToolManager(
            client=self.client,
            model=model,
            dataset_name=dataset_name
        )
        
        self.router = CognitiveRouter(
            client=self.client,
            model=model,
            config=self.config
        )
        
        self.adjudicator = DialecticalAdjudicator(
            client=self.client,
            model=model,
            config=self.config,
            dataset_name=dataset_name
        )
        
        # Paths
        self.path_config = DEFAULT_PATH_CONFIG
        self.base_path = self.path_config.get_dataset_path(dataset_name)
        self.image_base_path = self.path_config.get_image_path(dataset_name)
    
    def _load_knowledge_base(self):
        """Load knowledge base from SSR results and training data with explanations"""
        self.knowledge_base = KnowledgeBase(
            self.dataset_name,
            config=self.config
        )
        
        try:
            # Try to load with explanations first
            self.knowledge_base.load(require_explanations=False)
            print(f"Knowledge base loaded: {self.knowledge_base.get_statistics()}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            self.knowledge_base = None
    
    def _get_item_data(self, item: Dict) -> Tuple[str, str, int]:
        """Extract image filename, text, and label from data item"""
        config = self.dataset_config
        image_key = config["image_key"]
        text_key = config["text_key"]
        label_key = config["label_key"]
        
        image_filename = item.get(image_key)
        text = item.get(text_key)
        raw_label = item.get(label_key)
        
        # Process label
        if self.dataset_name == "HarM":
            if isinstance(raw_label, list) and raw_label:
                label = 0 if raw_label[0].lower() == "not harmful" else 1
            else:
                label = 1
        else:
            label = raw_label if raw_label is not None else 0
            
        return image_filename, text, label
    
    def process_single(
        self,
        image_path: str,
        text: str,
        sample_index: int = 0,
        actual_label: Optional[int] = None,
        use_retrieval: bool = True
    ) -> PipelineResult:
        """
        Process a single meme sample through the full pipeline
        
        Args:
            image_path: Path to meme image
            text: Meme text
            sample_index: Index of the sample
            actual_label: Ground truth label (optional)
            use_retrieval: Whether to use retrieval augmentation
            
        Returns:
            PipelineResult with all intermediate and final results
        """
        start_time = time.time()
        
        # Step 1: Retrieval (if enabled)
        # Uses pre-computed SSR results from SSR/{dataset}_SSR.jsonl
        retrieval_result = None
        if use_retrieval and self.use_knowledge_base and self.knowledge_base:
            try:
                retrieval_result = self.knowledge_base.retrieve_by_test_index(
                    test_index=sample_index,
                    top_k=self.config.top_k_retrieval
                )
            except KeyError:
                # Test index not in SSR results
                print(f"Warning: Test index {sample_index} not found in SSR results")
                retrieval_result = None
        
        # Step 2: Routing (tool selection)
        if retrieval_result:
            routing_plan = self.router.route(
                image_path, text, retrieval_result
            )
        else:
            routing_plan = self.router.route_simple(image_path, text)
        
        # Step 3: Execute selected tools (parallel if configured)
        tool_observations = self.tool_manager.execute_tools(
            routing_plan.selected_tools,
            image_path,
            text,
            parallel=self.config.parallel_tool_execution,
            max_workers=self.config.max_workers
        )
        
        # Step 4: Adjudication (final decision)
        adjudication_result = self.adjudicator.adjudicate(
            image_path, text,
            tool_observations,
            routing_plan=routing_plan,
            retrieval_result=retrieval_result
        )
        
        processing_time = time.time() - start_time
        
        return PipelineResult(
            sample_index=sample_index,
            image_path=image_path,
            text=text,
            actual_label=actual_label,
            retrieval_result=retrieval_result,
            routing_plan=routing_plan,
            tool_observations=tool_observations,
            adjudication_result=adjudication_result,
            predicted_label=adjudication_result.prediction,
            confidence=adjudication_result.confidence,
            reasoning=adjudication_result.reasoning_summary,
            processing_time=processing_time
        )
    
    def process_dataset(
        self,
        test_jsonl_path: Optional[str] = None,
        output_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        start_from: int = 0,
        use_retrieval: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Process entire test dataset
        
        Args:
            test_jsonl_path: Path to test JSONL file
            output_path: Path to save results
            max_samples: Maximum samples to process
            start_from: Start processing from this index
            use_retrieval: Whether to use retrieval
            show_progress: Show progress bar
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load test data
        if test_jsonl_path is None:
            test_jsonl_path = os.path.join(self.base_path, "test.jsonl")
        
        with open(test_jsonl_path, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f.readlines()]
        
        if max_samples:
            test_data = test_data[start_from:start_from + max_samples]
        else:
            test_data = test_data[start_from:]
        
        # Prepare output
        if output_path is None:
            results_dir = self.path_config.get_results_path(self.dataset_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, f"results_{timestamp}.jsonl")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Process samples
        results: List[PipelineResult] = []
        correct_count = 0
        total_count = 0
        
        all_actual = []
        all_predicted = []
        
        iterator = tqdm(enumerate(test_data), total=len(test_data), desc="Processing") if show_progress else enumerate(test_data)
        
        with open(output_path, 'a', encoding='utf-8') as f_out:
            for idx_offset, item in iterator:
                sample_index = start_from + idx_offset
                
                # Get item data
                image_filename, text, label = self._get_item_data(item)
                
                if not image_filename or not text:
                    continue
                
                image_path = os.path.join(self.image_base_path, image_filename)
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                
                try:
                    # Process sample
                    result = self.process_single(
                        image_path, text,
                        sample_index=sample_index,
                        actual_label=label,
                        use_retrieval=use_retrieval
                    )
                    
                    results.append(result)
                    
                    # Update metrics
                    total_count += 1
                    all_actual.append(label)
                    all_predicted.append(result.predicted_label)
                    
                    if result.is_correct():
                        correct_count += 1
                    
                    # Save result
                    result_dict = result.to_dict()
                    result_dict["ratio"] = [total_count, correct_count]
                    json.dump(result_dict, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    f_out.flush()
                    
                    if show_progress:
                        acc = correct_count / total_count if total_count > 0 else 0
                        iterator.set_postfix({"acc": f"{acc:.4f}"})
                        
                except Exception as e:
                    print(f"Error processing sample {sample_index}: {e}")
                    continue
        
        # Calculate final metrics
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        # Calculate F1 score
        from sklearn.metrics import f1_score, precision_score, recall_score
        macro_f1 = f1_score(all_actual, all_predicted, average='macro') if all_actual else 0
        precision = precision_score(all_actual, all_predicted, average='macro') if all_actual else 0
        recall = recall_score(all_actual, all_predicted, average='macro') if all_actual else 0
        
        # Calculate average processing time
        avg_time = sum(r.processing_time for r in results) / len(results) if results else 0
        
        # Summary
        summary = {
            "dataset": self.dataset_name,
            "model": self.model,
            "total_samples": total_count,
            "correct_predictions": correct_count,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
            "average_processing_time": avg_time,
            "use_retrieval": use_retrieval,
            "top_k_retrieval": self.config.top_k_retrieval,
            "output_path": output_path
        }
        
        # Save summary
        with open(output_path, 'a', encoding='utf-8') as f_out:
            json.dump({"summary": summary}, f_out, ensure_ascii=False)
            f_out.write('\n')
        
        print(f"\n=== Results for {self.dataset_name} ===")
        print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Avg Processing Time: {avg_time:.2f}s")
        print(f"Results saved to: {output_path}")
        
        return summary


class AblationPipeline(MemeDetectionPipeline):
    """
    Extended pipeline for ablation studies
    Allows disabling specific components to measure their contribution
    """
    
    def __init__(
        self,
        dataset_name: str,
        config: Optional[FrameworkConfig] = None,
        model: str = DEFAULT_MODEL,
        use_knowledge_base: bool = True,
        use_routing: bool = True,
        use_all_tools: bool = False,
        specific_tools: Optional[List[ToolType]] = None
    ):
        super().__init__(
            dataset_name=dataset_name,
            config=config,
            model=model,
            use_knowledge_base=use_knowledge_base,
            preload_knowledge_base=use_knowledge_base
        )
        
        self.use_routing = use_routing
        self.use_all_tools = use_all_tools
        self.specific_tools = specific_tools
    
    def process_single(
        self,
        image_path: str,
        text: str,
        sample_index: int = 0,
        actual_label: Optional[int] = None,
        use_retrieval: bool = True
    ) -> PipelineResult:
        """Process with ablation settings"""
        start_time = time.time()
        
        # Step 1: Retrieval (uses pre-computed SSR results)
        retrieval_result = None
        if use_retrieval and self.use_knowledge_base and self.knowledge_base:
            try:
                retrieval_result = self.knowledge_base.retrieve_by_test_index(
                    test_index=sample_index,
                    top_k=self.config.top_k_retrieval
                )
            except KeyError:
                retrieval_result = None
        
        # Step 2: Tool selection (ablation options)
        if self.use_all_tools:
            # Use all tools (no routing)
            selected_tools = list(ToolType)
            routing_plan = RoutingPlan(
                selected_tools=selected_tools,
                reasoning="Using all tools (ablation)",
                reference_patterns=[],
                priority_order=selected_tools,
                confidence=1.0
            )
        elif self.specific_tools:
            # Use specific tools only
            routing_plan = RoutingPlan(
                selected_tools=self.specific_tools,
                reasoning=f"Using specific tools: {[t.value for t in self.specific_tools]}",
                reference_patterns=[],
                priority_order=self.specific_tools,
                confidence=1.0
            )
        elif self.use_routing:
            # Normal routing
            if retrieval_result:
                routing_plan = self.router.route(image_path, text, retrieval_result)
            else:
                routing_plan = self.router.route_simple(image_path, text)
        else:
            # Default tools without routing
            default_tools = [
                ToolType.SENTIMENT_REVERSAL,
                ToolType.IMAGE_TEXT_ALIGNER,
                ToolType.CULTURE_RETRIEVER
            ]
            routing_plan = RoutingPlan(
                selected_tools=default_tools,
                reasoning="Default tools (no routing)",
                reference_patterns=[],
                priority_order=default_tools,
                confidence=1.0
            )
        
        # Step 3: Execute tools
        tool_observations = self.tool_manager.execute_tools(
            routing_plan.selected_tools,
            image_path,
            text,
            parallel=self.config.parallel_tool_execution,
            max_workers=self.config.max_workers
        )
        
        # Step 4: Adjudication
        adjudication_result = self.adjudicator.adjudicate(
            image_path, text,
            tool_observations,
            routing_plan=routing_plan,
            retrieval_result=retrieval_result
        )
        
        processing_time = time.time() - start_time
        
        return PipelineResult(
            sample_index=sample_index,
            image_path=image_path,
            text=text,
            actual_label=actual_label,
            retrieval_result=retrieval_result,
            routing_plan=routing_plan,
            tool_observations=tool_observations,
            adjudication_result=adjudication_result,
            predicted_label=adjudication_result.prediction,
            confidence=adjudication_result.confidence,
            reasoning=adjudication_result.reasoning_summary,
            processing_time=processing_time
        )


def run_ablation_study(
    dataset_name: str,
    max_samples: int = 100
) -> Dict[str, Dict]:
    """
    Run ablation study with different configurations
    """
    results = {}
    
    configurations = [
        ("full", {"use_knowledge_base": True, "use_routing": True}),
        ("no_retrieval", {"use_knowledge_base": False, "use_routing": True}),
        ("no_routing", {"use_knowledge_base": True, "use_routing": False}),
        ("no_retrieval_no_routing", {"use_knowledge_base": False, "use_routing": False}),
        ("all_tools", {"use_knowledge_base": True, "use_routing": False, "use_all_tools": True}),
    ]
    
    for config_name, config_params in configurations:
        print(f"\n=== Running ablation: {config_name} ===")
        
        pipeline = AblationPipeline(
            dataset_name=dataset_name,
            **config_params
        )
        
        summary = pipeline.process_dataset(
            max_samples=max_samples,
            output_path=f"results/{dataset_name}/ablation_{config_name}.jsonl"
        )
        
        results[config_name] = summary
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Testing MemeDetectionPipeline...")
    
    # Test with a single sample
    pipeline = MemeDetectionPipeline(
        dataset_name="FHM",
        model="gemini-flash",
        use_knowledge_base=False,  # Skip KB for quick test
        preload_knowledge_base=False
    )
    
    test_image = "data/FHM/images/16395.png"
    test_text = "handjobs sold seperately"
    
    if os.path.exists(test_image):
        print(f"\nProcessing: {test_text}")
        result = pipeline.process_single(
            test_image, test_text,
            actual_label=1,
            use_retrieval=False
        )
        
        print(f"\nPrediction: {result.predicted_label} (actual: {result.actual_label})")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Correct: {result.is_correct()}")
        print(f"Selected tools: {[t.value for t in result.routing_plan.selected_tools]}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Processing time: {result.processing_time:.2f}s")
    else:
        print(f"Test image not found: {test_image}")
