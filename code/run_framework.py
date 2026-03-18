# -*- coding: utf-8 -*-
"""
Run Framework: Entry point for Retrieval-Augmented Multi-Tools Meme Detection

This script provides interfaces for:
1. Running main experiments
2. Running ablation studies
3. Running parameter sensitivity analysis
4. Running LLM robustness experiments
5. Running efficiency analysis
"""
import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Optional, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.config import (
    FrameworkConfig, AVAILABLE_MODELS, DATASET_CONFIGS,
    DEFAULT_MODEL, DEFAULT_FRAMEWORK_CONFIG
)
from framework.pipeline import MemeDetectionPipeline, AblationPipeline
from framework.tools import ToolType
from framework.knowledge_base import KnowledgeBase, verify_knowledge_base


def run_main_experiment(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    max_samples: Optional[int] = None,
    output_dir: str = "results",
    use_retrieval: bool = True,
    top_k: int = 5
):
    """
    Run main experiment on a dataset
    
    Args:
        dataset_name: Dataset to evaluate (FHM, HarM, MAMI)
        model: Model to use
        max_samples: Maximum samples to process
        output_dir: Output directory for results
        use_retrieval: Whether to use retrieval augmentation
        top_k: Number of samples to retrieve
    """
    print(f"\n{'='*60}")
    print(f"Running Main Experiment")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model}")
    print(f"Use Retrieval: {use_retrieval}")
    print(f"Top-K: {top_k}")
    print(f"{'='*60}\n")
    
    config = FrameworkConfig(
        top_k_retrieval=top_k,
        llm_model=model,
        vision_model=model
    )
    
    pipeline = MemeDetectionPipeline(
        dataset_name=dataset_name,
        config=config,
        model=model,
        use_knowledge_base=use_retrieval,
        preload_knowledge_base=use_retrieval
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir, dataset_name,
        f"main_{model}_{timestamp}.jsonl"
    )
    
    summary = pipeline.process_dataset(
        max_samples=max_samples,
        output_path=output_path,
        use_retrieval=use_retrieval
    )
    
    return summary


def run_ablation_study(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    max_samples: Optional[int] = None,
    output_dir: str = "results"
):
    """
    Run ablation study to analyze component contributions
    
    Configurations tested:
    1. Full model (all components)
    2. No retrieval (disable knowledge base)
    3. No routing (use all tools instead of selective routing)
    4. No retrieval + No routing
    5. Single tool baselines
    """
    print(f"\n{'='*60}")
    print(f"Running Ablation Study")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configuration 1: Full model
    print("\n--- Ablation 1: Full Model ---")
    pipeline = AblationPipeline(
        dataset_name=dataset_name,
        model=model,
        use_knowledge_base=True,
        use_routing=True
    )
    results["full"] = pipeline.process_dataset(
        max_samples=max_samples,
        output_path=os.path.join(output_dir, dataset_name, f"ablation_full_{timestamp}.jsonl")
    )
    
    # Configuration 2: No retrieval
    print("\n--- Ablation 2: No Retrieval ---")
    pipeline = AblationPipeline(
        dataset_name=dataset_name,
        model=model,
        use_knowledge_base=False,
        use_routing=True
    )
    results["no_retrieval"] = pipeline.process_dataset(
        max_samples=max_samples,
        output_path=os.path.join(output_dir, dataset_name, f"ablation_no_retrieval_{timestamp}.jsonl")
    )
    
    # Configuration 3: No routing (use all tools)
    print("\n--- Ablation 3: No Routing (All Tools) ---")
    pipeline = AblationPipeline(
        dataset_name=dataset_name,
        model=model,
        use_knowledge_base=True,
        use_routing=False,
        use_all_tools=True
    )
    results["no_routing"] = pipeline.process_dataset(
        max_samples=max_samples,
        output_path=os.path.join(output_dir, dataset_name, f"ablation_no_routing_{timestamp}.jsonl")
    )
    
    # Configuration 4: Minimal (no retrieval, no routing)
    print("\n--- Ablation 4: Minimal (No Retrieval, Default Tools) ---")
    pipeline = AblationPipeline(
        dataset_name=dataset_name,
        model=model,
        use_knowledge_base=False,
        use_routing=False
    )
    results["minimal"] = pipeline.process_dataset(
        max_samples=max_samples,
        output_path=os.path.join(output_dir, dataset_name, f"ablation_minimal_{timestamp}.jsonl")
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("Ablation Study Summary")
    print(f"{'='*60}")
    for config_name, summary in results.items():
        print(f"{config_name}: Acc={summary['accuracy']:.4f}, F1={summary['macro_f1']:.4f}")
    
    # Save summary
    summary_path = os.path.join(output_dir, dataset_name, f"ablation_summary_{timestamp}.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_retrieval_sensitivity(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    k_values: List[int] = [1, 3, 5, 7, 10],
    max_samples: Optional[int] = None,
    output_dir: str = "results"
):
    """
    Run parameter sensitivity analysis for retrieval k value
    """
    print(f"\n{'='*60}")
    print(f"Running Retrieval Sensitivity Analysis")
    print(f"Dataset: {dataset_name}")
    print(f"K values: {k_values}")
    print(f"{'='*60}\n")
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for k in k_values:
        print(f"\n--- Testing k={k} ---")
        
        config = FrameworkConfig(top_k_retrieval=k)
        pipeline = MemeDetectionPipeline(
            dataset_name=dataset_name,
            config=config,
            model=model,
            use_knowledge_base=True
        )
        
        summary = pipeline.process_dataset(
            max_samples=max_samples,
            output_path=os.path.join(output_dir, dataset_name, f"sensitivity_k{k}_{timestamp}.jsonl")
        )
        results[f"k={k}"] = summary
    
    # Print summary
    print(f"\n{'='*60}")
    print("Retrieval Sensitivity Summary")
    print(f"{'='*60}")
    for k_name, summary in results.items():
        print(f"{k_name}: Acc={summary['accuracy']:.4f}, F1={summary['macro_f1']:.4f}")
    
    return results


def run_tool_sensitivity(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    tool_counts: List[int] = [1, 2, 3, 4, 5, 6],
    max_samples: Optional[int] = None,
    output_dir: str = "results"
):
    """
    Run parameter sensitivity analysis for number of tools selected
    """
    print(f"\n{'='*60}")
    print(f"Running Tool Count Sensitivity Analysis")
    print(f"Dataset: {dataset_name}")
    print(f"Tool counts: {tool_counts}")
    print(f"{'='*60}\n")
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for n_tools in tool_counts:
        print(f"\n--- Testing n_tools={n_tools} ---")
        
        config = FrameworkConfig(
            max_tools_to_select=n_tools,
            min_tools_to_select=min(2, n_tools)
        )
        pipeline = MemeDetectionPipeline(
            dataset_name=dataset_name,
            config=config,
            model=model,
            use_knowledge_base=True
        )
        
        summary = pipeline.process_dataset(
            max_samples=max_samples,
            output_path=os.path.join(output_dir, dataset_name, f"sensitivity_tools{n_tools}_{timestamp}.jsonl")
        )
        results[f"n_tools={n_tools}"] = summary
    
    print(f"\n{'='*60}")
    print("Tool Count Sensitivity Summary")
    print(f"{'='*60}")
    for name, summary in results.items():
        print(f"{name}: Acc={summary['accuracy']:.4f}, F1={summary['macro_f1']:.4f}")
    
    return results


def run_llm_robustness(
    dataset_name: str,
    models: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    output_dir: str = "results"
):
    """
    Run LLM robustness experiment with different model sizes
    """
    if models is None:
        models = ["gemini-flash", "gpt-4o-mini", "gpt-4o", "qwen-plus"]
    
    print(f"\n{'='*60}")
    print(f"Running LLM Robustness Experiment")
    print(f"Dataset: {dataset_name}")
    print(f"Models: {models}")
    print(f"{'='*60}\n")
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model in models:
        if model not in AVAILABLE_MODELS:
            print(f"Warning: Model {model} not configured, skipping")
            continue
            
        print(f"\n--- Testing model: {model} ---")
        
        try:
            pipeline = MemeDetectionPipeline(
                dataset_name=dataset_name,
                model=model,
                use_knowledge_base=True
            )
            
            summary = pipeline.process_dataset(
                max_samples=max_samples,
                output_path=os.path.join(output_dir, dataset_name, f"llm_{model}_{timestamp}.jsonl")
            )
            results[model] = summary
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("LLM Robustness Summary")
    print(f"{'='*60}")
    for model_name, summary in results.items():
        print(f"{model_name}: Acc={summary['accuracy']:.4f}, F1={summary['macro_f1']:.4f}, Time={summary['average_processing_time']:.2f}s")
    
    return results


def run_efficiency_analysis(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    max_samples: int = 50,
    output_dir: str = "results"
):
    """
    Run efficiency analysis comparing parallel vs sequential tool execution
    """
    print(f"\n{'='*60}")
    print(f"Running Efficiency Analysis")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test 1: Parallel execution
    print("\n--- Testing Parallel Execution ---")
    config_parallel = FrameworkConfig(parallel_tool_execution=True, max_workers=4)
    pipeline_parallel = AblationPipeline(
        dataset_name=dataset_name,
        config=config_parallel,
        model=model,
        use_knowledge_base=False,  # Skip KB for pure efficiency test
        use_routing=False,
        use_all_tools=True  # Use all tools to maximize parallel benefit
    )
    results["parallel"] = pipeline_parallel.process_dataset(
        max_samples=max_samples,
        output_path=os.path.join(output_dir, dataset_name, f"efficiency_parallel_{timestamp}.jsonl")
    )
    
    # Test 2: Sequential execution
    print("\n--- Testing Sequential Execution ---")
    config_sequential = FrameworkConfig(parallel_tool_execution=False)
    pipeline_sequential = AblationPipeline(
        dataset_name=dataset_name,
        config=config_sequential,
        model=model,
        use_knowledge_base=False,
        use_routing=False,
        use_all_tools=True
    )
    results["sequential"] = pipeline_sequential.process_dataset(
        max_samples=max_samples,
        output_path=os.path.join(output_dir, dataset_name, f"efficiency_sequential_{timestamp}.jsonl")
    )
    
    # Calculate speedup
    parallel_time = results["parallel"]["average_processing_time"]
    sequential_time = results["sequential"]["average_processing_time"]
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    
    print(f"\n{'='*60}")
    print("Efficiency Analysis Summary")
    print(f"{'='*60}")
    print(f"Parallel: {parallel_time:.2f}s avg")
    print(f"Sequential: {sequential_time:.2f}s avg")
    print(f"Speedup: {speedup:.2f}x")
    
    results["speedup"] = speedup
    
    return results


def run_context_awareness_analysis(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    max_samples: Optional[int] = None,
    output_dir: str = "results"
):
    """
    Run context awareness analysis showing how retrieval helps with novel patterns
    
    This demonstrates the framework's ability to quickly adapt to new meme patterns
    by adding them to the knowledge base.
    """
    print(f"\n{'='*60}")
    print(f"Running Context Awareness Analysis")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}\n")
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test 1: Without retrieval (baseline)
    print("\n--- Baseline: Without Retrieval ---")
    pipeline_no_retrieval = MemeDetectionPipeline(
        dataset_name=dataset_name,
        model=model,
        use_knowledge_base=False
    )
    results["no_retrieval"] = pipeline_no_retrieval.process_dataset(
        max_samples=max_samples,
        output_path=os.path.join(output_dir, dataset_name, f"context_no_retrieval_{timestamp}.jsonl"),
        use_retrieval=False
    )
    
    # Test 2: With retrieval (full knowledge base)
    print("\n--- With Full Knowledge Base ---")
    pipeline_with_kb = MemeDetectionPipeline(
        dataset_name=dataset_name,
        model=model,
        use_knowledge_base=True
    )
    results["with_retrieval"] = pipeline_with_kb.process_dataset(
        max_samples=max_samples,
        output_path=os.path.join(output_dir, dataset_name, f"context_with_retrieval_{timestamp}.jsonl"),
        use_retrieval=True
    )
    
    # Calculate improvement
    no_ret_acc = results["no_retrieval"]["accuracy"]
    with_ret_acc = results["with_retrieval"]["accuracy"]
    improvement = with_ret_acc - no_ret_acc
    
    print(f"\n{'='*60}")
    print("Context Awareness Summary")
    print(f"{'='*60}")
    print(f"Without Retrieval: Acc={no_ret_acc:.4f}")
    print(f"With Retrieval: Acc={with_ret_acc:.4f}")
    print(f"Improvement: +{improvement:.4f} ({improvement/no_ret_acc*100:.1f}%)")
    
    results["improvement"] = improvement
    
    return results


def build_knowledge_bases(datasets: List[str] = ["FHM", "HarM", "MAMI"]):
    """
    Verify knowledge bases for all datasets
    (Knowledge bases are loaded from SSR results and training data with explanations)
    """
    print("Verifying knowledge bases for all datasets...")
    
    for dataset in datasets:
        print(f"\n--- Verifying KB for {dataset} ---")
        try:
            # Verify setup
            if verify_knowledge_base(dataset):
                # Load and show statistics
                kb = KnowledgeBase(dataset)
                kb.load(require_explanations=False)
                stats = kb.get_statistics()
                print(f"✓ Knowledge base ready: {stats['num_train_samples']} train samples, "
                      f"{stats['num_test_samples_with_ssr']} test samples with SSR")
            else:
                print(f"✗ Knowledge base not ready for {dataset}")
        except Exception as e:
            print(f"Error verifying KB for {dataset}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval-Augmented Multi-Tools Framework for Meme Detection"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="main",
        choices=["main", "ablation", "retrieval_sensitivity", "tool_sensitivity", 
                 "llm_robustness", "efficiency", "context_awareness", "build_kb"],
        help="Experiment mode to run"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="FHM",
        choices=["FHM", "HarM", "MAMI"],
        help="Dataset to use"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-flash",
        help="Model to use"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--no_retrieval",
        action="store_true",
        help="Disable retrieval augmentation"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of samples to retrieve"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "main":
        run_main_experiment(
            dataset_name=args.dataset,
            model=args.model,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            use_retrieval=not args.no_retrieval,
            top_k=args.top_k
        )
    
    elif args.mode == "ablation":
        run_ablation_study(
            dataset_name=args.dataset,
            model=args.model,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
    
    elif args.mode == "retrieval_sensitivity":
        run_retrieval_sensitivity(
            dataset_name=args.dataset,
            model=args.model,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
    
    elif args.mode == "tool_sensitivity":
        run_tool_sensitivity(
            dataset_name=args.dataset,
            model=args.model,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
    
    elif args.mode == "llm_robustness":
        run_llm_robustness(
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
    
    elif args.mode == "efficiency":
        run_efficiency_analysis(
            dataset_name=args.dataset,
            model=args.model,
            max_samples=args.max_samples or 50,
            output_dir=args.output_dir
        )
    
    elif args.mode == "context_awareness":
        run_context_awareness_analysis(
            dataset_name=args.dataset,
            model=args.model,
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )
    
    elif args.mode == "build_kb":
        build_knowledge_bases()


if __name__ == "__main__":
    main()
