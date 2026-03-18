# -*- coding: utf-8 -*-
"""
Generate AI Explanations for Training Data

This script generates explanations for why each meme in the training set 
is classified as harmful or harmless. These explanations will be used 
as part of the knowledge base for retrieval-augmented inference.

Output: Saves enhanced training data with explanations to:
    - data/{dataset}/train_with_explanations.jsonl
"""
import os
import sys
import json
import base64
from typing import Optional
from tqdm import tqdm

from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.config import API_BASE_URL, DATASET_CONFIGS, DEFAULT_API_KEY

API_KEY = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
BASE_URL = os.environ.get("OPENAI_BASE_URL", API_BASE_URL)


# Initialize client with extended timeout for image requests
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=300.0  
)


def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_explanation(
    image_path: str,
    text: str,
    label: int,
    dataset_name: str = "FHM",
    model: str = "gemini-2.0-flash"
) -> str:
    """
    Generate explanation for why a meme is harmful/harmless
    
    Args:
        image_path: Path to meme image
        text: Text content of the meme
        label: 0 = harmless, 1 = harmful
        dataset_name: Dataset name for label terminology
        model: Model to use for generation
        
    Returns:
        Explanation string
    """
    # Get label terminology for dataset
    config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["FHM"])
    if label == 1:
        label_str = config.get("positive_label", "harmful")
    else:
        label_str = config.get("negative_label", "harmless")
    
    prompt = f'''You are analyzing a meme to explain why it is classified as {label_str}.

## Meme Information:
- Text on meme: "{text}"
- Ground truth label: {label_str}

## Your Task:
Write a detailed explanation (3-4 sentences) covering:

1. **Image-Text Relationship**: Describe what the image shows and how it relates to the text.
2. **Semantic Analysis**: Is there irony, sarcasm, contradiction, or literal meaning?
3. **Harmful Elements** (if label is harmful): What makes it offensive? (e.g., targets a group, uses stereotypes, promotes hate)
4. **Why {label_str}**: Conclude why this combination results in a {label_str} classification.

Output ONLY the analysis directly, no prefix like "Analysis:" or "Explanation:".'''

    try:
        
        image_b64 = encode_image(image_path)
        
        
        if not image_b64 or len(image_b64) < 100:
            print(f"  Warning: Invalid Base64 encoding for {image_path}")
            return f"This meme is classified as {label_str} based on the combination of its visual and textual content."
        
        
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            mime_type = "image/png"
        elif ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        else:
            mime_type = "image/png"  
        
        
        image_url = f"data:{mime_type};base64,{image_b64}"
        
        
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }],
            temperature=0.0,
            max_tokens=300,
            stream=False  
        )
        
        
        result = response.choices[0].message.content.strip()
        
        
        if hasattr(response, 'usage') and response.usage:
            # 只在verbose模式下打印
            pass
        
        if not result:
            print(f"  Warning: Empty response for {image_path}")
            return f"This meme is classified as {label_str} based on the combination of its visual and textual content."
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"  Warning: API error for {image_path}: {error_msg}")
        
        
        if "503" in error_msg or "No available channels" in error_msg:
            print(f"    → Model {model} may not be available. Try another model.")
        
        # Fallback explanation
        return f"This meme is classified as {label_str} based on the combination of its visual and textual content."


def get_item_data(item: dict, dataset_name: str):
    """Extract image, text, label from data item"""
    config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["FHM"])
    
    image_filename = item.get(config["image_key"])
    text = item.get(config["text_key"])
    raw_label = item.get(config["label_key"])
    
    # Process label
    if dataset_name == "HarM":
        if isinstance(raw_label, list) and raw_label:
            label = 0 if raw_label[0].lower() == "not harmful" else 1
        else:
            label = 1
    else:
        label = raw_label if raw_label is not None else 0
        
    return image_filename, text, label


def generate_explanations_for_dataset(
    dataset_name: str,
    model: str = "gemini-2.0-flash",
    max_samples: Optional[int] = None,
    start_from: int = 0,
    output_suffix: str = "_with_explanations",
    parallel: bool = False,
    max_workers: int = 4
):
    """
    Generate explanations for all training samples in a dataset
    
    Args:
        dataset_name: Dataset name (FHM, HarM, MAMI)
        model: Model to use
        max_samples: Maximum samples to process (None = all)
        start_from: Start index for continuation
        output_suffix: Suffix for output file
        parallel: Enable parallel generation (faster but may hit rate limits)
        max_workers: Number of parallel workers
    """
    print(f"\n{'='*60}")
    print(f"Generating Explanations for {dataset_name}")
    print(f"Model: {model}")
    print(f"Parallel: {parallel} (workers: {max_workers})")
    print(f"{'='*60}\n")
    
    # Paths
    base_path = f"data/{dataset_name}"
    image_path = f"{base_path}/images"
    train_path = f"{base_path}/train.jsonl"
    output_path = f"{base_path}/train{output_suffix}.jsonl"
    
    # Load training data
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f.readlines()]
    
    print(f"Loaded {len(train_data)} training samples")
    
    # Check for existing output to continue
    existing_data = []
    if os.path.exists(output_path) and start_from == 0:
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = [json.loads(line) for line in f.readlines()]
        start_from = len(existing_data)
        print(f"Found existing output with {start_from} samples, continuing...")
    
    # Limit samples if specified
    if max_samples:
        end_idx = min(start_from + max_samples, len(train_data))
    else:
        end_idx = len(train_data)
    
    # Prepare items to process
    items_to_process = []
    for idx in range(start_from, end_idx):
        item = train_data[idx]
        image_filename, text, label = get_item_data(item, dataset_name)
        
        if not image_filename or not text:
            continue
        
        full_image_path = os.path.join(image_path, image_filename)
        if not os.path.exists(full_image_path):
            continue
        
        items_to_process.append({
            "idx": idx,
            "item": item,
            "image_path": full_image_path,
            "text": text,
            "label": label
        })
    
    print(f"Items to process: {len(items_to_process)}")
    
    # Process function for single item
    def process_single(item_data):
        explanation = generate_explanation(
            item_data["image_path"],
            item_data["text"],
            item_data["label"],
            dataset_name=dataset_name,
            model=model
        )
        enhanced_item = item_data["item"].copy()
        enhanced_item["explanation"] = explanation
        enhanced_item["train_index"] = item_data["idx"]
        return enhanced_item
    
    # Process samples
    mode = 'a' if start_from > 0 else 'w'
    results = []
    
    if parallel:
        # Parallel processing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single, item): item["idx"] 
                      for item in items_to_process}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Generating (parallel)"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    print(f"Error processing {idx}: {e}")
        
        # Sort by index and save
        results.sort(key=lambda x: x["train_index"])
        with open(output_path, mode, encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
    else:
        # Sequential processing
        with open(output_path, mode, encoding='utf-8') as f:
            for item_data in tqdm(items_to_process, desc="Generating explanations"):
                try:
                    enhanced_item = process_single(item_data)
                    json.dump(enhanced_item, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()
                except Exception as e:
                    print(f"Error processing {item_data['idx']}: {e}")
    
    print(f"\nCompleted! Output saved to: {output_path}")
    print(f"Total samples processed: {len(items_to_process)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AI explanations for training data")
    parser.add_argument("--dataset", type=str, default="FHM", 
                       choices=["FHM", "HarM", "MAMI"],
                       help="Dataset to process")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                       help="Model to use for generation")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to process")
    parser.add_argument("--start_from", type=int, default=0,
                       help="Start index (for manual continuation)")
    parser.add_argument("--parallel", action="store_true",
                       help="Enable parallel generation (faster)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    generate_explanations_for_dataset(
        dataset_name=args.dataset,
        model=args.model,
        max_samples=args.max_samples,
        start_from=args.start_from,
        parallel=args.parallel,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
