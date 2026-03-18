This framework implements a comprehensive approach to zero-shot multimodal meme detection using CBR with retrieval augmentation and multi-tool cognitive analysis.

## Quick Start

```python
from framework import MemeDetectionPipeline

pipeline = MemeDetectionPipeline(
    dataset_name="FHM",
    model="gemini-flash",
    use_knowledge_base=True
)

result = pipeline.process_single(
    image_path="data/FHM/images/16395.png",
    text="handjobs sold seperately",
    actual_label=1
)

print(f"Prediction: {result.predicted_label}")
print(f"Correct: {result.is_correct()}")
print(f"Reasoning: {result.reasoning}")
```

```bash
python code/run_framework.py --mode main --dataset FHM --model gemini-flash
```

## File Structure

```
code/
├── __init__.py               # Module exports
├── config.py                 # Configuration file
├── case_base.py              # Case base module
├── tools.py                  # Eight cognitive tools
├── router.py                 # Intelligent router
├── adjudicator.py            # Dialectical adjudicator
├── pipeline.py               # Main pipeline
├── run_framework.py          # Entry point
├── prompts.py                # Prompt templates
├── README.md                 # Documentation
└── generate_explanations.py  # Explanation generation
utils/
└── ...
```

## API Configuration

Edit `config.py`:

```python
# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Set the API base URL
API_BASE_URL = "https://your-api-endpoint/v1"

# Set the default model
DEFAULT_MODEL = "gemini-flash"
```

## Output Format

```json
{
    "index": 0,
    "image_path": "data/FHM/images/16395.png",
    "text": "handjobs sold seperately",
    "actual": 1,
    "predicted": 1,
    "confidence": 0.85,
    "reasoning": "The meme uses sexual innuendo...",
    "selected_tools": ["sentiment_reversal", "target_identifier"],
    "key_evidence": ["Sexual content detected", "Targets women"],
    "core_contradiction": "Inappropriate sexualization",
    "processing_time": 3.45
}
```

## Dependencies

```
openai>=1.0.0
numpy>=1.26.0
scikit-learn>=1.3.0
tqdm>=4.66.0
Pillow>=10.0.0
```

## License

MIT License
