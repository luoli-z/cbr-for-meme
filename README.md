# Retrieving Precedents, Adapting Tool Plans, and Revising Judgments: Case-Based Reasoning for Zero-shot Harmful Meme Detection

This repository contains code, datasets, and prompts related to the paper titled "Retrieving Precedents, Adapting Tool Plans, and Revising Judgments: Case-Based Reasoning for Zero-shot Harmful Meme Detection". 

## Keywords

CBR, RAG, Harmful Meme Detection, LLM Agents, Tool Adaptation

## Repository Structure

- `code/`: This directory contains the codebase for implementing / evaluating RAMTA.
- `data/`: This directory contains datasets utilized / generated in the experiments mentioned in the paper.
- `results/`: This directory contains the main experimental results for RAMTA.
- `utils/`: This directory contains utility scripts and supporting resources, including prompt templates and data processing helpers.
- `README.md`: This file provides an overview of the repository.
## Prompts
All detailed prompt templates are available in `utils/prompts.py` and `code/prompts.py`.
## Quick Start

1. Prepare the datasets.  
   Please obtain FHM, HarM, and MAMI, and place them in the following directories:

```text
MIND/
├── data/
│   ├── FHM/
│   │   ├── images/
│   │   │   └── ...
│   │   ├── test.jsonl
│   │   └── train.jsonl
│   ├── HarM/
│   │   ├── images/
│   │   │   └── ...
│   │   ├── test.jsonl
│   │   └── train.jsonl
│   └── MAMI/
│       ├── images/
│       │   └── ...
│       ├── test.jsonl
│       └── train.jsonl
└── ...
```
2. Configure the API.  
   Open `config.py` and set your API key, API base URL, and default model.

3. Run the framework.

```bash
python code/run_framework.py --mode main --dataset FHM --model gemini-flash
```

For more implementation details, please refer to `code/README.md`.
