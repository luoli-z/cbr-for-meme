# Retrieving Precedents, Adapting Tool Plans, and Revising Judgments: Case-Based Reasoning for Zero-shot Harmful Meme Detection

This repository contains code, datasets, and prompts related to the paper titled "Retrieving Precedents, Adapting Tool Plans, and Revising Judgments: Case-Based Reasoning for Zero-shot Harmful Meme Detection". 

## Keywords

CBR, RAG, Harmful Meme Detection, LLM Agents, Tool Adaptation

## Repository Structure

- `framework/`: This directory contains the codebase for implementing / evaluating RAMTA.
- `data/`: This directory contains datasets utilized / generated in the experiments mentioned in the paper.
- `results/`: This directory contains the main experimental results for RAMTA.
- `utils/`: This directory contains utility scripts and supporting resources, including prompt templates and data processing helpers.
- `README.md`: This file provides an overview of the repository.
## Prompts
All detailed prompt templates are available in `utils/prompts.py` and `framework/prompts.py`.

## Efficiency Analysis

A common critique of multi-agent frameworks is their high latency and computational overhead. To address this, we evaluate the inference efficiency of RAMTA against representative baselines. Fig.5 illustrates the average Time Consumption (seconds) and Token Consumption (thousands of tokens) per sample on the HarM dataset using a single NVIDIA RTX 4090.

**Empirical Efficiency via Prior Guidance and Parallelization.** Single-pass direct inference models (e.g., InstructBLIP, LLaVA-1.5) are inherently fast ($\le$2.1s) and token-efficient ($\le$1.2k) but struggle to decode complex memes. Proprietary APIs like GPT-4o (CoT) offer moderate latency (5.5s) and token usage (2.5k) but lack structured multi-view reasoning capabilities. Conversely, multi-agent frameworks naturally consume more resources. For instance, the multi-agent baseline MIND requires 8.4s and 3.5k tokens.

RAMTA effectively mitigates this multi-agent overhead through two structural optimizations. First, explicit prior guidance prevents verbose, unguided CoT reasoning, effectively capping token usage for the decoupled experts at 4.1k. Second, asynchronous parallel execution of these agents (RAMTA (Ours)) slashes sequential latency (RAMTA (Seq)) from 14.5s to 7.8s.

Despite executing a rigorous three-stage reasoning pipeline, RAMTA maintains competitive latency, actually slightly outperforming MIND (7.8s vs. 8.4s). Overall, the consistent performance gains achieved by RAMTA strongly justify this moderate resource trade-off, demonstrating its practicality for real-world deployment.

<p align="center">
  <img src="assets/efficiency.png" alt="Empirical efficiency comparison on the HarM dataset" width="700">
</p>


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
python framework/run_framework.py --mode main --dataset FHM --model gemini-flash
```

For more implementation details, please refer to `framework/README.md`.
