# Retrieving Precedents, Adapting Tool Plans, and Revising Judgments: Case-Based Reasoning for Zero-shot Harmful Meme Detection

This repository contains code, datasets, and prompts related to the paper titled "Retrieving Precedents, Adapting Tool Plans, and Revising Judgments: Case-Based Reasoning for Zero-shot Harmful Meme Detection". 

## Keywords

CBR, RAG, Harmful Meme Detection, LLM Agents, Tool Adaptation

## Repository Structure

- `code/`: This directory contains the codebase for implementing / evaluating CBR-RAG.
- `data/`: This directory contains datasets utilized / generated in the experiments mentioned in the paper.
- `results/`: This directory contains the main experimental results for RAMTA.
- `README.md`: This file provides an overview of the repository.

## Quick Start
Before running the framework:

1. Prepare the dataset.  
   Download the required harmful meme detection dataset (e.g., FHM, HarM, or MAMI) and place the images and annotations in the corresponding `data/` directory.

2. Configure the API.  
   Open `config.py` and set your API key, API base URL, and default model.

3. Run the framework.

```bash
python code/run_framework.py --mode main --dataset FHM --model gemini-flash
