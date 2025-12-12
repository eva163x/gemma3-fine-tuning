# Overview

This project fine-tunes the Gemma-3-270M model on the MATH dataset using LoRA.
All training, evaluation, and ablation studies are contained in the Jupyter notebook:

Gemma3_v2.ipynb


## The notebook includes:

- dataset preprocessing
- model loading
- training loop
- evaluation pipeline
- checkpoint comparison
- LoRA rank ablation study

No additional .py scripts need to be executed.

## Hardware Requirements

To run training end-to-end, you need one of the following:

Option A (Recommended)

Google Colab Pro or Colab Free with a GPU runtime (T4, L4, A100 all work).

Option B

A local machine with:

- CUDA-compatible GPU (12 GB+ VRAM recommended)
- Python 3.10+
- PyTorch with CUDA enabled

If you do not have access to a GPU, you can still run the evaluation cells, but training will be too slow.

## Environment Setup

- Clone the Repository

git clone eva163x/gemma3-fine-tuning
cd gemma3-fine-tuning


- If running on Colab:
Simply upload the repository and skip environment installation—Colab will install dependencies automatically.

## Running the Project
1. Open the Notebook
jupyter notebook Gemma3_v2.ipynb

Or on Colab (recommended):
- Open Google Colab
- Upload the Gemma3_v2.ipynb file
- Run all cells sequentially

2. Follow Notebook Sections
