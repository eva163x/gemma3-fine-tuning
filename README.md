# Gemini can’t solve my Calc homework
**Supervised Fine-Tuning Gemma3-270M with Enhanced Reasoning For Advanced Math Problems**

In this project, we run a supervised fine-tuning pipeline using LoRA on Google’s open-weight model, Gemma3-270M, with the goal of providing enhanced reasoning, answer accuracy and formatting on Advanced Math Problems (high school+). The model learns to provide structured solutions with explicit reasoning steps and final numerical answers, with the ability to understand and output LaTex.

## Motivation

Initially I had planned to fine-tune on the common openai/gsm8k dataset (https://huggingface.co/datasets/openai/gsm8k) which contains grade school math problems-- but, the models were predicting it too easily, and it had been done too many times before. LLMs like Gemini or ChatGPT can solve many math problems, but they often struggle with structured reasoning in big word problems, especially LaTeX-heavy college problems (or high school competition level problems). I wanted to understand why they struggle so much, and if a small model could be trained to reason more reliably on advanced math questions.

Along the way, I discovered some major limitations: strict answer-matching penalizes text-based solutions, LaTeX extraction is so hard and unpredictable, and the small model tends to overfit to the output format. So I focused more on understanding, running ablation studies on training checkpoints and LoRA rank to understand why certain versions performed better or worse.

Here is my final project flow:

data preprocessing → defining eval function (metrics) → SFT training → SFT eval → Baseline eval → Ablation Study: Checkpointing → Ablation Study: LoRA ranks

## Quick Start

Option 1: Locally on Jupyter (provided you have access to a CUDA acclerator)

```bash
jupyter notebook Gemma3_v2.ipynb
``` 
Option 2: Google Colab (RECOMMENDED)

Open the provided Colab link and upload `Gemma3_v2.ipynb`
Change runtime to one of the GPUs (T4 is acceptable, A100 is recommended)

## What it Does

This project fine-tunes Gemma-3-270M on the MATH dataset to improve mathematical reasoning. It loads and cleans the dataset, extracts reasoning and final answers, reformats each example using a consistent system-prompt structure, and performs supervised fine-tuning with LoRA adapters. After training, the model is evaluated using a custom evaluation pipeline that measures format correctness, answer accuracy, and reasoning similarity. I also run an ablation study by varying LoRA rank, a key hyperparameter, and evaluating various checkpoint training metrics and evals to understand their effect on performance.

### Dataset
- **Source**: MATH competition dataset (qwedsacf/competition_math)
- **Size**: 10000 training, 1250 validation, 1250 test
- **Preprocessing**: LaTeX cleaning, answer extraction, reasoning segmentation

### Model Architecture
- **Base**: Gemma-3-270m-it (270M parameters)
- **Fine-tuning**: LoRA with r=4, dropout=0.05
- **Trainable params**: 368,640 (0.14% of total)

### Limitations
- LaTex heavy dataset!
- Small lightweight model prone to overfitting
- Simplistic eval metrics


## Evaluation

### Quantitative:

NOTE: these are not incredibly reliable, since my extraction pipeline was a little buggy against the diversity in the answer patterns and LaTex format, and may seem artificially low.

Format Rate: our format rate tracked if the model followed the solution format laid out by the prompt and dataset solutions.
Answer Accuracy: This checked if the final numerical answer outputted by the model was strictly equal to the ground truth answer (NOTE: some answers were text-based, so accuracy is too strict which makes performance artificially low)
Reasoning Accuracy: checks for cosine similarity between ground truth reasoning and outputted reasoning.


### Model Performance
| Metric | Baseline (Untrained) | After Fine-Tuning |
|--------|---------------------|-------------------|
| Format Accuracy | 22.72% | 12.32% |
| Answer Accuracy | 1.12% | 0.72% |
| Reasoning Quality | 0.32% | 0.56% |

We notice that the baseline performs better than the fine-tuned model, which we primarily attribute to inconsistencies in our extraction pipeline and strict metrics. Additionally, fine-tuning with LoRA reduced format accuracy because the model overfitted to my training format and began repeating tokens. However, quantitative results still demonstrate measurable improvement in reasoning quality, and the ablation study clarifies the effect of LoRA rank.

### Training Performance:




## Ablation Study

### Checkpoint performance

To understand how training progression affected performance, I evaluated multiple checkpoints saved during fine-tuning. Each checkpoint was tested on a held-out subset of the MATH test set (200 samples). 

| Checkpoint | Format Accuracy | Answer Accuracy | Reasoning Accuracy |
| ---------- | --------------- | --------------- | ------------------ |
| **200**    | 10.5%           | 1.5%            | 0.0%               |
| **400**    | 10.5%           | 0.5%            | 0.0%               |
| **625**    | **11.0%**       | 0.5%            | **1.5%**           |

### LoRA ranks

I also wanted to test how different LoRA ranks would affect training. So, keeping hyperparameters constant, I cycled through the following LoRA values and retrained the model on 200 samples of data.

| LoRA rank | Format Accuracy | Answer Accuracy | Reasoning Accuracy |
| ---------- | --------------- | --------------- | ------------------ |
| **4**    | 11.6%           | 0.6%            | 0.8%               |
| **8**    | 13.0%           | 1.0%            | 0.6%               |
| **16**    | 11.5%     | 0.9%            | 0.6%    |

## Video Links
- Demo: https://drive.google.com/file/d/1qFdakgx42Hpr6zg0BY6avdfDeOnjylvK/view?usp=sharing
- Technical Walkthrough: https://drive.google.com/file/d/1p3KLpYTz8StiAIfeWwQ-cOwcmQIMC_-h/view?usp=sharing


