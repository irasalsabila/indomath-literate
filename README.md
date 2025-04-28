# Self-Revision in Multilingual Math Reasoning

## Overview

**Can Self-Revision Improve Mathematical Reasoning Accuracy in Multilingal Languages?**

This project investigates whether **self-revision** — allowing a language model to revise its own generated answers — can improve language accuracy and reasoning correctness in **math questions**, especially across low-resource languages such as: Bengali (bn), Telugu (te), Thai (th), Indonesian (id), Javanese (jv), Sundanese (su).

We evaluate different large language models (LLMs) by:
- Generating answers in multiple languages
- Allowing the model to revise its own outputs
- Rechecking and comparing the correctness before and after revision

## Project Structure

```bash
Project Root/
├── logs/
├── mgsm/ -> for the original mgsm dataset
├── mgsm_new/ -> mgsm for ind, jv, su
├── backtranslation.ipynb
├── error.ipynb
├── inference.py
├── recheck_extraction.ipynb
├── README.md
```

## How to Run Inference

General Command Format:

```bash
python inference.py --model_name "<model-name>" \
                    --max_waits 5 \
                    --num_questions 50 \
                    --languages <lang-codes> \
                    --results_dir "<results-folder>/" \
                    --api_key_file "<path-to-api-key-file>" \
                    --display_tables
```

Example Runs:
- `OpenAI GPT-4o` → openai/chatgpt-4o-latest
- `Cohere Command R+` → cohere/command-r-plus-08-2024
- `Claude 3.5 Haiku` → anthropic/claude-3.5-haiku
- `DeepSeek R1` → deepseek/deepseek-r1
- `Google Gemma 3 27B` → google/gemma-3-27b-it
- `--languages` (e.g., id jv su, bn en es te th ja)
- `--results_dir` to your output folder
- `--api_key_file` to your key path


## Available Notebooks
- `backtranslation.ipynb`: Evaluate backtranslation quality (multilingual → English).
- `error.ipynb`: Classify model revision mistakes (Arithmetic, Missing Calculation, Hallucination).
- `recheck_extraction.ipynb`: Re-extract numerical answers and recheck model outputs.

## Notes

- Store your API keys properly in files like `apis/api.txt`.
- Adjust the results_dir and language codes depending on your experiment needs.
- Main low-resource focus: **Bengali, Telugu, Thai, Indonesian, Javanese, Sundanese**.

## Project Link
[GitHub Repository](https://github.com/irasalsabila/indomath-literate)