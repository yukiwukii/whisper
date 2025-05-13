# whisper
This is a repository for my experimentations with OpenAI's Whisper model.

## Description
In this quick experiment, I finetuned the [original small Whisper model](https://huggingface.co/openai/whisper-small) using the widely used [common voice dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0), specifically the Indonesian split. All hyperparameters can be found in the script `./script/train.py`.

Main learning points:
1. Decoding / inference using Whisper models with HuggingFace endpoints.
2. Finetuning pipeline for Whisper models with available dataset, resulting in 71% reduction in loss.
3. Greater clarity of the architecture of Whisper models.

Limitations:
- Used the train/test split from the same dataset for finetuning, which may result in inflated accuracy.
- Need to write a proper `inference.py` code for usage.

## Results
| **Model Type** | **Test Loss** | **File**
| --- | --- | --- |
| Original whisper model | 1.12 | `whisper-no-finetune/all_results.json` |
| Finetuned whisper | 0.32 | `whisper-finetuned/all_results.json` |