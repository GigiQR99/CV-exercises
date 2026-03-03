# HuggingFace Models Review 

This repo includes a Google Colab notebook that demonstrates how to use Hugging Face **Transformers** (and `diffusers`) with a consistent `pipeline()` workflow across many tasks.  Run multiple pretrained models across NLP, vision, and audio tasks with minimal code changes.

## What this notebook does
- Installs key libraries/dependencies  (e.g., `transformers`, `torch`, `accelerate`, `sentencepiece`) and shows optional Hugging Face token login for accessing gated resources.
- Runs a sequence of examples (“Model 1 → Model X”), each with a progress bar, short explanation, code, and printed outputs.
- Includes notes about what gets downloaded/loaded when you run a model (config, tokenizer files, model weights) and and typical warnings.

## Models & tasks covered
1. **Sentiment analysis** — `distilbert-base-uncased-finetuned-sst-2-english`
2. **Zero-shot text classification** — `facebook/bart-large-mnli` (includes a medical-specialty example).
3. **Named Entity Recognition (NER)** — `dslim/bert-base-NER`
4. **Question answering** — `deepset/roberta-base-squad2`
5. **Text generation** — `gpt2`
6. **Translation (Seq2Seq via generate, not pipeline)** — `Helsinki-NLP/opus-mt-en-fr` (and/or OPUS-MT variants used in the notebook)
7. **Summarization (Seq2Seq)** — `facebook/bart-large-cnn`
8. **Image classification (ViT)** — `google/vit-base-patch16-224`
9. **Object detection (DETR + ResNet-50 backbone)** — `facebook/detr-resnet-50`
10. **Image classification (ResNet-50)** — `microsoft/resnet-50`
11. **Speech-to-text (ASR)** — `openai/whisper-base`.

## How to run
- In colab, add your Hugging Face access token (set HF_TOKEN=hf_...) so the app can authenticate to the Hub; alternatively run hf auth login (or huggingface-cli login) to store the token locally.
- GPU recommended for faster inference on larger models
