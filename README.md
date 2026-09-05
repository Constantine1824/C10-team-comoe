# TRI-AI-SLM

## Benchmark workflow

This project fine-tunes `google/medgemma-4b-it` with LoRA and generates concise answers for the 11-question medical advice benchmark. The checked-in retrieval baseline can run without the model and is written to `submissions/retrieval_baseline.csv`.

### Run the baseline

From the repository root:

```bash
python -m rag.generate_baseline
```

### Run fine-tuning

Use a CUDA machine or Google Colab with GPU runtime enabled. Use Python 3.11 or 3.12 for the training environment; this local Python 3.14 environment is not compatible with the required PyTorch stack. Install the dependencies and authenticate with Hugging Face so the gated MedGemma model can be downloaded:

```bash
pip install -r requirements.txt
python -m finetune.run_training
```

The trained adapter checkpoints are stored under `checkpoints/`, and the generated submission is written to `submissions/finetuned_submission.csv`.

Before submitting, compare the fine-tuned file with `data/baseline_submission.csv` and keep the version with the better validation evidence. The required submission columns are `QuestionId,Answer`, with exactly 11 non-empty rows.