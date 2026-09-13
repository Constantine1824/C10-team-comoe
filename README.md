# TRI-AI-SLM: Domain Specific Small Language Model for medical advice

TRI-AI-SLM is an educational medical question-answering system. It retrieves supporting public-health guidance, uses metadata to preserve context, and generates concise answers with safety instructions. It is not a clinical tool and does not replace a licensed clinician.

## Dataset

The repository contains 24 synthetic public-health factsheets in `data/documents.csv`. Each row has a document ID, title, guidance text, topic, care setting, population, origin, source URL, and license. The factsheets were created for this benchmark and are labeled synthetic/CC0 in the data; they are not a clinical database.

`data/train_qa.csv` contains 43 labeled question-answer pairs. Each pair includes topic, care setting, population, a supporting `document_id`, and a reference answer. `data/test_questions.csv` contains 11 held-out questions without answers. `data/sample_submission.csv` defines the required `QuestionId,Answer` format, and `data/dataset-metadata.json` records the benchmark description and mean Levenshtein scoring rule.

## Training Pipeline

The active implementation is under `scripts/`. Documents are loaded from CSV and split into sentence windows of at most 600 characters with one-sentence overlap; metadata is prepended to each embedded chunk. The chunking notebook also documents the source-row serialization approach for these short factsheets.

For each question, retrieval first restricts candidates by exact topic, care setting, and population, then falls back to topic matches and finally the full corpus. The RAG retriever combines `BAAI/bge-small-en-v1.5` dense embeddings with unigram/bigram TF-IDF using reciprocal-rank fusion. A TF-IDF embedder is available when the dense model cannot be loaded. Training uses the gold document as context; inference uses the top three retrieved chunks, with the same prompt formatting in both paths.

The Kaggle notebook (`scripts/kaggle_main.ipynb`) logs into Hugging Face, loads the repository, prepares train/test prompts, fine-tunes `google/medgemma-1.5-4b-it` with PEFT LoRA, generates 11 answers, removes a leading `Answer:` label, and writes `/kaggle/working/submission.csv`. The checked-in LoRA settings are rank 16, alpha 16, dropout 0.05, all linear target modules, no bias adaptation, 4-bit NF4 quantization, learning rate `2e-4`, batch size 4, gradient accumulation 4, gradient checkpointing, linear scheduling, and 20 epochs.

## Evaluation

`scripts/rag/evaluate_retrieval.py` evaluates the 43 training questions using their labeled document IDs as retrieval ground truth. It reports hybrid recall@1, recall@3, recall@5, and MRR, and compares hybrid recall@1 with the document-level lexical baseline. The Kaggle notebook also asserts that the final submission has the expected two columns, 11 rows in test-question order, and no blank answers. Since the test set has no local reference answers, final answer quality must be assessed by the competition's mean Levenshtein score.

## Reproduction

Use Python 3.11 or 3.12. CUDA is required for the MedGemma/LoRA run; Hugging Face access and `HF_TOKEN` are also required. Install the listed dependencies from the implementation directory:

```bash
cd scripts
python -m venv .venv
# Activate .venv using the command for your operating system.
pip install -r requirements.txt
```

For the submitted workflow, open `scripts/kaggle_main.ipynb` in Colab/Kaggle, provide `GITHUB_TOKEN` and `HF_TOKEN`, then run the cells from top to bottom. The notebook clones the repository, prepares retrieval context, trains, generates answers, validates the submission, and saves `submission.csv`.

The standalone modules document the intended auxiliary order: run the retrieval baseline, run retrieval evaluation, then run fine-tuning and generation. In this checkout, those scripts still need path/import cleanup before they are a guaranteed local one-command reproduction: data is at the repository root while some defaults resolve under `scripts/`, and `evaluate_retrieval.py` uses an import that is not package-relative. This limitation is recorded rather than presenting unverified commands as working.

## Repository Contents

- `data/`: source factsheets, train/test questions, metadata, and submissions.
- `scripts/chunking/`: chunking notebook, script, and exported chunks.
- `scripts/rag/`: chunking, embeddings, indexing, retrieval, baseline, and evaluation.
- `scripts/finetune/`: model configuration, formatting, training, and generation.
- `scripts/kaggle_main.ipynb`: the Kaggle submission workflow.
- `docs/`: problem statement, data card, impact statement, and stakeholder engagement documents.

## Appendix: Contributors and Mentors

Team leads: Ayomide Taiwo; Fatiha Adetola Azeez.

Team members: Hilary Orefo; Uyime Anthony George; Olabode Fasasi Akande; Felicia Ojochenemi Yakubu.

Mentor: Kosi Ashara.
