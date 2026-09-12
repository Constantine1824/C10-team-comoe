# TRI-AI-SLM: Safety-Aware Medical Question Answering

## Project Overview

TRI-AI-SLM is a medical question-answering benchmark system for concise, safety-aware patient education. It combines metadata-filtered retrieval, hybrid semantic/lexical search, and optional LoRA fine-tuning of `google/medgemma-4b-it`. The project is educational and does not replace a licensed clinician.

## Objectives

- Ground answers in a small, traceable public-health corpus.
- Preserve topic, care-setting, and population context during retrieval.
- Add safety guardrails for emergencies, diagnosis, and prescription changes.
- Produce a valid `QuestionId,Answer` submission for the 11-question test set.

## Dataset

The repository contains 24 synthetic public-health factsheets in `data/documents.csv`. Each record has a document ID, title, text, topic, care setting, population, origin, source URL, and license. The corpus was created as an educational, CC0-labeled dataset covering areas such as chronic disease, maternal health, and emergency triage; it is not a clinical database.

The labeled set has 43 question-answer pairs in `data/train_qa.csv`. Each question includes metadata, a `document_id` pointing to its supporting factsheet, and a reference answer. The held-out `data/test_questions.csv` has 11 questions with metadata but no answers. The test questions and the required output format are also represented by `data/sample_submission.csv`.

## Training Pipeline

1. Documents are loaded from CSV and split into sentence windows of up to 600 characters, with one sentence of overlap. Metadata is prepended to each chunk before indexing. The checked-in chunk export contains 313 chunks.
2. Retrieval first filters chunks by exact topic, care setting, and population; it falls back to topic, then to the full corpus. Dense search uses `BAAI/bge-small-en-v1.5`; lexical search uses unigram/bigram TF-IDF. Reciprocal Rank Fusion combines both rankings. If the dense model is unavailable, TF-IDF is used as a testable fallback.
3. Training examples use all chunks from the labeled gold document as context. Test prompts use the top three retrieved chunks, so training and inference share the same context format. A safety system prompt instructs the model to triage emergencies, avoid diagnosis/prescription changes, and recommend professional care.
4. Fine-tuning uses MedGemma with PEFT LoRA: rank 16, alpha 16, dropout 0.05, all linear layers, and no bias adaptation. The fixed SFT configuration is three epochs, learning rate `2e-4`, batch size 4, gradient accumulation 4, gradient checkpointing, linear scheduling, and 4-bit NF4 quantization on CUDA.

There is currently no automated hyperparameter sweep in the repository. These values are a fixed, resource-conscious configuration; comparison is performed against the retrieval baseline and the labeled retrieval metrics.

## Evaluation

Run `python -m rag.evaluate_retrieval` to evaluate retrieval against the 43 training questions, whose document IDs provide ground truth. The script reports hybrid retrieval recall@1, recall@3, recall@5, and mean reciprocal rank (MRR), and compares recall@1 with the document-level lexical baseline. This checks whether context reaches the correct source document before generation.

Run the baseline and inspect its required schema and non-empty answers. For the fine-tuned model, `finetune.run_training` checks that one answer is generated for every test question and rejects blank answers. Final benchmark scoring should use the competition's metric; the included dataset metadata identifies mean Levenshtein distance as the scoring measure. Because the test set has no local gold answers, generation quality cannot be fully verified without the benchmark evaluation.

## Tools & Libraries

Python 3.11 or 3.12, PyTorch, Hugging Face Transformers, Datasets, TRL, PEFT, Accelerate, BitsAndBytes, Sentence Transformers, scikit-learn, pandas, NumPy, and TensorBoard are used. CUDA is recommended for fine-tuning; CPU is sufficient for the lexical baseline and fallback retrieval.

## How to Use the Code

Run commands from the repository root:

```bash
python -m venv .venv
# Activate .venv using the command for your operating system.
pip install -r requirements.txt

# 1. Generate the deterministic document-retrieval submission.
python -m rag.generate_baseline

# 2. Evaluate hybrid retrieval on the labeled questions.
python -m rag.evaluate_retrieval

# 3. Optional: fine-tune and generate model answers (CUDA, Hugging Face access required).
python -m finetune.run_training
```

Outputs are written to `submissions/retrieval_baseline.csv` and, after training, `submissions/finetuned_submission.csv`. Training checkpoints are written to `checkpoints/`. The submission must contain exactly 11 non-empty rows and the columns `QuestionId,Answer`.

## Repository Structure

- 📁`data/`: source documents, train/test questions, metadata, and sample submissions.
- 📁`chunking/`: document chunking pipeline and serialized chunk export.
-📁 `rag/`: chunking, embeddings, vector storage, retrieval, baseline generation, and retrieval evaluation.
- 📁`finetune/`: model configuration, prompt preparation, LoRA training, and generation.
- 📁`utils/`: shared safety prompt and chat-formatting helpers.
- 📁`submissions/`: checked-in and generated submission files.
- 📁 `docs/`: problem_statement.pdf, data_card.pdf, impact_statement_card.pdf, stakeholder_engagement.pdf

## Appendix

### Contributors and Mentors
## Team Lead: 
- Ayomide Taiwo
- Fatiha Adetola Azeez

## Team Members
- Hilary Orefo
- Uyime Anthony George
- Olabode Fasasi Akande
- Felicia Ojochenemi Yakubu

## Mentors
- Kosi Ashara.

### Acknowledgement

This project was developed as part of the AI Saturdays Lagos Machine Learning Program, focusing on applying the concept learned to real-world sustainability problems.
Special thanks to our mentor and cohort peers for their guidance and contributions
