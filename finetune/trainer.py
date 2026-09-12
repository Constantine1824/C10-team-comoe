import torch
from trl import SFTTrainer
from .config import peft_config, sft_config, MODEL_ID, MODEL_KWARGS
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import Dataset
from pandas import DataFrame
from .format import format_train_data, format_test_data
from rag.retriever import HybridRetriever

model = None
processor = None

def load_model(model_path:str, model_kwargs:dict):
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = 'right'
    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    return model, processor

def get_model_and_processor():
    global model, processor
    if model is None or processor is None:
        model, processor = load_model(MODEL_ID, MODEL_KWARGS)
    return model, processor

def prepare_data(
    train_set: DataFrame,
    test_set: DataFrame,
    retriever: HybridRetriever,
    retrieval_k: int = 3
) -> tuple[Dataset, Dataset, DataFrame]:

    train_set["context"] = [
        retriever.context_for_document(document_id)
        for document_id in train_set["document_id"]
    ]
    test_set["context"] = [
        retriever.retrieve_context(
            row["question"],
            row["topic"],
            row["care_setting"],
            row["population"],
            k=retrieval_k,
        )
        for _, row in test_set.iterrows()
    ]

    train_data = Dataset.from_pandas(train_set).map(format_train_data)
    test_data = Dataset.from_pandas(test_set).map(format_test_data)
    return train_data, test_data, test_set

def collate_fn(data:list[dict[str, str]]):
    _, processor = get_model_and_processor()
    texts = []
    for item in data:
        texts.append(processor.apply_chat_template(item['messages'],
            add_generation_prompt=False,
            tokenize=False).strip())
    batch = processor(text=texts, return_tensors='pt', padding=True)
    batch['labels'] = batch['input_ids'].clone()
    batch['labels'][batch['labels'] == processor.tokenizer.pad_token_id] = -100
    return batch

def collate_fn_inf(data:list[dict[str, str]]):
    _, processor = get_model_and_processor()
    texts = [
        processor.apply_chat_template(item['messages'],
            add_generation_prompt=True,
            tokenize=False).strip()
        for item in data
    ]
    return processor(text=texts, return_tensors='pt', padding=True)

def finetune(data:Dataset):
    model, _ = get_model_and_processor()
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        data_collator=collate_fn,
        peft_config=peft_config,
        args=sft_config,
    )
    trainer.train()
    return trainer

def evaluate(trainer,test_data, batch_size=8, max_new_tokens=256):
    model = trainer.model
    _, processor = get_model_and_processor()
    model.eval()
    processor.tokenizer.padding_side = 'left'

    outputs = []
    for i in range(0, len(test_data), batch_size):
        batch = [test_data[j] for j in range(i, min(i+batch_size, len(test_data)))]
        inputs = collate_fn_inf(batch).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,
            )
        input_len = inputs['input_ids'].shape[1]
        outputs.extend(processor.tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True))
    return outputs
