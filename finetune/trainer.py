import torch
from trl import SFTTrainer
from .config import peft_config, sft_config, processor, MODEL
from transformers import pipeline
from datasets import Dataset

def collate_fn(data:list[dict[str, str]]):
    texts = []
    for item in data:
        texts.append(processor.apply_chat_template(item['messages'],
            add_generation_prompt=False,
            tokenize=False).strip())
    batch = processor(text=texts, return_tensors='pt', padding=True)
    batch['labels'] = batch['input_ids'].clone()
    batch['labels'][batch['labels'] == processor.tokenizer.pad_token_id] = -100
    return batch

def finetune(data:Dataset):
    trainer = SFTTrainer(
        model=MODEL,
        train_dataset=data,
        data_collator=collate_fn,
        peft_config=peft_config,
        args=sft_config,
    )
    trainer.train()
    return trainer

def run_inference(trainer,test_data):
    pipe = pipeline('image-text-to-text',
        model=trainer.model,
        processor=processor,
        torch_dtype=torch.bfloat16)
    pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenzer.padding_side = 'left'
    results = pipe(
        text=test_data['messages'],
        batch_size=64,
    )
    return list(results)
