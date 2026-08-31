import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_ID = "google/medgemma-4b-it"

MODEL_KWARGS = {
    'attn_implementation': 'eager',
    'torch_dtype': torch.bfloat16,
    'device_map': device,
}

MODEL_KWARGS['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=MODEL_KWARGS['torch_dtype'],
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)

MODEL = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **MODEL_KWARGS)
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = 'right'

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias='none',
    target_modules='all-linear',
    task_type='CAUSAL_LM',
    #modules_to_save=['lm_head', 'embed_tokens'],
)

sft_config = SFTConfig(
    output_dir='/checkpoints/',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim='adamw_torch_fused',
    logging_steps=50,
    save_strategy='epoch',
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    lr_scheduler_type='linear',
    dataset_kwargs={'skip_prepare_dataset': True},
    remove_unused_columns=False,
    label_names=['labels']
)
