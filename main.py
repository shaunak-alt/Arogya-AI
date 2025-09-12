import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# --- Configuration ---
model_id = "google/gemma-2b"
dataset_folder = "./dataset"
output_dir = "./gemma-2b-chat-patient"

# Login to Hugging Face
login()

# Training settings for CPU
# Training settings for CPU
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=1,
    fp16=False,
    bf16=False,
    push_to_hub=False,
    remove_unused_columns=True, # <-- CHANGE THIS LINE
)

# --- Data Preparation ---
print("Loading up the datasets from the local folder...")
try:
    all_data = []
    for file_name in ["HealthCareMagic-100k.json", "iCliniq.json"]:
        file_path = os.path.join(dataset_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)

    dataset = Dataset.from_list(all_data)

    def formatting_prompts_func(example):
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Output:\n{example['output']}"
        return {"text": text}

    dataset = dataset.map(formatting_prompts_func)
    print("Dataset mapping complete. Now tokenizing...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # The fix is here: we add padding and truncation to the tokenizer
    # Remove padding from the tokenization step
    def tokenize_function(examples):
        # Only truncate here. The data collator will handle padding.
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

except FileNotFoundError:
    print(f"Error: The dataset files were not found in the '{dataset_folder}' folder.")
    exit()

# --- Model and Tokenizer Setup ---
print("Grabbing the model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- Fine-Tuning ---
print("Time to start fine-tuning. This might take a while...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# --- Saving the Model ---
print("Done! Saving the fine-tuned model...")
trainer.save_model(output_dir)

# --- Quick Test ---
print("\n--- Testing the new model out ---")
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
)

test_prompt = "### Instruction:\nYou are a patient speaking to a doctor. Describe your symptoms.\n\n### Input:\nHello, what seems to be the problem today?"
result = pipe(test_prompt)
generated_text = result[0]['generated_text']

print("--- What the model said ---")
print(generated_text)