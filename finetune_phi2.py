import os
import torch
import random
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)
from sklearn.model_selection import train_test_split

# ----------------------------------------
# CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

# ----------------------------------------
# Configuration
MODEL_NAME    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR    = "./tinyllama_container_model"
BATCH_SIZE    = 1
EPOCHS        = 8
LEARNING_RATE = 1e-5
MAX_SEQ_LEN   = 128
TEST_SIZE     = 0.2
SEED          = 42

# ----------------------------------------
# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------
# Define dataset entries (item -> container)
types_to_container = [
    # Hazardous waste
    ("Battery", "Hazardous waste"),
    ("Used battery", "Hazardous waste"),
    ("Broken battery", "Hazardous waste"),
    ("Button cell", "Hazardous waste"),
    ("Lithium battery", "Hazardous waste"),
    ("Corroded metal parts", "Hazardous waste"),
    ("Scrap metal", "Hazardous waste"),
    ("Metal waste", "Hazardous waste"),
    ("Broken glass", "Hazardous waste"),
    ("Shards of glass", "Hazardous waste"),
    ("Paint can (with paint)", "Hazardous waste"),
    ("Oil container", "Hazardous waste"),
    ("Cleaning solvent", "Hazardous waste"),
    ("AA battery", "Hazardous waste"),
    ("Rechargeable battery", "Hazardous waste"),
    ("Old battery", "Hazardous waste"),
    ("Used oil can", "Hazardous waste"),
    ("Paint bucket", "Hazardous waste"),
    ("Chemical bottle", "Hazardous waste"),
    ("Nail polish remover", "Hazardous waste"),
    ("Bleach bottle", "Hazardous waste"),
    ("Broken lightbulb", "Hazardous waste"),
    ("Powerbank", "Hazardous waste"),
    ("Gas canister", "Hazardous waste"),


    # Organic waste bin
    ("Food waste", "Organic waste bin"),
    ("Leftover food", "Organic waste bin"),
    ("Rotten vegetables", "Organic waste bin"),
    ("Fruit peels", "Organic waste bin"),
    ("Tea bags", "Organic waste bin"),
    ("Coffee grounds", "Organic waste bin"),
    ("Paper straw", "Organic waste bin"),
    ("Used napkins", "Organic waste bin"),
    ("Eggshells", "Organic waste bin"),
    ("Bread crusts", "Organic waste bin"),
    ("Banana peel", "Organic waste bin"),
    ("Orange peel", "Organic waste bin"),
    ("Apple core", "Organic waste bin"),
    ("Moldy bread", "Organic waste bin"),
    ("Compostable plate", "Organic waste bin"),
    ("Compostable cup", "Organic waste bin"),
    ("Used coffee filter", "Organic waste bin"),
    ("Vegetable scraps", "Organic waste bin"),
    ("Leftover salad", "Organic waste bin"),
    ("Avocado skin", "Organic waste bin"),


    # Glass container
    ("Glass bottle", "Glass container"),
    ("Green glass bottle", "Glass container"),
    ("Brown glass bottle", "Glass container"),
    ("Clear glass bottle", "Glass container"),
    ("Glass jar", "Glass container"),
    ("Glass cup", "Glass container"),
    ("Perfume bottle (empty)", "Glass container"),
    ("Vinegar bottle", "Glass container"),
    ("Sauce bottle", "Glass container"),
    ("Jam jar", "Glass container"),
    ("Broken wine glass", "Glass container"),
    ("Olive jar", "Glass container"),
    ("Jam jar", "Glass container"),
    ("Beer bottle", "Glass container"),
    ("Sauce jar", "Glass container"),
    ("Glass vase", "Glass container"),
    ("Sparkling water bottle", "Glass container"),
    ("Clear bottle", "Glass container"),
    ("Used glass container", "Glass container"),
    ("Glass tumbler", "Glass container"),


    # Blue paper bin
    ("Pizza box", "Blue paper bin"),
    ("Magazine paper", "Blue paper bin"),
    ("Newspaper", "Blue paper bin"),
    ("Normal paper", "Blue paper bin"),
    ("Paper bag", "Blue paper bin"),
    ("Corrugated carton", "Blue paper bin"),
    ("Egg carton", "Blue paper bin"),
    ("Drink carton", "Blue paper bin"),
    ("Toilet paper roll", "Blue paper bin"),
    ("Cardboard box", "Blue paper bin"),
    ("Wrapping paper (non-metallic)", "Blue paper bin"),
    ("Cereal box", "Blue paper bin"),
    ("Notebook paper", "Blue paper bin"),
    ("Paper packaging", "Blue paper bin"),
    ("Paper envelope", "Blue paper bin"),
    ("Office paper", "Blue paper bin"),
    ("Gift wrapping paper", "Blue paper bin"),
    ("Used calendar", "Blue paper bin"),
    ("Brochure", "Blue paper bin"),
    ("Flyer", "Blue paper bin"),
    ("Receipt", "Blue paper bin"),
    ("Notebook cover", "Blue paper bin"),


    # Residual waste bin
    ("Tissues", "Residual waste bin"),
    ("Used tissues", "Residual waste bin"),
    ("Cigarette", "Residual waste bin"),
    ("Shoe", "Residual waste bin"),
    ("Styrofoam piece", "Residual waste bin"),
    ("Plastic gloves", "Residual waste bin"),
    ("Rope & strings", "Residual waste bin"),
    ("Foam cup", "Residual waste bin"),
    ("Sanitary pad", "Residual waste bin"),
    ("Toothbrush", "Residual waste bin"),
    ("Vacuum bag", "Residual waste bin"),
    ("Dirty paper towel", "Residual waste bin"),
    ("Dirty napkin", "Residual waste bin"),
    ("Chewing gum", "Residual waste bin"),
    ("Dust cloth", "Residual waste bin"),
    ("Used bandage", "Residual waste bin"),
    ("Cotton pad", "Residual waste bin"),
    ("Vacuum cleaner bag", "Residual waste bin"),
    ("Old sponge", "Residual waste bin"),
    ("Used razor", "Residual waste bin"),
    ("Candle stump", "Residual waste bin"),
    ("Plastic-coated paper", "Residual waste bin"),


    # Yellow bag / plastic container
    ("Plastic film", "Yellow bag / plastic container"),
    ("Plastic bag", "Yellow bag / plastic container"),
    ("Single-use carrier bag", "Yellow bag / plastic container"),
    ("Crisp packet", "Yellow bag / plastic container"),
    ("Yogurt cup", "Yellow bag / plastic container"),
    ("Tupperware", "Yellow bag / plastic container"),
    ("Aerosol", "Yellow bag / plastic container"),
    ("Plastic straw", "Yellow bag / plastic container"),
    ("Plastic utensils", "Yellow bag / plastic container"),
    ("Plastic container", "Yellow bag / plastic container"),
    ("Polypropylene bag", "Yellow bag / plastic container"),
    ("Other plastic bottle", "Yellow bag / plastic container"),
    ("Clear plastic bottle", "Yellow bag / plastic container"),
    ("Drink can", "Yellow bag / plastic container"),
    ("Aluminium foil", "Yellow bag / plastic container"),
    ("Aluminium blister pack", "Yellow bag / plastic container"),
    ("Toothpaste tube", "Yellow bag / plastic container"),
    ("Ice cream tub", "Yellow bag / plastic container"),
    ("Plastic lid", "Yellow bag / plastic container"),
    ("Butter wrapper", "Yellow bag / plastic container"),
    ("Shampoo bottle", "Yellow bag / plastic container"),
    ("Detergent bottle", "Yellow bag / plastic container"),
    ("Squeezable tube", "Yellow bag / plastic container"),
    ("Ketchup bottle", "Yellow bag / plastic container"),
    ("Mustard tube", "Yellow bag / plastic container"),
    ("Empty soap bottle", "Yellow bag / plastic container"),
    ("Plastic egg box", "Yellow bag / plastic container"),
    ("Chocolate wrapper", "Yellow bag / plastic container"),
    ("Empty deodorant can", "Yellow bag / plastic container"),
    ("Snack bag", "Yellow bag / plastic container"),
    ("Plastic netting", "Yellow bag / plastic container"),
    ("Microwave meal tray", "Yellow bag / plastic container"),
    ("Plastic spoon", "Yellow bag / plastic container"),

]

# Build prompt-format data
full_data = []
for typ, container in types_to_container:
    prompt = (
        "### Instruction:\n"
        "Which container should the following waste item go into?\n\n"
        "### Waste item:\n" + typ + "\n\n"
        "### Answer:\n" + container
    )
    full_data.append({"text": prompt})

# Split into train/test
train_data, test_data = train_test_split(full_data, test_size=TEST_SIZE, random_state=SEED)
train_ds = Dataset.from_list(train_data)
eval_ds  = Dataset.from_list(test_data)

# ----------------------------------------
# Load tokenizer and model in FP32 for stability
print("Loading tokenizer and model in FP32...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

# Disable caching for training
model.config.use_cache = False
# Use eos_token as pad_token
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# ----------------------------------------
# Tokenization function with label masking

def tokenize_fn(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN,
    )
    # Mask padding tokens in labels (set to -100)
    labels = []
    for input_ids in enc["input_ids"]:
        labels.append([
            tok if tok != tokenizer.pad_token_id else -100
            for tok in input_ids
        ])
    enc["labels"] = labels
    return enc

# Apply tokenization
train_ds = train_ds.map(tokenize_fn, batched=True)
eval_ds  = eval_ds.map(tokenize_fn, batched=True)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ----------------------------------------
# Training arguments with FP32 (no mixed precision)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,

    fp16=False,  # use full precision

    remove_unused_columns=False
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ----------------------------------------
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ----------------------------------------
# Main
if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    print("Saving model and tokenizer...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved in {OUTPUT_DIR}")
