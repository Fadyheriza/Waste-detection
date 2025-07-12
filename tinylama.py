import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Path to fine-tuned model ===
MODEL_DIR = "./tinyllama_container_model"

# === Load tokenizer and model locally ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)

if torch.cuda.is_available():
    model = model.to("cuda")

# === Streamlit UI ===
st.set_page_config(page_title="Waste Container Classifier", page_icon="♻️", layout="centered")
st.title("♻️ Waste Container Classifier (Austria)")

# === User input ===
input_text = st.text_input("Enter a waste item (e.g., 'Battery'):")

# === On button click ===
if st.button("Predict container") and input_text.strip():
    # Prompt must match the fine-tuning format
    prompt = (
        "### Instruction:\n"
        "Which container should the following waste item go into?\n\n"
        f"### Waste item:\n{input_text}\n\n### Answer:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_pred = decoded.replace(prompt, "").strip()
    prediction = raw_pred.split("/")[0].strip()


    st.subheader("Recommended container:")
    st.success(prediction)

# === Footer ===
st.markdown("---")
st.caption("Fine-tuned TinyLlama · Waste NLP 2025")


