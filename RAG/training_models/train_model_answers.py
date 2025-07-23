import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

df = pd.read_csv("C:/Users/raman/Downloads/Financial-QA-10k.csv/Financial-QA-10k.csv")
df = df.dropna(subset=["question", "context", "answer"])
df["question"] = df["question"].astype(str)
df["context"] = df["context"].astype(str)
df["answer"] = df["answer"].astype(str)

df["input_text"] = df.apply(lambda row: f"question: {row['question']} context: {row['context']}", axis=1)
df["target_text"] = df["answer"]

dataset = Dataset.from_pandas(df[["input_text", "target_text"]])


t5_model_path = "C:/Users/raman/Downloads/New folder (7)"
tokenizer = AutoTokenizer.from_pretrained(t5_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_path)

def preprocess(example):
    model_inputs = tokenizer(
        example["input_text"], max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        example["target_text"], max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./t5-finetuned-qa",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    eval_strategy="no",
    learning_rate=3e-5,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
