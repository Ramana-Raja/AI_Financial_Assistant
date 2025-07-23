import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util


embed_model = SentenceTransformer("C:\\Users\\raman\\PycharmProjects\\pythonProject1\\pytorch\\RAG\\fine-tuned-e5-finance")  # Retriever
t5_model_path = "C:\\Users\\raman\\PycharmProjects\\pythonProject1\\pytorch\\RAG\\t5-finetuned-qa\\checkpoint-2625"  # Fine-tuned T5 model path
tokenizer = AutoTokenizer.from_pretrained(t5_model_path)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_path)
df = pd.read_csv("C:/Users/raman/Downloads/Financial-QA-10k.csv/Financial-QA-10k.csv")
df = df.dropna(subset=["question", "context", "answer"])
df["input_text"] = df.apply(lambda row: f"question: {row['question']} context: {row['context']}", axis=1)

input_texts = df["input_text"].tolist()
corpus_embeddings = embed_model.encode(input_texts, convert_to_tensor=True)

def get_top_contexts(query, top_k=3):
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    return [input_texts[hit["corpus_id"]] for hit in hits]

def t5_answer(question, contexts):
    for context in contexts:
        input_text = f"question: {question} context: {context}"
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
        output_ids = t5_model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("=" * 80)
        print(f"> Question: {question}")
        print(f"> Context: {context[:200]}...")
        print(f"> Answer: {output}")

if __name__ == "__main__":
    question = "what does nvidia do?"
    contexts = get_top_contexts(question, top_k=3)
    t5_answer(question, contexts)
