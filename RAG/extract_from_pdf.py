import fitz
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
import numpy as np


def extract_pdf_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        text = page.get_text()
        texts.append(text)
    return texts

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    print("NLTK 'punkt' tokenizer downloaded.")

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("NLTK 'punkt_tab' tokenizer not found. Downloading...")
    nltk.download('punkt_tab')
    print("NLTK 'punkt_tab' tokenizer downloaded.")

def semantic_chunker(text, embed_model, percentile_threshold=90):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return sentences
    sentence_embeddings = embed_model.encode(sentences, convert_to_tensor=True)
    cos_scores = util.cos_sim(sentence_embeddings[:-1], sentence_embeddings[1:])
    cos_scores = cos_scores.cpu().numpy().flatten()
    split_threshold = np.percentile(cos_scores, percentile_threshold)
    split_indices = np.where(cos_scores < split_threshold)[0]
    chunks = []
    start_index = 0
    for end_index in split_indices:
        chunk = " ".join(sentences[start_index : end_index + 1])
        chunks.append(chunk)
        start_index = end_index + 1
    if start_index < len(sentences):
        chunk = " ".join(sentences[start_index:])
        chunks.append(chunk)
    final_chunks = [c for c in chunks if len(c.strip()) > 30]
    return final_chunks

embed_model = SentenceTransformer("C:\\Users\\raman\\PycharmProjects\\pythonProject1\\pytorch\\RAG\\fine-tuned-e5-finance")
t5_model_path = "C:\\Users\\raman\\PycharmProjects\\pythonProject1\\pytorch\\RAG\\t5-finetuned-qa\\checkpoint-2625"
tokenizer = AutoTokenizer.from_pretrained(t5_model_path)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_path)

pdf_path = "C:/Users/raman/Downloads/Documents/1912.09363v3.pdf"
pages = extract_pdf_text_by_page(pdf_path)
all_text = "\n".join(pages)

semantic_chunks = semantic_chunker(all_text, embed_model, percentile_threshold=90)
print(f"Created {len(semantic_chunks)} semantic chunks.")

corpus_embeddings = embed_model.encode(semantic_chunks, convert_to_tensor=True)

def get_top_contexts(question):
    query_embedding = embed_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)
    top_k = torch.topk(scores, k=5)
    top_chunks = [semantic_chunks[i] for i in top_k.indices[0]]
    return top_chunks

def t5_answer(question, contexts):
    print(f"> Question: {question}")
    for i, context in enumerate(contexts):
        print(f"\n--- Context {i+1} ---\n{context}\n--- Answer from Context {i+1} ---")
        input_text = f"question: {question} context: {context}"
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
        output_ids = t5_model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(output)

question = "give summary of Temporal Fusion Transformers"
contexts = get_top_contexts(question)
t5_answer(question, contexts)