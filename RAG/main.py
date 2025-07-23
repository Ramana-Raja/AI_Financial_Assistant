import fitz
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
import time
import random

MAX_HISTORY_TURNS = 6


def extract_pdf_text_by_page(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        texts = [page.get_text() for page in doc]
        print(f"Successfully extracted text from {len(texts)} pages.")
        return texts
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []


def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found.Downloading...")
        nltk.download('punkt', quiet=True)
        print("NLTK 'punkt' tokenizer downloaded.")


def fixed_size_chunker(text, chunk_size, chunk_overlap):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            overlap_text = " ".join(current_chunk.split()[-int(chunk_overlap / 5):])
            current_chunk = overlap_text + " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [c for c in chunks if len(c) > 30]


def get_top_contexts(question, chat_history, embed_model, corpus_embeddings, semantic_chunks, k=5):
    history_str = " ".join([f"User asked: {q} Bot answered: {a}" for q, a in chat_history])
    contextualized_query = f"{history_str} Current question: {question}"
    query_embedding = embed_model.encode(contextualized_query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k_indices = torch.topk(scores, k=min(k, len(semantic_chunks))).indices
    top_chunks = [semantic_chunks[i] for i in top_k_indices]
    return top_chunks

def generate_consolidated_answer(question, contexts, chat_history, t5_model, tokenizer):
    history_prompt = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history])
    context_prompt = "\n\n".join(contexts)
    input_text = (
        f"You are a helpful Q&A assistant. Using the chat history and the provided document context, "
        f"answer the user's question concisely.\n\n"
        f"--- Chat History ---\n{history_prompt}\n\n"
        f"--- Document Context ---\n{context_prompt}\n\n"
        f"--- Question ---\n{question}"
    )
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).input_ids
    output_ids = t5_model.generate(
        input_ids,
        max_length=256,
        num_beams=5,
        early_stopping=True
    )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

def print_human_like(text):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(random.uniform(0.01, 0.05))
    print()

def main():
    print("loading models...")
    try:
        embed_model_path = "all-MiniLM-L6-v2"
        t5_model_path = "google/flan-t5-base"
        pdf_path = "C:/Users/raman/Downloads/Documents/1912.09363v3.pdf"
        embed_model = SentenceTransformer(embed_model_path)
        tokenizer = AutoTokenizer.from_pretrained(t5_model_path)
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_path)
        print("models loaded successfully")
    except Exception as e:
        print(f"error loading models:{e}")
        return

    download_nltk_data()
    pages = extract_pdf_text_by_page(pdf_path)
    if not pages:
        print("pdf processing failed")
        return

    all_text = "\n".join(pages)
    print("chunking document...")
    chunks = fixed_size_chunker(all_text, chunk_size=512, chunk_overlap=100)
    print(f"Created {len(chunks)} text chunks")
    print("encoding document chunks...")
    corpus_embeddings = embed_model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    print("document processing complete")

    chat_history = []

    while True:
        question = input("\nYou: ")
        if question.lower() in ["-1"]:
            print_human_like("AI:goodbye")
            break

        contexts = get_top_contexts(question, chat_history, embed_model, corpus_embeddings, chunks)
        answer = generate_consolidated_answer(question, contexts, chat_history, t5_model, tokenizer)
        print_human_like(f"AI: {answer}")
        chat_history.append((question, answer))
        if len(chat_history) > MAX_HISTORY_TURNS:
            chat_history.pop(0)
if __name__ == "__main__":
    main()