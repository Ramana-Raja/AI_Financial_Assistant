from sentence_transformers import SentenceTransformer

model = SentenceTransformer("C:\\Users\\raman\\Downloads\\New folder (8)")

embeddings = model.encode(["hello world", "goodbye world"],convert_to_tensor=True)
print(embeddings.shape)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

t5_model_name = "C:\\Users\\raman\\Downloads\\New folder (7)"

tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)

def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

question = "What is RAG?"
context = "RAG stands for Retrieval-Augmented Generation. It retrieves relevant documents and then generates answers using them."
print(generate_answer(question, context))

