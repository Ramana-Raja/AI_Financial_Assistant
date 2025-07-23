# AI Financial Assistant 🤖

A bot designed to read and analyze financial PDF reports. Users can ask questions in plain English, and it will find and deliver relevant information, eliminating the need to manually search through lengthy documents.

## Features

- **Conversational Q&A**: Interact with financial PDFs using natural language to get clear, concise answers
- **Advanced RAG Pipeline**: Utilizes Retrieval-Augmented Generation to find and synthesize information
- **Conversational Memory**: Retains context over the last six turns for complex follow-up questions
- **High-Performance Models**: Powered by `google/flan-t5-base` for generation and `all-MiniLM-L6-v2` for semantic retrieval
- **Extendable**: Fine-tune models on your own data to improve performance for specific domains

## How It Works

The application uses a Retrieval-Augmented Generation (RAG) architecture:

1. **PDF Ingestion**: Extracts all text from the source PDF
2. **Chunking**: Segments text into smaller, overlapping chunks to maintain context
3. **Embedding**: Converts text chunks into numerical vectors (embeddings)
4. **Retrieval**: Finds most relevant text chunks when a question is asked
5. **Synthesis**: Generates a final, human-like answer using the FLAN-T5 model

## Tech Stack

- **Core**: Python 3.9+
- **AI Frameworks**: PyTorch, Hugging Face (transformers, sentence-transformers)
- **PDF Processing**: PyMuPDF (fitz)
- **Text Utilities**: NLTK

## 🚀 Usage

### 1. Set your PDF path

```python
pdf_path = "your_document.pdf"
```

### 2. Run The File

```python
python main.py
```

## 📁 Project Structure

```
Azure
├── modules_download
│   └── nlt.py #to download punkt
│
├── testing #contains some testings(for personal test)
│
├── training_model
│   ├── inference.py #testing with a single prompt
│   ├── train_model_answers.py #for training FLAN-T5
│   └── train_new_knowledge.py #for training all-MiniLM-L6-v2
│     
└── main.py #main file which requires pdf
```

## Road Map

### Web interface development

### Additional document format support

### Advanced agent capabilities