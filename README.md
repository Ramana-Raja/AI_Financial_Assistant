AI Financial Assistant 🤖

This project is a bot designed to read and analyze financial PDF reports. Users can ask it questions in plain English, and it will find and deliver the relevant information, removing the need to manually search through lengthy documents.
What It Does

    Conversational Q&A: Allows users to interact with financial PDFs using natural language to get clear, concise answers.

    Advanced RAG Pipeline: Utilizes a Retrieval-Augmented Generation (RAG) pipeline to find the right information and piece together a coherent answer.

    Conversational Memory: The system retains context over the last six turns, allowing for complex follow-up questions without needing to restart the conversation.

    High-Performance Models: Powered by google/flan-t5-base for generation and all-MiniLM-L6-v2 for semantic retrieval.

    Extendable: Users can fine-tune the models on their own data to improve performance for specific domains.

How It Works (The Guts)

The application is built on a Retrieval-Augmented Generation (RAG) architecture:

    PDF Ingestion: It first extracts all text from the source PDF.

    Chunking: The text is then segmented into smaller, overlapping chunks to maintain semantic context.

    Embedding: An AI model converts these text chunks into numerical vectors (embeddings) to understand their meaning.

    Retrieval: When a user asks a question, the system finds the most relevant text chunks by comparing the question's embedding to the document's embeddings.

    Synthesis: The best-matching chunks, along with the chat history and the original question, are fed to the FLAN-T5 model to generate a final, human-like answer.

The Tech Stack

    Core: Python 3.9+

    AI Frameworks: PyTorch, Hugging Face (transformers, sentence-transformers)

    PDF Processing: PyMuPDF (fitz)

    Text Utilities: NLTK


How to Use It
1. Chat with a PDF

This is the main functionality of the application.

Configuration:

    Open main.py.

    Update the pdf_path variable to the file path of your PDF.

# Inside main.py
pdf_path = "path/to/your/financial_report.pdf"

Run the application:

python main.py

The script will load the necessary models and process the document. Once it's ready, you can start asking questions. Type -1 to exit.
2. Fine-Tuning the Models (Optional)

Users can enhance the bot's performance by training it on custom financial Q&A data.
A. Train the Generator (FLAN-T5)

    Open fine_tune_t5.py.

    Ensure the path to your CSV file is correct. The CSV must have question, context, and answer columns.

    Run python fine_tune_t5.py.

B. Train the Retriever (Sentence Transformer)

    Open fine_tune_embedding_model.py.

    Point it to your CSV dataset.

    Run python fine_tune_embedding_model.py.

Project Structure

.
├── main.py                     # The main chat application
├── fine_tune_t5.py             # Script to train the generator AI
├── fine_tune_embedding_model.py # Script to train the retriever AI
├── requirements.txt            # List of dependencies
└── README.md                   # You're reading it

Future Roadmap

    [ ] Develop a web UI (e.g., using Streamlit or Flask).

    [ ] Add support for more document formats like .docx or URLs.

    [ ] Integrate a vector database for more efficient and scalable indexing.

    [ ] Explore more advanced agentic behaviors and tool integrations.

License

This project is available under the MIT License.