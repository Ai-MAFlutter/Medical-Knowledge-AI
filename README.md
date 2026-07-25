# 🩺 Medical Knowledge AI

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python">
<img src="https://img.shields.io/badge/RAG-Production--Ready-purple">
<img src="https://img.shields.io/badge/LLM-Groq-orange">
<img src="https://img.shields.io/badge/Vector%20Search-Embeddings-green">
<img src="https://img.shields.io/badge/Hybrid%20Search-BM25%20%2B%20Vector-red">
<img src="https://img.shields.io/badge/Monitoring-Enabled-success">
<img src="https://img.shields.io/badge/Docker-Ready-blue?logo=docker">
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit">
<img src="https://img.shields.io/badge/License-MIT-yellow">

</p>

<h1 align="center">🩺 Medical Knowledge AI</h1>

<p align="center">
<b>An end-to-end, evaluation-driven Retrieval-Augmented Generation system for medical knowledge retrieval and grounded question answering.</b>
</p>

<p align="center">
Built with modern AI engineering practices including hybrid retrieval, vector search, reranking, query rewriting, automated evaluation, observability, monitoring, testing, and containerized deployment.
</p>

---

## 📌 Project Overview

**Medical Knowledge AI** is a production-oriented Retrieval-Augmented Generation (RAG) system designed to retrieve relevant medical knowledge and generate grounded answers using an LLM.

Instead of relying only on the language model's internal knowledge, the system follows a complete pipeline:

```text
User Question
      ↓
Query Rewriting
      ↓
Document Retrieval
      ↓
Hybrid Search
      ↓
Reranking
      ↓
Context Assembly
      ↓
LLM Generation
      ↓
Grounded Answer
      ↓
Monitoring & Feedback

🎯 Project Goals

The main goals of this project are:

Build an end-to-end RAG pipeline
Improve retrieval quality using multiple retrieval strategies
Combine lexical and semantic search
Reduce irrelevant context using reranking
Improve query understanding using query rewriting
Evaluate retrieval quality automatically
Evaluate generated answers
Monitor system behavior and performance
Containerize the application
Build a production-oriented AI architecture
flowchart TD

    A[👤 User Query] --> B[🔄 Query Rewriting]

    B --> C[🔎 Retrieval Layer]

    C --> D[🔤 Keyword Search]
    C --> E[📚 BM25 Search]
    C --> F[🧠 Vector Search]

    D --> G[🔀 Hybrid Search]
    E --> G
    F --> G

    G --> H[🎯 Reranking]

    H --> I[🧩 Context Assembly]

    I --> J[🤖 Groq LLM]

    J --> K[💬 Grounded Answer]

    K --> L[📊 Monitoring]
    K --> M[🧪 Evaluation]
    K --> N[👍 User Feedback]

🔎 Retrieval Pipeline

The retrieval system combines multiple retrieval strategies.

1️⃣ Keyword Search

Keyword search is useful for exact medical terms and specific keywords.
Query
  ↓
Keyword Matching
  ↓
Relevant Documents

This is especially useful when the query contains:

Medical terminology
Disease names
Drug names
Specific symptoms
Exact phrases

BM25 Retrieval

BM25 is a strong lexical retrieval algorithm that ranks documents based on term relevance.

Query
  ↓
Tokenization
  ↓
BM25 Scoring
  ↓
Ranked Documents

BM25 is particularly useful when exact terminology matters.

3️⃣ Vector Semantic Search

Documents are converted into embeddings and retrieved based on semantic similarity.

User Query
    ↓
Embedding Model
    ↓
Vector Representation
    ↓
Similarity Search
    ↓
Relevant Documents

Vector search allows the system to retrieve documents even when the user uses different words with a similar meaning.

4️⃣ Hybrid Search

The system combines lexical and semantic retrieval.
             ┌───────────────┐
             │   User Query  │
             └───────┬───────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
     BM25 Search          Vector Search
          │                     │
          └──────────┬──────────┘
                     ▼
              Score Fusion
                     ▼
              Hybrid Results

Hybrid retrieval improves recall by combining:

Exact keyword matching
Lexical relevance
Semantic similarity

This makes the system more robust to different query styles.

🎯 Reranking

After the initial retrieval stage, candidate documents are reranked to improve the final context.

Initial Retrieval
       ↓
Candidate Documents
       ↓
Reranking
       ↓
Most Relevant Context

The goal is to provide the LLM with the most relevant information possible.

🔄 Query Rewriting

User questions may be:

Ambiguous
Too short
Informal
Missing important context

The query rewriting stage improves the original question before retrieval.

Original Query
      ↓
Query Rewriter
      ↓
Improved Search Query
      ↓
Retrieval

This improves retrieval quality and helps the system understand user intent.

🤖 Retrieval-Augmented Generation

The generation pipeline is grounded in retrieved medical knowledge.
┌────────────────────┐
│    User Question   │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│  Query Rewriting   │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│  Hybrid Retrieval  │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│     Reranking      │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│ Context Assembly   │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│     Groq LLM       │
└──────────┬─────────┘
           ▼
┌────────────────────┐
│  Grounded Answer   │
└────────────────────┘

The system is designed to reduce unsupported answers by grounding responses in retrieved medical context.

🧪 Evaluation Framework

A key focus of this project is evaluation-driven AI engineering.

The system evaluates both:

Retrieval quality
Generation quality
🔎 Retrieval Evaluation

Retrieval evaluation measures whether the correct documents are retrieved for a query.

The evaluation pipeline includes:

Evaluation Queries
       ↓
Ground Truth
       ↓
Retrieval System
       ↓
Retrieved Results
       ↓
Evaluation Metrics
       ↓
Saved Results

Relevant files:

evaluation/
├── create_evaluation_dataset.py
├── retrieval_queries.json
├── retrieval_evaluation.py
└── retrieval_results.json
🧠 Generation Evaluation

The generation pipeline evaluates the quality of generated answers.

Retrieved Context
       ↓
LLM Generation
       ↓
Generated Answer
       ↓
Evaluation
       ↓
Evaluation Results

Relevant files:

evaluation/
├── generation_evaluation.py
├── generation_results.json
└── evaluation_results.json

This makes the system measurable instead of relying only on manual testing.

📊 Monitoring & Observability

The project includes a monitoring layer to track system behavior.

The monitoring system can track:

Request counts
Response latency
Retrieval behavior
System metrics
User feedback
Application events

Structure:

monitoring/
├── feedback.py
├── logger.py
├── metrics.py
└── test_monitoring.py
📈 Monitoring Dashboard

The project includes an interactive Streamlit monitoring dashboard.

The dashboard provides visibility into:

System usage
Request activity
Performance metrics
User feedback
Application behavior
app/
├── app.py
├── style.css
└── pages/
    └── 2_📊_Monitoring_Dashboard.py
🗂️ Data Ingestion Pipeline

The ingestion pipeline processes the medical knowledge source before retrieval.

Raw Medical Data
       ↓
Data Loading
       ↓
Validation
       ↓
Document Processing
       ↓
Chunking
       ↓
Metadata Processing
       ↓
Embeddings
       ↓
Vector Cache
       ↓
Retrieval System

The ingestion pipeline includes:

ingestion/
├── load_medlineplus.py
├── chunk_documents.py
├── inspect_metadata.py
├── validate_data.py
└── schema.py
📁 Project Structure
Medical-Knowledge-AI/
│
├── app/
│   ├── app.py
│   ├── style.css
│   └── pages/
│       └── 2_📊_Monitoring_Dashboard.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── medlineplus_chunks.json
│       ├── medlineplus_documents.json
│       └── vector_cache/
│
├── evaluation/
│   ├── create_evaluation_dataset.py
│   ├── evaluation_results.json
│   ├── generation_evaluation.py
│   ├── generation_results.json
│   ├── retrieval_evaluation.py
│   ├── retrieval_queries.json
│   └── retrieval_results.json
│
├── ingestion/
│   ├── chunk_documents.py
│   ├── inspect_metadata.py
│   ├── load_medlineplus.py
│   ├── schema.py
│   └── validate_data.py
│
├── monitoring/
│   ├── feedback.py
│   ├── logger.py
│   ├── metrics.py
│   └── test_monitoring.py
│
├── orchestration/
│   └── docker-compose.yml
│
├── rag/
│   ├── generator.py
│   ├── query_rewriter.py
│   ├── retrieval_pipeline.py
│   └── test_generator.py
│
├── retrieval/
│   ├── bm25_search.py
│   ├── hybrid_search.py
│   ├── keyword_search.py
│   ├── reranker.py
│   └── vector_search.py
│
├── api.py
├── Dockerfile
├── evaluate.py
├── requirements.txt
├── requirements.docker.txt
└── README.md
🛠️ Technology Stack
Programming
Python
JSON
XML
Generative AI
Large Language Models
Retrieval-Augmented Generation
Prompt Engineering
Groq API
Information Retrieval
Keyword Search
BM25
Vector Search
Hybrid Search
Embeddings
Reranking
Data Processing
NumPy
Document Chunking
Metadata Processing
Data Validation
Evaluation
Retrieval Evaluation
Generation Evaluation
Ground Truth Datasets
Automated Evaluation Pipelines
Monitoring
Request Logging
Metrics Collection
User Feedback
Monitoring Dashboard
Deployment
Docker
Docker Compose
Containerized Services
Interface
Streamlit
⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/Ai-MAFlutter/Medical-Knowledge-AI.git
cd Medical-Knowledge-AI
2️⃣ Create a Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt

For Docker environments:

pip install -r requirements.docker.txt
🔐 Environment Variables

Create a .env file in the project root:

GROQ_API_KEY=your_api_key_here

⚠️ Never commit API keys or .env files to GitHub.

▶️ Running the Application

Run the Streamlit application:

streamlit run app/app.py

Then open:

http://localhost:8501
🐳 Docker

Build the Docker image:

docker build -t medical-knowledge-ai .

Run the container:

docker run -p 8501:8501 medical-knowledge-ai
⚙️ Docker Compose

Run the orchestration environment:

docker compose -f orchestration/docker-compose.yml up --build

Stop the services:

docker compose -f orchestration/docker-compose.yml down
🧪 Running Tests

Run monitoring tests:

python monitoring/test_monitoring.py

Run RAG generator tests:

python rag/test_generator.py
📊 Running Evaluation

Run retrieval evaluation:

python evaluation/retrieval_evaluation.py

Run generation evaluation:

python evaluation/generation_evaluation.py

Run the complete evaluation pipeline:

python evaluate.py
🧩 Engineering Principles

This project follows modern AI engineering principles.

✅ Modular Architecture

Each component has a clear responsibility.

✅ Retrieval First

The LLM does not operate alone. Relevant context is retrieved before generation.

✅ Evaluation Driven

The system quality is measured using automated evaluation pipelines.

✅ Observable

System behavior can be monitored and analyzed.

✅ Reproducible

The project includes structured data, evaluation results, and configuration files.

✅ Production Oriented

The system is designed with deployment, monitoring, testing, and maintainability in mind.

🗺️ Future Improvements

Potential future improvements include:

Advanced cross-encoder reranking
Improved retrieval metrics
Hallucination detection
Medical answer citations
Conversation memory
User authentication
Cloud deployment
Automated CI/CD
Advanced tracing
A/B testing for retrieval strategies
Continuous evaluation
Production database integration
⚠️ Medical Disclaimer

This project is intended for educational and research purposes only.

It is not a replacement for professional medical advice, diagnosis, or treatment.

Always consult a qualified healthcare professional for medical decisions.

👩‍💻 Author
Marina Wahid

Artificial Intelligence Developer | Generative AI | RAG | Machine Learning | Flutter Developer

Building intelligent applications using:

🤖 Generative AI
🧠 Large Language Models
🔍 Retrieval-Augmented Generation
📊 Machine Learning
🐍 Python
📱 Flutter
<p align="center">

⭐ If you find this project useful, consider giving it a star!

</p> ```