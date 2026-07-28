# 🩺 Medical Knowledge AI

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python">
<img src="https://img.shields.io/badge/RAG-Production--Ready-purple">
<img src="https://img.shields.io/badge/LLM-Groq-orange">
<img src="https://img.shields.io/badge/Hybrid%20Search-BM25%20%2B%20Vector-red">
<img src="https://img.shields.io/badge/Monitoring-Enabled-success">
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit">
<img src="https://img.shields.io/badge/Docker-Ready-blue?logo=docker">
<img src="https://img.shields.io/badge/Kestra-Orchestration-orange">
<img src="https://img.shields.io/badge/License-MIT-yellow">

</p>

<h1 align="center">🩺 Medical Knowledge AI</h1>

<p align="center">
<b>An end-to-end Retrieval-Augmented Generation (RAG) system for reliable medical knowledge retrieval and grounded question answering.</b>
</p>

<p align="center">
Built with Hybrid Retrieval, Vector Search, Document Reranking, Evaluation, Monitoring, Streamlit, Kestra Workflow Orchestration, FastAPI, and Docker.
</p>

---

## 🚀 Overview

**Medical Knowledge AI** is an AI Engineering project that retrieves reliable medical knowledge and generates grounded answers using a complete Retrieval-Augmented Generation (RAG) pipeline.

Instead of relying solely on a Large Language Model, the system first retrieves relevant information from a curated medical knowledge base and then generates answers grounded in retrieved evidence. This approach improves answer reliability while reducing hallucinations.

---

## 🩺 Problem

Medical knowledge is distributed across multiple trusted sources, making it difficult to retrieve accurate and reliable information efficiently.

Traditional Large Language Models may generate hallucinated or unsupported responses because they rely only on pretrained knowledge rather than verified medical documents.

---

## 💡 Solution

Medical Knowledge AI addresses this challenge through an end-to-end Retrieval-Augmented Generation (RAG) pipeline that combines lexical retrieval, semantic search, document reranking, and Large Language Models to generate grounded medical answers.

The system:

- Retrieves relevant medical documents using **Hybrid Search (Keyword Search + BM25 + Vector Search)**
- Rewrites user queries to improve retrieval quality
- Reranks retrieved documents before answer generation
- Generates grounded responses using retrieved medical context
- Evaluates both retrieval quality and LLM generation quality
- Monitors system performance through interactive dashboards
- Provides a REST API using FastAPI
- Supports reproducible deployment using Docker

---

### Key Features

- 🔎 Keyword Search
- 📚 BM25 Retrieval
- 🧠 Vector Semantic Search
- 🔀 Hybrid Search
- 🎯 Document Reranking
- 🧩 Query Rewriting
- 🤖 Retrieval-Augmented Generation
- 📊 Retrieval Evaluation
- 🧪 LLM Generation Evaluation
- 📈 Monitoring and Observability
- 🖥️ Streamlit Application
- 🐳 Docker Deployment
- ⚙️ Kestra Workflow Orchestration

The goal is to build a **reliable, measurable, observable, and production-oriented RAG system**.

---

## 🎥 Project Demo

The demo showcases the complete application workflow:

- Asking medical questions
- Query rewriting
- Semantic retrieval
- Hybrid search
- Document reranking
- Context assembly
- Grounded answer generation
- Monitoring dashboard

📹 **Watch the demo:**

[▶️ Click here to watch the Medical Knowledge AI Demo](./assets/medical-knowledge-ai-demo.mp4)

---

## 🧠 System Architecture

```text
                         ┌────────────────────┐
                         │      User Query     │
                         └──────────┬─────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │   Query Rewriting   │
                         └──────────┬─────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │       Retrieval Layer       │
                    └──────────────┬──────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               │                   │                   │
               ▼                   ▼                   ▼
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │ Keyword      │   │ BM25 Search  │   │ Vector Search│
        │ Search       │   │              │   │ Embeddings   │
        └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
               │                  │                  │
               └──────────────────┼──────────────────┘
                                  ▼
                         ┌────────────────────┐
                         │    Hybrid Search    │
                         └──────────┬─────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │      Reranking      │
                         └──────────┬─────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │  Context Assembly   │
                         └──────────┬─────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │    Groq LLM         │
                         └──────────┬─────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │    Grounded Answer  │
                         └────────────────────┘
```

---

## 🔄 RAG Pipeline

```text
User Question
      │
      ▼
Query Rewriting
      │
      ▼
Keyword Search ──────┐
      │              │
      ▼              │
BM25 Search ────────┤
      │              │
      ▼              │
Vector Search ──────┘
      │
      ▼
Hybrid Retrieval
      │
      ▼
Document Reranking
      │
      ▼
Context Assembly
      │
      ▼
Groq LLM
      │
      ▼
Grounded Medical Answer
```

---

## 🔍 Retrieval System

The retrieval layer combines multiple retrieval strategies.

### Keyword Search

Useful for exact medical terms and important keywords.

### BM25 Search

Provides strong lexical retrieval and is useful when exact terminology is important.

### Vector Semantic Search

Uses embeddings to retrieve documents based on semantic similarity rather than exact word matching.

### Hybrid Search

Combines lexical and semantic retrieval signals to improve retrieval quality and robustness.

```text
BM25 Results
      +
Vector Search Results
      +
Keyword Results
      │
      ▼
Score Fusion
      │
      ▼
Hybrid Results
      │
      ▼
Reranking
      │
      ▼
Best Context
```

---

## 🧩 Query Rewriting

The system includes a query rewriting stage that improves the original user question before retrieval.

```text
Original Query
      │
      ▼
Query Rewriting
      │
      ▼
Improved Search Query
      │
      ▼
Retrieval Pipeline
```

This can help improve search clarity, retrieval recall, and the quality of the final context.

---

## 🤖 Retrieval-Augmented Generation

The final answer is generated using retrieved medical context.

The LLM receives relevant documents retrieved from the knowledge base, helping reduce unsupported answers and improve grounding.

```text
User Question
      ↓
Query Rewriting
      ↓
Hybrid Retrieval
      ↓
Reranking
      ↓
Relevant Context
      ↓
Prompt Construction
      ↓
Groq LLM
      ↓
Grounded Answer
```

---

## 📚 Data Ingestion Pipeline

The project includes a structured ingestion pipeline for processing medical knowledge.

```text
Raw Medical Data
       ↓
Data Loading
       ↓
Validation
       ↓
Document Chunking
       ↓
Metadata Processing
       ↓
Embeddings
       ↓
Vector Cache
       ↓
Retrieval System
```

Relevant files:

```text
ingestion/
├── load_medlineplus.py
├── chunk_documents.py
├── inspect_metadata.py
├── schema.py
└── validate_data.py
```

---

## 📊 Retrieval Evaluation

The retrieval layer is evaluated using predefined medical queries and ground-truth data.

The evaluation compares different retrieval approaches:

- Keyword Search
- BM25 Search
- Vector Search
- Hybrid Search

### Evaluation Workflow

```text
Evaluation Queries
        │
        ▼
Ground Truth
        │
        ▼
Run Retrieval Strategies
        │
        ▼
Collect Retrieved Documents
        │
        ▼
Compare with Ground Truth
        │
        ▼
Calculate Metrics
        │
        ▼
Save Results
```

This makes it possible to measure retrieval quality and compare different retrieval strategies instead of relying only on manual testing.

Relevant files:

```text
evaluation/
├── create_evaluation_dataset.py
├── retrieval_queries.json
├── retrieval_evaluation.py
└── retrieval_results.json
```

---

## 🧠 LLM Generation Evaluation

The generation pipeline is evaluated to measure the quality of the generated answers.

The evaluation focuses on:

- 🎯 Answer Relevance
- 📚 Grounding in Retrieved Context
- ✅ Factual Correctness
- 🧩 Answer Completeness
- 🚫 Reduction of Unsupported Information

```text
Retrieved Context
        │
        ▼
Prompt Configuration
        │
        ▼
LLM Generation
        │
        ▼
Generated Answer
        │
        ▼
Quality Evaluation
        │
        ▼
Saved Results
```

Relevant files:

```text
evaluation/
├── generation_evaluation.py
├── generation_results.json
└── evaluation_results.json
```

---

## 📈 Monitoring and Observability

The project includes a monitoring layer for tracking system behavior and performance.

Tracked information includes:

- 📊 Request Counts
- ⚡ Request Latency
- ✅ Success Rates
- ❌ Failed Requests
- 🔍 Retrieval Information
- 👍 User Feedback

Relevant files:

```text
monitoring/
├── feedback.py
├── logger.py
├── metrics.py
└── test_monitoring.py
```

---

## 📊 Monitoring Dashboard

An interactive Streamlit dashboard provides visibility into application performance.

The dashboard includes:

- Total Requests
- Success and Failure Rates
- Average Latency
- Request Trends
- Latency Distribution
- Retrieved Documents
- User Feedback

```text
Application
     │
     ▼
Request Logging
     │
     ▼
Metrics Collection
     │
     ▼
Feedback Tracking
     │
     ▼
Streamlit Monitoring Dashboard
```

---

## ⚙️ Workflow Orchestration

The project includes workflow orchestration using **Kestra**.

The orchestration layer can be used to organize and automate AI Engineering workflows such as:

- Data Processing
- Evaluation Pipelines
- RAG Workflows
- Reproducible AI Tasks

Configuration:

```text
orchestration/
└── docker-compose.yml
```

---

## 🐳 Docker Deployment

The project supports containerized deployment using Docker.

```text
Dockerfile
requirements.docker.txt
```

Build the image:

```bash
docker build -t medical-knowledge-ai .
```

Run the container:

```bash
docker run -p 8501:8501 medical-knowledge-ai
```

---

## 📁 Project Structure

```text
Medical-Knowledge-AI/
│
├── app/
│   ├── app.py
│   ├── style.css
│   └── pages/
│       └── 2_📊_Monitoring_Dashboard.py
│
├── assets/
│   └── medical-knowledge-ai-demo.mp4
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
│   ├── generation_evaluation.py
│   ├── retrieval_evaluation.py
│   ├── retrieval_queries.json
│   ├── retrieval_results.json
│   ├── generation_results.json
│   └── evaluation_results.json
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
```

---

## 🛠️ Tech Stack

### AI & LLM

- Generative AI
- Large Language Models
- Retrieval-Augmented Generation
- Groq LLM
- Prompt Engineering

### Information Retrieval

- Keyword Search
- BM25
- Vector Search
- Embeddings
- Hybrid Search
- Document Reranking
- Query Rewriting

### Data Processing

- Python
- JSON
- XML
- Document Chunking
- Metadata Processing

### Evaluation

- Retrieval Evaluation
- LLM Generation Evaluation
- Ground-Truth Datasets
- Automated Evaluation Pipelines

### Monitoring

- Request Logging
- Metrics Collection
- Latency Tracking
- User Feedback
- Streamlit Dashboard

### Deployment

- Docker
- Docker Compose
- Kestra Workflow Orchestration

---

## 🏆 Engineering Principles

This project follows modern AI Engineering practices:

### 🔀 Hybrid Retrieval

Combines lexical and semantic retrieval to improve robustness.

### 🎯 Reranking

Improves the relevance of the final context passed to the LLM.

### 📊 Evaluation-Driven Development

Measures both retrieval quality and generation quality.

### 📈 Observability

Tracks system behavior, performance, and user feedback.

### 🧪 Reproducibility

Uses structured datasets, saved evaluation results, dependency files, and containerization.

### 🧱 Modular Architecture

Separates ingestion, retrieval, RAG generation, evaluation, monitoring, and application components.

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/Ai-MAFlutter/Medical-Knowledge-AI.git
cd Medical-Knowledge-AI
```

### Create Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

⚠️ Never commit API keys or `.env` files to GitHub.

---

## ▶️ Run the Application

```bash
streamlit run app/app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🧪 Testing

Run monitoring tests:

```bash
python monitoring/test_monitoring.py
```

Run RAG generator tests:

```bash
python rag/test_generator.py
```

---

## 📊 Run Evaluation

Run retrieval evaluation:

```bash
python evaluation/retrieval_evaluation.py
```

Run generation evaluation:

```bash
python evaluation/generation_evaluation.py
```

Run the complete evaluation pipeline:

```bash
python evaluate.py
```

---

## 🗺️ Future Improvements

- 🔬 Advanced Cross-Encoder Reranking
- 📊 Improved Retrieval Metrics
- 🧠 Hallucination Detection
- 📚 Medical Answer Citations
- 💬 Conversation Memory
- 🔐 User Authentication
- ☁️ Cloud Deployment
- 🔄 CI/CD Pipeline
- 🔍 Advanced Tracing
- 🧪 A/B Testing
- ♻️ Continuous Evaluation
- 🗄️ Production Database Integration

---

## ⚠️ Medical Disclaimer

This project is intended for educational and research purposes only.

It is not a replacement for professional medical advice, diagnosis, or treatment.

Always consult a qualified healthcare professional for medical decisions.

---

## 👩‍💻 Author

**Marina Wahid**

Artificial Intelligence Developer | Generative AI | RAG | Machine Learning | Flutter Developer

Building intelligent applications using:

- 🤖 Generative AI
- 🧠 Large Language Models
- 🔍 Retrieval-Augmented Generation
- 📊 Machine Learning
- 🔎 Information Retrieval
- 📈 Monitoring and Observability
- 🐍 Python
- 📱 Flutter

---

<p align="center">

⭐ If you find this project useful, consider giving it a star!

</p>