
# 🩺 Medical Knowledge AI

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python">
<img src="https://img.shields.io/badge/RAG-Production--Ready-purple">
<img src="https://img.shields.io/badge/LLM-Groq-orange">
<img src="https://img.shields.io/badge/Vector%20Search-Embeddings-green">
<img src="https://img.shields.io/badge/Hybrid%20Search-BM25%20%2B%20Vector-red">
<img src="https://img.shields.io/badge/Reranking-Enabled-blueviolet">
<img src="https://img.shields.io/badge/Evaluation-RAGAS%20%7C%20Custom-success">
<img src="https://img.shields.io/badge/Monitoring-Enabled-success">
<img src="https://img.shields.io/badge/Docker-Ready-blue?logo=docker">
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit">
<img src="https://img.shields.io/badge/License-MIT-yellow">

</p>

<h1 align="center">🩺 Medical Knowledge AI</h1>

<p align="center">
<b>An end-to-end Retrieval-Augmented Generation (RAG) system for reliable medical knowledge retrieval and question answering.</b>
</p>

<p align="center">
Built with modern AI engineering practices including hybrid retrieval, vector search, reranking, evaluation, observability, monitoring, and containerized deployment.
</p>

---

## 🚀 Overview

**Medical Knowledge AI** is an end-to-end AI system designed to retrieve and generate grounded answers from a curated medical knowledge base.

Unlike a simple chatbot that relies only on an LLM, this project implements a complete AI engineering pipeline focused on:

- 🔎 Information Retrieval
- 📚 BM25 Lexical Search
- 🧠 Vector Semantic Search
- 🔀 Hybrid Retrieval
- 🎯 Document Reranking
- 🧩 Query Rewriting
- 🤖 Retrieval-Augmented Generation
- 📊 Retrieval Evaluation
- 🧪 Generation Evaluation
- 📈 Monitoring and Observability
- 🐳 Docker Deployment
- ⚙️ Workflow Orchestration
- 📊 Interactive Monitoring Dashboard

The main goal is to build a **reliable, measurable, observable, and production-oriented RAG system** instead of a simple LLM chatbot.

---

## 🎥 Project Demo

Watch the Medical Knowledge AI system in action:

<p align="center">

🎬 **Application Demo**

</p>

> The demo showcases the complete RAG workflow, including medical question answering, query rewriting, semantic retrieval, document reranking, grounded answer generation, and the monitoring dashboard.

📹 **Watch the demo video:**

[▶️ Download and Watch the Medical Knowledge AI Demo](./assets/medical-knowledge-ai-demo.mp4)

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
                         │    LLM Generator    │
                         └──────────┬─────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │    Grounded Answer  │
                         └────────────────────┘
````

---

## 🔍 Retrieval Pipeline

The retrieval layer combines multiple search strategies to improve recall and relevance.

### 🔹 Keyword Search

Keyword search is useful for exact medical terms and important keywords.

```text
User Query
    ↓
Keyword Matching
    ↓
Relevant Medical Documents
```

---

### 🔹 BM25 Search

BM25 is a lexical information retrieval algorithm that ranks documents based on term frequency and document relevance.

```text
User Query
    ↓
Tokenization
    ↓
BM25 Scoring
    ↓
Ranked Documents
```

BM25 is particularly useful when exact medical terminology is important.

---

### 🔹 Vector Semantic Search

Documents are converted into embeddings and searched based on semantic similarity.

```text
User Query
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Semantic Results
```

This allows the system to retrieve documents with similar meaning even when the exact words are different.

---

### 🔹 Hybrid Search

The system combines lexical and semantic retrieval.

```text
                 ┌──────────────┐
                 │ BM25 Search  │
                 └──────┬───────┘
                        │
                        ▼
                   BM25 Results
                        │
                        │
                        ├──────────────┐
                        │              │
                        ▼              ▼
                ┌──────────────┐ ┌──────────────┐
                │ Vector Search│ │ Score Fusion │
                └──────┬───────┘ └──────┬───────┘
                       │                │
                       ▼                ▼
                 Vector Results ──► Hybrid Results
```

Hybrid retrieval improves robustness by combining:

* Exact lexical matching
* Semantic similarity
* Medical terminology matching
* Query variations

---

## 🎯 Reranking

After retrieving candidate documents, the system applies a reranking stage to improve the final ordering of results.

```text
Initial Retrieval
        ↓
Candidate Documents
        ↓
Reranking
        ↓
Most Relevant Context
```

Reranking helps ensure that the LLM receives the most relevant documents instead of simply using the first retrieved results.

---

## 🧩 Query Rewriting

The system supports query rewriting to improve retrieval quality.

```text
Original User Query
        ↓
Query Rewriter
        ↓
Improved Search Query
        ↓
Retrieval Pipeline
```

Query rewriting can improve retrieval when the original question is:

* Ambiguous
* Too short
* Informal
* Missing important medical keywords

---

## 🤖 Retrieval-Augmented Generation

The final answer is generated using retrieved medical context.

```text
User Question
      ↓
Query Processing
      ↓
Query Rewriting
      ↓
Hybrid Retrieval
      ↓
Reranking
      ↓
Context Assembly
      ↓
LLM Generator
      ↓
Grounded Answer
```

The system is designed to ground the generated answer in retrieved medical knowledge and reduce unsupported responses.

---

## 📚 Knowledge Base

The project uses a curated medical knowledge base based on structured medical information.

The data pipeline includes:

```text
Raw Medical Data
       ↓
Data Loading
       ↓
Document Validation
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
```

---

## 🗂️ Data Ingestion Pipeline

The ingestion layer is responsible for preparing medical documents for retrieval.

```text
ingestion/
├── load_medlineplus.py
├── chunk_documents.py
├── inspect_metadata.py
├── validate_data.py
└── schema.py
```

The pipeline includes:

* 📥 Data Loading
* 🧹 Data Cleaning
* ✅ Schema Validation
* ✂️ Document Chunking
* 🏷️ Metadata Processing
* 🧠 Embedding Preparation
* 💾 Vector Cache Generation

---

## 🧪 Evaluation Framework

A major focus of this project is building an evaluation-driven RAG system.

The system evaluates both:

1. Retrieval Quality
2. Generation Quality

---

### 🔎 Retrieval Evaluation

Retrieval evaluation measures whether the system retrieves the correct documents for a given query.

```text
Evaluation Queries
        ↓
Retrieval Pipeline
        ↓
Retrieved Documents
        ↓
Ground Truth Comparison
        ↓
Evaluation Metrics
        ↓
Saved Results
```

Relevant files:

```text
evaluation/
├── create_evaluation_dataset.py
├── retrieval_evaluation.py
├── retrieval_queries.json
└── retrieval_results.json
```

Retrieval evaluation helps analyze:

* Search relevance
* Retrieval quality
* Ranking performance
* Retrieval failures

---

### 🧠 Generation Evaluation

The generation pipeline evaluates the quality of the final answers.

```text
User Question
        ↓
RAG Pipeline
        ↓
Generated Answer
        ↓
Evaluation
        ↓
Quality Metrics
```

Relevant files:

```text
evaluation/
├── generation_evaluation.py
├── generation_results.json
└── evaluation_results.json
```

The evaluation process helps measure:

* Answer correctness
* Context relevance
* Groundedness
* Answer quality
* Retrieval-to-generation performance

---

## 📊 Monitoring & Observability

The project includes monitoring and observability capabilities for tracking system behavior.

The monitoring layer tracks important information such as:

* 📈 Request Counts
* ⏱️ Latency
* 🔎 Retrieval Performance
* 💬 User Feedback
* 🧠 System Behavior
* ⚠️ Errors and Failures

```text
monitoring/
├── feedback.py
├── logger.py
├── metrics.py
└── test_monitoring.py
```

---

## 📈 Monitoring Dashboard

The project includes an interactive Streamlit monitoring dashboard.

The dashboard provides visibility into application performance and user interactions.

```text
app/
├── app.py
├── style.css
└── pages/
    └── 2_📊_Monitoring_Dashboard.py
```

The dashboard can be used to inspect:

* System Metrics
* Request Performance
* Retrieval Behavior
* User Feedback
* Application Monitoring Data

---

## 🧱 Project Structure

```text
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
```

---

## 🛠️ Tech Stack

### Programming

* Python
* NumPy
* JSON
* XML

### Generative AI

* Large Language Models
* Groq LLM
* Prompt Engineering
* Retrieval-Augmented Generation

### Information Retrieval

* BM25
* Keyword Search
* Vector Search
* Embeddings
* Hybrid Retrieval
* Reranking

### Data Processing

* Document Loading
* XML Processing
* Document Chunking
* Metadata Processing
* Data Validation

### Evaluation

* Retrieval Evaluation
* Generation Evaluation
* Ground Truth Datasets
* Automated Evaluation Pipelines
* Evaluation Metrics

### Monitoring

* Request Logging
* Metrics Collection
* User Feedback
* Application Monitoring
* Interactive Dashboards

### Deployment

* Docker
* Docker Compose
* Containerized Services
* Workflow Orchestration

### Interface

* Streamlit

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ai-MAFlutter/Medical-Knowledge-AI.git
cd Medical-Knowledge-AI
```

---

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For Docker environments:

```bash
pip install -r requirements.docker.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

Never commit your `.env` file or API keys to GitHub.

Make sure your `.gitignore` includes:

```text
.venv/
.env
__pycache__/
*.pyc
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app/app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🐳 Docker Deployment

### Build the Docker Image

```bash
docker build -t medical-knowledge-ai .
```

### Run the Container

```bash
docker run -p 8501:8501 medical-knowledge-ai
```

---

## ⚙️ Docker Compose

The project also includes orchestration configuration.

Start the services:

```bash
docker compose -f orchestration/docker-compose.yml up --build
```

Stop the services:

```bash
docker compose -f orchestration/docker-compose.yml down
```

---

## 🧪 Running Tests

Run monitoring tests:

```bash
python monitoring/test_monitoring.py
```

Run RAG generator tests:

```bash
python rag/test_generator.py
```

---

## 📊 Running Evaluation

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

## 🧩 Engineering Principles

This project follows modern AI engineering principles:

### ✅ Modular Architecture

Each component has a clear responsibility and can be developed and tested independently.

### ✅ Retrieval First

The LLM does not operate alone. Relevant context is retrieved before generation.

### ✅ Evaluation Driven

System quality is measured using structured evaluation pipelines.

### ✅ Observable

System behavior can be monitored and analyzed through logs and metrics.

### ✅ Reproducible

The project includes structured data, evaluation results, and configuration files.

### ✅ Production Oriented

The system is designed with deployment, monitoring, testing, and maintainability in mind.

---

## 🗺️ Future Improvements

Potential future improvements include:

* 🔬 Advanced Cross-Encoder Reranking
* 📚 Medical Answer Citations
* 🧠 Improved Hallucination Detection
* 💬 Conversation Memory
* 🔐 User Authentication
* ☁️ Cloud Deployment
* 🔄 Automated CI/CD Pipeline
* 🔍 Advanced Distributed Tracing
* 🧪 A/B Testing for Retrieval Strategies
* 📈 Continuous Evaluation
* 🗄️ Production Vector Database Integration
* 📊 Advanced Analytics
* ⚡ API Performance Optimization

---

## ⚠️ Medical Disclaimer

This project is intended for educational and research purposes only.

It is **not a replacement for professional medical advice, diagnosis, or treatment**.

Always consult a qualified healthcare professional for medical decisions.

---

## 👩‍💻 Author

### Marina Wahid

**Artificial Intelligence Developer | Generative AI | RAG | Machine Learning | Flutter Developer**

Building intelligent applications using:

* 🤖 Generative AI
* 🧠 Large Language Models
* 🔍 Retrieval-Augmented Generation
* 📊 Machine Learning
* 🐍 Python
* 📱 Flutter

---

<p align="center">

⭐ If you find this project useful, consider giving it a star!

</p>
```
