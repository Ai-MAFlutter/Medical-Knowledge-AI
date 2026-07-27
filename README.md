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
<img src="https://img.shields.io/badge/Kestra-Orchestration-orange">
<img src="https://img.shields.io/badge/License-MIT-yellow">

</p>

<h1 align="center">🩺 Medical Knowledge AI</h1>

<p align="center">
<b>An end-to-end Retrieval-Augmented Generation (RAG) system for reliable medical knowledge retrieval and grounded question answering.</b>
</p>

<p align="center">
Built with modern AI engineering practices including hybrid retrieval, vector search, reranking, query rewriting, evaluation, observability, monitoring, workflow orchestration, and containerized deployment.
</p>

---

## 🚀 Overview

**Medical Knowledge AI** is an end-to-end AI Engineering project designed to retrieve and generate grounded answers from a curated medical knowledge base.

Unlike a simple chatbot that relies only on an LLM, this project implements a complete and measurable RAG pipeline.

The system combines:

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
- 🖥️ Streamlit Web Application
- 🐳 Docker Containerization
- ⚙️ Kestra Workflow Orchestration

The main goal is to build a **reliable, measurable, observable, reproducible, and production-oriented RAG system**.

---

## 🎥 Project Demo

<p align="center">

🎬 <b>Medical Knowledge AI Demo</b>

</p>

The demo showcases the complete application workflow, including:

- Asking medical questions
- Query rewriting
- Semantic retrieval
- Document reranking
- Context building
- Grounded answer generation
- Monitoring and observability dashboard

📹 **Watch the demo video:**

[▶️ Click here to watch the Medical Knowledge AI Demo](./assets/medical-knowledge-ai-demo.mp4)

---

## 🧠 Problem Description

Medical information is distributed across large collections of documents, making it difficult to quickly retrieve relevant and reliable information.

A simple LLM chatbot can generate fluent answers but may produce:

- Unsupported information
- Hallucinations
- Incorrect medical claims
- Answers that are not grounded in a trusted knowledge source

This project addresses the problem by building a **Retrieval-Augmented Generation system** that retrieves relevant medical knowledge before generating an answer.

The LLM is therefore provided with relevant retrieved context instead of relying only on its internal knowledge.

---

## 🏗️ System Architecture

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
```

---

## 🔄 Complete RAG Pipeline

The complete pipeline follows this workflow:

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

The retrieval layer supports multiple retrieval strategies.

### 🔹 Keyword Search

Keyword search is useful for exact medical terminology and important keywords.

```text
User Query
    │
    ▼
Keyword Matching
    │
    ▼
Relevant Medical Documents
```

This approach is useful when the query contains specific medical terms that should be matched directly.

---

### 🔹 BM25 Search

BM25 provides strong lexical retrieval performance and is useful when exact terminology matters.

```text
User Query
    │
    ▼
Tokenization
    │
    ▼
BM25 Scoring
    │
    ▼
Ranked Documents
```

BM25 is particularly useful for retrieving documents that contain important exact words from the user's query.

---

### 🔹 Vector Semantic Search

Documents are transformed into vector embeddings and searched semantically.

```text
Medical Documents
       │
       ▼
   Embeddings
       │
       ▼
 Vector Index
       │
       ▼
Semantic Similarity Search
       │
       ▼
Relevant Documents
```

Vector search allows the system to retrieve documents based on meaning rather than exact word matching.

This helps handle:

- Synonyms
- Different wording
- Similar medical concepts
- Semantic variations of questions

---

### 🔀 Hybrid Search

The system combines lexical and semantic retrieval strategies.

```text
┌─────────────────┐
│   BM25 Search   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BM25 Candidates │
└────────┬────────┘
         │
         ├──────────────┐
         │              │
         ▼              ▼
┌─────────────┐  ┌───────────────┐
│ Vector      │  │ Keyword       │
│ Search      │  │ Search        │
└──────┬──────┘  └───────┬───────┘
       │                 │
       └────────┬────────┘
                ▼
        Score Combination
                │
                ▼
          Hybrid Results
```

Hybrid retrieval improves robustness by combining:

- Exact lexical matching
- Semantic similarity
- Multiple retrieval signals

This allows the system to handle both exact medical terminology and semantically similar queries.

---

## 🎯 Document Reranking

After the initial retrieval stage, candidate documents are reranked to improve the final context quality.

```text
Initial Retrieval
        │
        ▼
Candidate Documents
        │
        ▼
Document Reranking
        │
        ▼
Most Relevant Context
```

Reranking improves the quality of the context passed to the LLM by prioritizing the most relevant retrieved documents.

---

## 🧩 Query Rewriting

The system includes a query rewriting stage that improves the original user question before retrieval.

```text
Original User Query
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

Query rewriting can help:

- Improve search clarity
- Expand ambiguous questions
- Improve retrieval recall
- Create better search queries

---

## 🤖 Retrieval-Augmented Generation

The final answer is generated using retrieved medical context.

```text
User Question
      │
      ▼
Query Rewriting
      │
      ▼
Hybrid Retrieval
      │
      ▼
Document Reranking
      │
      ▼
Relevant Context
      │
      ▼
Prompt Construction
      │
      ▼
Groq LLM
      │
      ▼
Grounded Answer
```

The system is designed to reduce unsupported answers by grounding the generation process in retrieved medical knowledge.

---

## 📚 Knowledge Base and Data Pipeline

The project includes a structured ingestion pipeline for processing medical knowledge.

```text
Raw Medical Data
       │
       ▼
Data Loading
       │
       ▼
Document Validation
       │
       ▼
Document Chunking
       │
       ▼
Metadata Processing
       │
       ▼
Embeddings
       │
       ▼
Vector Cache
       │
       ▼
Retrieval System
```

The ingestion pipeline includes:

- Loading medical knowledge
- Validating document structure
- Splitting documents into chunks
- Processing metadata
- Preparing data for retrieval
- Creating vector representations

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

## 🔎 Retrieval Evaluation

The retrieval layer was evaluated by comparing multiple retrieval strategies to measure their ability to retrieve relevant medical documents.

The evaluated approaches include:

- 🔤 Keyword Search
- 📚 BM25 Search
- 🧠 Vector Semantic Search
- 🔀 Hybrid Search

The evaluation process uses predefined medical queries with expected relevant documents.

```text
Evaluation Dataset
        │
        ▼
┌───────────────────────┐
│   Medical Queries     │
│   + Ground Truth      │
└───────────┬───────────┘
            │
            ▼
┌─────────────────────────────────────┐
│        Retrieval Strategies          │
├─────────────────────────────────────┤
│  Keyword Search                     │
│  BM25 Search                        │
│  Vector Search                      │
│  Hybrid Search                      │
└───────────────────┬─────────────────┘
                    │
                    ▼
          Retrieved Documents
                    │
                    ▼
          Evaluation Metrics
```

### Retrieval Evaluation Workflow

1. Prepare evaluation queries.
2. Define ground-truth relevant documents.
3. Run each retrieval strategy.
4. Collect retrieved results.
5. Compare retrieved documents against the ground truth.
6. Calculate retrieval quality metrics.
7. Save and analyze the evaluation results.

This makes it possible to compare different retrieval approaches and identify the strongest retrieval strategy for the medical knowledge base.

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

The answer generation component was evaluated to compare different prompting and generation approaches.

The evaluation focuses on:

- 🎯 Answer Relevance
- 📚 Grounding in Retrieved Context
- ✅ Factual Correctness
- 🧩 Answer Completeness
- 🚫 Reduction of Unsupported Information

Different prompt configurations and generation approaches can be compared to identify the most effective approach for producing grounded medical answers.

```text
Retrieved Context
        │
        ▼
┌────────────────────────┐
│   Prompt Configuration  │
│   / Generation Approach │
└────────────┬───────────┘
             │
             ▼
           LLM
             │
             ▼
      Generated Answer
             │
             ▼
      Quality Evaluation
```

### Generation Evaluation Workflow

1. Prepare evaluation questions.
2. Retrieve relevant medical context.
3. Generate answers using different prompt approaches.
4. Compare generated answers against expected answers or evaluation criteria.
5. Analyze answer quality.
6. Select the most effective generation approach.

This evaluation layer ensures that the system is not only retrieving relevant documents but also generating reliable answers based on the retrieved medical knowledge.

Relevant files:

```text
evaluation/
├── generation_evaluation.py
├── generation_results.json
└── evaluation_results.json
```

---

## 📊 Evaluation Results

The evaluation results are stored as structured JSON files to make experiments reproducible and easy to analyze.

The evaluation framework supports comparing:

| Component | Compared Approaches |
|---|---|
| Retrieval | Keyword Search, BM25, Vector Search, Hybrid Search |
| Generation | Different prompts and generation approaches |
| Answer Quality | Relevance, grounding, correctness, completeness |

This evaluation-driven approach helps identify the strongest configuration for the complete RAG pipeline.

---

## 📈 Monitoring and Observability

The project includes a monitoring layer for tracking application behavior and system performance.

The monitoring system tracks:

- 📊 Request Counts
- ⚡ Request Latency
- ✅ Success Rates
- ❌ Failed Requests
- 🔍 Retrieval Information
- 👍 User Feedback
- 🧠 System Behavior

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

The project includes an interactive Streamlit monitoring dashboard.

The dashboard provides visibility into:

- Total Requests
- Request Success Rate
- Failed Requests
- Average Latency
- Requests Over Time
- Latency Distribution
- Request Success vs Failure
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
Monitoring Dashboard
```

The dashboard makes it possible to observe system behavior and analyze application performance.

---

## 🖥️ Streamlit Web Application

The main user interface is built with Streamlit.

The application provides:

- 💬 Medical Question Input
- 🧠 Explanation Level Selection
- 🔍 Query Rewriting
- 🧠 Vector Search
- 📊 Document Reranking
- 🧩 Context Building
- 🤖 Grounded Answer Generation
- 📈 Monitoring Dashboard

Relevant structure:

```text
app/
├── app.py
├── style.css
└── pages/
    └── 2_📊_Monitoring_Dashboard.py
```

---

## ⚙️ Workflow Orchestration with Kestra

The project includes workflow orchestration using Kestra.

Kestra is used to define and orchestrate project workflows such as:

- Data Processing
- Pipeline Execution
- Evaluation Workflows
- AI Engineering Tasks

The orchestration configuration is located in:

```text
orchestration/
└── docker-compose.yml
```

---

## 🐳 Containerization

The project supports Docker-based deployment.

Docker helps provide a reproducible environment for running the application.

```text
Dockerfile
requirements.docker.txt
```

The application can be built and executed inside a Docker container.

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

- Python 3.11+

### AI and LLM

- Large Language Models
- Retrieval-Augmented Generation
- Generative AI
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
- NumPy
- JSON
- XML
- Document Chunking
- Metadata Processing

### Evaluation

- Retrieval Evaluation
- Generation Evaluation
- Ground-Truth Datasets
- Evaluation Pipelines
- Structured Evaluation Results

### Monitoring

- Request Logging
- Metrics Collection
- Latency Tracking
- Success Rate Monitoring
- User Feedback
- Streamlit Monitoring Dashboard

### Deployment and Infrastructure

- Docker
- Docker Compose
- Containerized Services
- Kestra Workflow Orchestration

### Interface

- Streamlit
- Interactive Web Application

---

## 🏆 Best Practices

This project applies several modern AI Engineering best practices.

### 🔀 Hybrid Retrieval

Combines lexical retrieval using BM25 with semantic vector search to improve recall and handle both exact medical terminology and semantic variations.

### 🎯 Document Reranking

Retrieved candidates are reranked before being passed to the LLM to improve context relevance.

### 🧩 Query Rewriting

User queries can be transformed into improved search queries to increase retrieval quality.

### 📊 Evaluation-Driven Development

Both retrieval quality and LLM generation quality are evaluated instead of relying only on manual testing.

### 📈 Observability

The system tracks request behavior, latency, success rates, retrieval information, and user feedback.

### 🧪 Reproducibility

The project includes structured datasets, evaluation scripts, saved results, dependency files, and Docker configuration.

### 🧱 Modular Architecture

Each major component has a clear responsibility:

- Ingestion
- Retrieval
- Reranking
- RAG Generation
- Evaluation
- Monitoring
- Application Interface

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ai-MAFlutter/Medical-Knowledge-AI.git
cd Medical-Knowledge-AI
```

### 2. Create a Virtual Environment

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

⚠️ Never commit your `.env` file or API keys to GitHub.

---

## ▶️ Running the Application

Run the Streamlit application:

```bash
streamlit run app/app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🐳 Docker

Build the Docker image:

```bash
docker build -t medical-knowledge-ai .
```

Run the container:

```bash
docker run -p 8501:8501 medical-knowledge-ai
```

---

## ⚙️ Docker Compose

Run the orchestration configuration:

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

## 🗺️ Future Improvements

Potential future improvements include:

- 🔬 Advanced Cross-Encoder Reranking
- 📊 Improved Retrieval Metrics
- 🧠 Better Hallucination Detection
- 📚 Medical Answer Citations
- 💬 Conversation Memory
- 🔐 User Authentication
- ☁️ Cloud Deployment
- 🔄 Automated CI/CD Pipeline
- 🔍 Advanced Tracing
- 🧪 A/B Testing for Retrieval Strategies
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