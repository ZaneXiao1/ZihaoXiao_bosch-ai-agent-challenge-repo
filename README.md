# Bosch AI Agent Challenge — Zihao Xiao

A production-ready, multi-source RAG agent that answers questions about Electronic Control Unit (ECU) specifications across the ECU-700 and ECU-800 product lines.

## Project

See [me-engineering-assistant/](me-engineering-assistant/) for the full implementation, including:

- Multi-source ReAct agent with LangGraph
- Dual FAISS vector store retrieval (ECU-700 / ECU-800)
- OpenAI + Databricks model provider abstraction
- MLflow pyfunc wrapper for deployment
- Evaluation suite (10/10 pass rate)

## Quick Start

```bash
cd me-engineering-assistant

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Run evaluation
python scripts/run_evaluation.py
```

## Documentation

- [Implementation Details](me-engineering-assistant/README.md)
- [Evaluation Results](me-engineering-assistant/evaluation_results.md)
- [Challenge Brief](ai-engineering-coding-challenge.md)
