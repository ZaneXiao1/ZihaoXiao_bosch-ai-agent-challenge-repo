# Project Structure

```
├── README.md                           # Full project documentation
├── QUICKSTART.md                       # Quick start guide for local setup
├── SCALABILITY.md                      # Scalability analysis and migration paths
├── PROJECT_STRUCTURE.md                # This file
│
└── me-engineering-assistant/           # Main application package
    ├── databricks.yml                  # Databricks Asset Bundle config (dev/prod targets)
    ├── pyproject.toml                  # Python package config and dependencies
    ├── .env.example                    # Environment variable template
    │
    ├── src/me_assistant/               # Core source code
    │   ├── config.py                   # Environment and model configuration
    │   ├── exceptions.py               # Custom exception hierarchy
    │   ├── agent/                      # LangGraph agent implementation
    │   │   ├── graph.py                # Agent graph definition and entry point
    │   │   ├── nodes.py                # Graph node logic (retrieve, generate, etc.)
    │   │   ├── prompts.py              # System and user prompt templates
    │   │   ├── state.py                # Agent state schema
    │   │   └── tools.py               # Retrieval tools for vector store search
    │   ├── documents/                  # Document processing pipeline
    │   │   ├── loader.py              # Markdown loader and chunking
    │   │   └── store.py               # Chroma vector store creation
    │   ├── data/                       # ECU technical documentation (source data)
    │   │   ├── ECU-700_Series_Manual.md
    │   │   ├── ECU-800_Series_Base.md
    │   │   └── ECU-800_Series_Plus.md
    │   ├── evaluation/                 # Evaluation framework
    │   │   ├── evaluator.py           # LLM-as-judge scoring engine
    │   │   ├── prompts.py             # Judge prompt templates
    │   │   └── retrieval_eval.py      # Retrieval quality evaluation
    │   └── model/                      # MLflow deployment wrapper
    │       └── mlflow_wrapper.py      # pyfunc model (predict interface for serving)
    │
    ├── scripts/                        # Executable scripts
    │   ├── chat_cli.py                # Interactive chat CLI (local development)
    │   ├── run_evaluation.py          # Tier 1: keyword pass/fail (10Q)
    │   ├── run_eval.py                # Tier 3: LLM-as-judge (60Q)
    │   ├── run_retrieval_eval.py      # Retrieval quality evaluation
    │   └── train_and_log_model.py     # Full pipeline: evaluate + log to MLflow
    │
    ├── tests/                          # Test suite
    │   ├── conftest.py                # Shared test fixtures
    │   ├── test_agent.py              # Agent integration tests
    │   ├── test_retrieval.py          # Retrieval accuracy tests
    │   ├── test_error_handling.py     # Exception handling tests
    │   ├── test_queries.csv           # Official 10-question test set
    │   └── test_queries_extended.csv  # Extended 60-question test set
    │
    └── evaluation_results.md          # Evaluation run output (10/10 pass record)
```
