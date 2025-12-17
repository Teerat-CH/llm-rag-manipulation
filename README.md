# llm-rag-manipulation
A custom RAG system to investigate the impact of different prompt injection techniques on downstream product recommendations and text classifiers to gauge the effectiveness of various defense mechanisms.

### Getting Started
Prerequisites
- Python 3.9+
- Gemini API key

**Setup**
```bash
# clone repository

git clone https://github.com/Teerat-CH/llm-rag-manipulation
cd llm-rag-manipulation

# create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
uv install
```

**Configure Gemini API**
Create a `.env` file at `llm-rag-manipulation/RAG`
```bash
GEMINI_API_KEY='YOUR_API_KEY'
```

### Playground
run
```bash
cd Playground
streamlit run main.py
# the webapp should start running on your localhost
```

### Project Structure
```bash
llm-rag-manipulation/
├── Playground/                # Streamlit app: demo for experimenting with RAG Manipulation
├── RAG/
│   ├── .env                   # Environment variables (e.g., GEMINI_API_KEY)
│   ├── RAG.py                 # RAG orchestrator: retrieval + generation pipeline
│   ├── index.py               # FAISS-based vector indexing and retrieval
│   ├── encode.py              # Text preprocessing and embedding generation
│   ├── generation.py          # LLM interface (e.g., Google Gemini, Mistral)
├── Classifier/
│   ├── classification.py      # Unified text classification API
│   ├── Bert/                  # Folder for anything related to Bert classifier training
│   └── XGBoost/               # Folder for anything related to Embeded-based XGBoost classifier training
└── Experiment/                # Overall Experimentation
```
