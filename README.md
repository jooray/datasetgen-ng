# Dataset Generator NG

AI-powered dataset generator for creating question-answer pairs from text files for finetuning purposes. This tool processes text documents and generates structured datasets through a multi-phase approach with FAISS-powered vector storage for improved RAG performance.

## Features

- **Phase 1**: Generate meaningful questions from text chunks and set up FAISS vector store
- **Phase 2**: Generate answers using RAG (Retrieval-Augmented Generation) with FAISS
- **Phase 3**: Quality approval checking with configurable prompts
- **Phase 4**: Export approved pairs to training formats (JSONL)
- **Phase addcontext**: Import text to FAISS vector store without generating questions
- **Duplicate prevention**: Automatic detection and prevention of re-importing the same text
- **Persistent vector storage**: FAISS vector stores are saved and reused across runs
- **Context-aware approval**: Optional context inclusion during approval checking
- **Verbose logging**: Detailed logging of all API interactions

## Installation

### Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- Venice AI API key
- Ollama (for embeddings)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd datasetgen-ng
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

3. **Set up your Venice AI API key:**
   ```bash
   export VENICE_API_KEY="your-api-key-here"
   ```

4. **Install Ollama and pull embedding model:**
   ```bash
   # Install Ollama (visit https://ollama.ai for instructions)
   ollama pull nomic-embed-text
   ollama pull llama3  # Default embedding model
   ```

5. **Create configuration file (optional):**
   ```bash
   cp datasetgen.json-example ~/.datasetgen.json
   # Edit the config file to customize prompts and settings
   ```

## Usage

### Basic Usage

```bash
# Run all phases
poetry run datasetgen --dataset path/to/text.txt

# Run specific phases
poetry run datasetgen --dataset path/to/text.txt --phase 1  # Generate questions + setup vector store
poetry run datasetgen --phase 2  # Generate answers using FAISS RAG (no dataset needed)
poetry run datasetgen --phase 3  # Check approvals (no dataset needed)
poetry run datasetgen --phase 4  # Export dataset (no dataset needed)
poetry run datasetgen --dataset path/to/text.txt --phase addcontext  # Only import to vector store

# Custom output, database, and vector store paths
poetry run datasetgen --dataset text.txt --output my_dataset.jsonl --db my_dataset.db --vector-store my_vector_store

# Enable verbose logging to see all API interactions
poetry run datasetgen --dataset text.txt --verbose

# Phase 3 with context information for better approval checking
poetry run datasetgen --phase 3 --with-context
```

### Advanced Usage

```bash
# Use custom config file
poetry run datasetgen --dataset text.txt --config /path/to/custom-config.json

# Process specific phases with custom paths and verbose output
poetry run datasetgen --phase 2 \
  --db existing_dataset.db \
  --vector-store existing_vector_store \
  --config custom_config.json \
  --verbose

# Import additional context to existing vector store
poetry run datasetgen --dataset additional_text.txt \
  --phase addcontext \
  --vector-store existing_vector_store \
  --db existing_dataset.db

# Reprocess previously rejected pairs with context
poetry run datasetgen --phase 3 --reprocess-rejected --with-context

# Context-aware approval checking
poetry run datasetgen --phase 3 --with-context --verbose
```

## Configuration

The tool uses a JSON configuration file to customize behavior. By default, it looks for `~/.datasetgen.json`.

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `prompts.question_generation` | string | Default question prompt | Prompt for generating questions from text chunks |
| `prompts.answer_generation` | string | Default answer prompt | Prompt for generating answers using RAG |
| `prompts.approval_prompts` | array | Default approval prompts | List of prompts for quality checking |
| `context_size` | integer | 128000 | Maximum tokens per chunk for processing |
| `model` | string | "gpt-3.5-turbo" | Model name for text generation |
| `embedding_model` | string | "llama3" | Ollama model for embeddings |
| `question_chunk_size` | integer | 10000 | Chunk size for question generation |

### Example Configuration

```json
{
  "prompts": {
    "question_generation": "Generate meaningful questions that can be answered based on the provided text. Focus on questions that would help someone understand the key concepts, facts, and insights from the text. Return 3-5 questions as a JSON array of strings.",
    "answer_generation": "Based on the provided context, answer the following question comprehensively and accurately. Use only information from the context provided.",
    "approval_prompts": [
      "Look at this question and answer. If the question is answered by the answer, just return PASS without quotes and nothing else. Otherwise explain what the problem is.",
      "Check if the answer is factually correct based on the context. If correct, return PASS. Otherwise explain the issue."
    ]
  },
  "context_size": 128000,
  "model": "gpt-3.5-turbo",
  "api_base": "https://api.venice.ai/api/v1",
  "embedding_model": "llama3",
  "question_chunk_size": 10000
}
```

## Command Line Options

```
Usage: datasetgen [OPTIONS]

Options:
  --dataset PATH              Path to input text file (required for phases 1, addcontext, all)
  --config PATH               Path to custom configuration file
  --phase {1,2,3,4,addcontext,all}  Which phase to run (default: all)
  --output PATH               Output path for dataset (default: dataset.jsonl)
  --db PATH                   Database path (default: dataset.db)
  --vector-store PATH         Vector store path (default: vector_store)
  --verbose                   Enable verbose logging to stderr
  --reprocess-rejected        Reprocess previously rejected question-answer pairs in phase 3
  --with-context              Include context information during approval checking (phase 3)
  --help                      Show this message and exit.
```

### Phase-Specific Requirements

- **Phases 1, addcontext, all**: Require `--dataset` argument with path to text file
- **Phases 2, 3, 4**: Work with existing database and vector store, no dataset file needed
- **Phase 3 with --with-context**: Uses both database context and vector store for enhanced approval checking

## Workflow

### Phase addcontext: Vector Store Import
- Imports text into FAISS vector store for later RAG operations
- Checks for duplicates using text hash to avoid re-importing same content
- Saves persistent FAISS index to disk for reuse

### Phase 1: Question Generation + Vector Store Setup
- Splits input text into ~10k character chunks
- Sets up FAISS vector store with duplicate detection
- Generates multiple questions per chunk using structured output
- Stores questions with their source context in SQLite database

### Phase 2: Answer Generation with FAISS RAG
- Loads or reuses existing FAISS vector store
- For each unanswered question, retrieves relevant context using FAISS similarity search
- Generates answers using both original context and retrieved context
- Stores answers in database

### Phase 3: Approval Checking
- Runs configurable approval prompts on each question-answer pair
- Questions must pass ALL approval prompts to be approved
- **With --with-context**: Includes original context and additional vector store context in approval prompts
- **Without --with-context**: Uses only question-answer pairs (original behavior)
- Updates approval and processing status in database

### Phase 4: Dataset Export
- Exports all approved question-answer pairs to specified format
- Currently supports JSONL with "question" and "answer" keys

## Vector Store and Duplicate Prevention

The tool uses FAISS for efficient similarity search and implements automatic duplicate prevention:

- **Text Hashing**: Each input text is hashed (SHA-256) for duplicate detection
- **Import Tracking**: Database tracks which text hashes have been imported
- **Persistent Storage**: FAISS indices are saved to disk and reused across runs
- **Automatic Deduplication**: Same text content is never imported twice

## Verbose Logging

Use the `--verbose` flag to see detailed logging of all API interactions and vector store operations:

```bash
poetry run datasetgen text.txt --verbose 2> debug.log
```

This will log all prompts sent to the API, responses received, and vector store operations, helping with debugging and prompt optimization.

## Database Schema

The tool uses SQLite to track progress and imported datasets:

```sql
CREATE TABLE qa_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    context TEXT NOT NULL,
    answer TEXT,
    approved INTEGER DEFAULT 0,
    processed INTEGER DEFAULT 0,
    rejection_reason TEXT
);

CREATE TABLE imported_datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_hash TEXT UNIQUE NOT NULL,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Parallel processing

It is safe to run one of each Phase 1, Phase 2 and Phase 3 processes at the same time, after
you add all the datasets to the vector store (addcontext phase).

You can also run multiple Phase 1 processes at the same time with different datasets.

## Support and value4value

If you like this project, I would appreciate if you contributed time, talent or treasure.

Time and talent can be used in testing it out, fixing bugs, spreading the word.

Treasure can be [sent back through here](https://juraj.bednar.io/en/support-me/).

## License

This project is licensed under the [GLWTSPL](https://github.com/me-shaon/GLWTPL/blob/master/NSFW_LICENSE)
