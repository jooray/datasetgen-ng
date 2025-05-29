# Dataset Generator NG

AI-powered dataset generator for creating question-answer pairs from text files for finetuning purposes. This tool processes text documents and generates structured datasets through a multi-phase approach.

## Features

- **Phase 1**: Generate meaningful questions from text chunks
- **Phase 2**: Generate answers using RAG (Retrieval-Augmented Generation)
- **Phase 3**: Quality approval checking with configurable prompts
- **Phase 4**: Export approved pairs to training formats (JSONL)
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
poetry run datasetgen path/to/text.txt

# Run specific phases
poetry run datasetgen path/to/text.txt --phase 1  # Generate questions
poetry run datasetgen path/to/text.txt --phase 2  # Generate answers
poetry run datasetgen path/to/text.txt --phase 3  # Check approvals
poetry run datasetgen path/to/text.txt --phase 4  # Export dataset

# Custom output and database paths
poetry run datasetgen text.txt --output my_dataset.jsonl --db my_dataset.db

# Enable verbose logging to see all API interactions
poetry run datasetgen text.txt --verbose
```

### Advanced Usage

```bash
# Use custom config file
poetry run datasetgen text.txt --config /path/to/custom-config.json

# Process specific phases with custom paths and verbose output
poetry run datasetgen text.txt \
  --phase 2 \
  --db existing_dataset.db \
  --config custom_config.json \
  --verbose
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

## Workflow

### Phase 1: Question Generation
- Splits input text into ~10k character chunks
- Generates multiple questions per chunk using structured output
- Stores questions with their source context in SQLite database

### Phase 2: Answer Generation
- Creates vector store from input text for RAG
- For each unanswered question, retrieves relevant context
- Generates answers using both original context and retrieved context
- Stores answers in database

### Phase 3: Approval Checking
- Runs configurable approval prompts on each question-answer pair
- Questions must pass ALL approval prompts to be approved
- Updates approval and processing status in database

### Phase 4: Dataset Export
- Exports all approved question-answer pairs to specified format
- Currently supports JSONL with "question" and "answer" keys

## Command Line Options

```
Usage: datasetgen [OPTIONS] TEXT_PATH

Arguments:
  TEXT_PATH  Path to the input text file  [required]

Options:
  --config PATH            Path to custom configuration file
  --phase {1,2,3,4,all}   Which phase to run (default: all)
  --output PATH           Output path for dataset (default: dataset.jsonl)
  --db PATH               Database path (default: dataset.db)
  --verbose               Enable verbose logging to stderr
  --help                  Show this message and exit.
```

## Verbose Logging

Use the `--verbose` flag to see detailed logging of all API interactions:

```bash
poetry run datasetgen text.txt --verbose 2> debug.log
```

This will log all prompts sent to the API and responses received, helping with debugging and prompt optimization.

## Database Schema

The tool uses SQLite to track progress:

```sql
CREATE TABLE qa_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    context TEXT NOT NULL,
    answer TEXT,
    approved INTEGER DEFAULT 0,
    processed INTEGER DEFAULT 0
);
```

## Support and value4value

If you like this project, I would appreciate if you contributed time, talent or treasure.

Time and talent can be used in testing it out, fixing bugs, spreading the word.

Treasure can be [sent back through here](https://juraj.bednar.io/en/support-me/).

## License

This project is licensed under the [GLWTSPL](https://github.com/me-shaon/GLWTPL/blob/master/NSFW_LICENSE)
