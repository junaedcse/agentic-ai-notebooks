# Agentic RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph that intelligently routes queries to appropriate tools and provides contextual responses.

##  Features

- **Intelligent Tool Selection**: Automatically determines when to use RAG vs. direct knowledge
- **Multi-Tool Support**: Integrates document search and custom data retrieval
- **Conversational Memory**: Maintains context across multiple interactions
- **Rate Limit Handling**: Built-in retry logic with exponential backoff
- **FAISS Vector Store**: Efficient similarity search for document retrieval
- **Agentic Workflow**: Uses LangGraph for orchestrating complex agent behaviors

##  Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git (optional, for cloning)

##  Installation

### 1. Clone or Download the Repository

```bash
# If using git
git clone <your-repository-url>
cd agentic-rag

# Or download and extract the ZIP file
```

### 2. Create a Virtual Environment

```bash
# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
```

Or export directly in your terminal:

```bash
# On macOS/Linux
export OPENAI_API_KEY='your_openai_api_key_here'

# On Windows (Command Prompt)
set OPENAI_API_KEY=your_openai_api_key_here

# On Windows (PowerShell)
$env:OPENAI_API_KEY='your_openai_api_key_here'
```

##  Project Structure

```
agentic-rag/
├── agentic_rag_complete.ipynb  # Main notebook
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .env                         # Environment variables (create this)
├── data/                        # Your documents (optional)
│   └── sample_docs.txt
└── .venv/                       # Virtual environment (created during setup)
```

##  Usage

### Quick Start

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook agentic_rag_complete.ipynb
   ```

2. **Run the cells sequentially** to:
   - Initialize the vector store
   - Set up the agent
   - Test with sample queries

### Example Queries

```python
# Single tool usage
answer = run_agent(
    "What are recent developments in AI?",
    thread_id="test1"
)

# Multiple tool usage
answer = run_agent(
    "Tell me about AI predictions and also show me some album data",
    thread_id="test2"
)

# Direct knowledge (no tools)
answer = run_agent(
    "What is the capital of France?",
    thread_id="test3"
)
```

### Key Functions

- **`run_agent(query, thread_id, verbose)`**: Main function to interact with the agent
- **`search_documents(query, top_k)`**: Search the vector store
- **`get_albums()`**: Retrieve sample album data
- **`agent_node(state)`**: Core agent decision-making logic

##  Configuration

### Model Selection

By default, the system uses `gpt-4o-mini` for cost-effectiveness and higher rate limits:

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=4000
)
```

To use GPT-4o (requires higher rate limits):

```python
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=4000
)
```

### Adjusting Parameters

**Message History Limit**:
```python
MAX_MESSAGES = 15  # In agent_node function
```

**Document Truncation**:
```python
MAX_CHARS_PER_DOC = 1500  # In search_documents function
```

**Vector Search Results**:
```python
top_k = 3  # Number of documents to retrieve
```

**Retry Configuration**:
```python
max_retries = 3  # Number of retry attempts for rate limits
```

##  Troubleshooting

### Rate Limit Errors

If you encounter `RateLimitError: 429`:

1. **Switch to gpt-4o-mini** (recommended):
   ```python
   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
   ```

2. **Reduce context size**:
   - Decrease `MAX_MESSAGES`
   - Decrease `MAX_CHARS_PER_DOC`
   - Reduce `top_k` in search queries

3. **Check your OpenAI usage limits**:
   - Visit: https://platform.openai.com/account/rate-limits
   - Consider upgrading your plan

### ImportError or ModuleNotFoundError

```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### FAISS Installation Issues

**On macOS with Apple Silicon (M1/M2)**:
```bash
pip install faiss-cpu
```

**On Windows**:
```bash
pip install faiss-cpu
```

**For GPU support** (requires CUDA):
```bash
pip install faiss-gpu
```

### OpenAI API Key Issues

Verify your API key is set:
```python
import os
print(os.getenv('OPENAI_API_KEY'))  # Should not be None
```

##  Performance Tips

1. **Use gpt-4o-mini** for most use cases (95% quality, 10x cheaper)
2. **Batch similar queries** together when possible
3. **Cache frequently accessed documents** in memory
4. **Adjust `top_k`** based on your document size and query complexity
5. **Monitor token usage** to stay within rate limits

##  Customization

### Adding Your Own Documents

```python
# Load your documents
from langchain.document_loaders import TextLoader, PDFLoader

loader = TextLoader('path/to/your/documents.txt')
documents = loader.load()

# Split into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# Create vector store
vectorstore = FAISS.from_documents(splits, embeddings)
```

### Adding New Tools

```python
def your_custom_tool(param1: str, param2: int) -> str:
    """Your custom tool description"""
    # Your logic here
    return result

# Register in the tools dictionary
tools = {
    "search_documents": search_documents,
    "get_albums": get_albums,
    "your_custom_tool": your_custom_tool
}
```

### Modifying the System Prompt

Edit the `system_prompt` in the `agent_node` function to customize agent behavior:

```python
system_prompt = """You are a helpful AI assistant with access to tools.

Available tools:
- search_documents: Search through document knowledge base
- get_albums: Retrieve music album data
- your_custom_tool: Description of your tool

Your custom instructions here...
"""
```

##  Example Output

```
================================================================================
QUERY: Tell me about AI predictions and also show me some album data
================================================================================

================================================================================
EXECUTION TRACE
================================================================================

Tools Called:

  • search_documents:
    Document 1: AI predictions for 2024 include widespread adoption of...
    
  • get_albums:
    [{'title': 'Abbey Road', 'artist': 'The Beatles', 'year': 1969}...]

================================================================================

FINAL ANSWER
================================================================================
Based on recent AI predictions, we're seeing trends toward multimodal AI 
systems and increased focus on AI safety...

Regarding the albums, here are some notable examples from the database:
- Abbey Road by The Beatles (1969)
- Dark Side of the Moon by Pink Floyd (1973)
...
================================================================================
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [OpenAI](https://openai.com/) for the language models
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: junaed@example.com (update with your email)

## Version History

- **v1.0.0** (2024-11-11): Initial release
  - Basic agentic RAG implementation
  - Multi-tool support
  - Rate limit handling
  - Conversational memory

---

**Happy coding! **
