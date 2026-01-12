# 🚀 RepoPilot – Local LLM Agent for Codebase Intelligence

**Offline-first codebase analysis using semantic search + local LLMs. No cloud APIs. Works on your machine.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Agastya910/RepoPilot?style=social)](https://github.com/Agastya910/RepoPilot)

---

## 📸 Screenshots

### CLI startup and local indexing
![RepoPilot CLI startup](assets/cli-startup.jpg)

### Cloning & analyzing a GitHub repo 
![RepoPilot analyzing Go repo](assets/cli-analysis.jpg)

---

## What is RepoPilot?

RepoPilot is an **AI agent for understanding codebases** that runs entirely on your local machine. Ask it questions about any repository (yours or public GitHub repos), and it will:

- 🔍 **Semantic search** through code using vector embeddings (FAISS + Ollama)
- 📚 **Generate architecture summaries** without needing documentation
- 🧠 **Answer natural language questions** about your codebase
- 🌐 **Clone & analyze GitHub repos** locally (supports public repos)
- 🔒 **100% offline** – your code never leaves your machine

### Example Queries

```text
RepoPilot > what is this project about?
→ Analyzes the repo and explains its purpose

RepoPilot > where is authentication handled?
→ Searches semantically and shows relevant code

RepoPilot > are you able to correct errors in this codebase?
→ Identifies issues and suggests improvements

RepoPilot > Analyze https://github.com/golang/go
→ Clones, indexes 10k+ files, ready for questions
```

---

## 🖥️ Tested Environment

RepoPilot has been developed and tested in:

- **OS**: Windows 11 with WSL2 (Ubuntu 22.04)
- **CPU**: Intel Core i7 (8 cores) @ 2.5GHz
- **RAM**: 16GB
- **GPU**: NVIDIA GTX/RTX (4-8GB VRAM) – Ollama utilizes ~30% during inference
- **Python**: 3.10 (works with 3.9+)
- **Ollama Models**: 
  - `qwen:7b` (7B parameters, ~4GB) for reasoning
  - `nomic-embed-text` (137M parameters) for embeddings
- **Shell**: bash (inside WSL2)

**Cross-platform support:**
- ✅ Linux (native)
- ✅ macOS (Apple Silicon / Intel)
- ✅ Windows (via WSL2, recommended)

---

## 🚀 Quick Start (5 minutes)

### 1. Install Prerequisites

- **Python 3.9+**
- **Ollama** – download from [ollama.ai](https://ollama.ai)

### 2. Install Ollama Models

```bash
# Required: Embedding model
ollama pull nomic-embed-text

# Required: LLM for reasoning (7B model, ~4GB)
ollama pull qwen:7b

# Optional: Faster alternatives
ollama pull all-minilm          # Faster embeddings
ollama pull mistral             # Alternative LLM
```

### 3. Start Ollama Service

```bash
# Linux / WSL
ollama serve

# macOS (if installed via Homebrew)
brew services start ollama

# Windows: Start Ollama app from Start menu
```

### 4. Clone and Install RepoPilot

```bash
git clone https://github.com/Agastya910/RepoPilot.git
cd RepoPilot

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows (WSL/Git Bash): source venv/bin/activate
                                # Windows (CMD): venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Run RepoPilot

```bash
# Analyze current directory
python cli.py .

# Or analyze any local repository
python cli.py /path/to/your/repo
```

### 6. Ask Questions!

```text
How can I help you? (or 'exit' to quit) > what does this repo do?

[PLANNING]...
[RETRIEVAL] 🔍 Searching for: 'what does this repo do?' (top 5)
[EXECUTING]...

============================================================
This project is an AI-powered codebase analysis tool that...
============================================================
```

### 7. Analyze GitHub Repos

```text
RepoPilot > Analyze https://github.com/golang/go
RepoPilot > Find authentication in torvalds/linux
RepoPilot > What is React's component system?
```

RepoPilot will:
1. Clone the repo into `./analyzed_repo`
2. Scan and index all code files
3. Build semantic embeddings
4. Answer your questions

---

## ⚡ Performance

**First-time indexing** (one-time cost per repo):

| Repository | Files | Chunks | Index Time |
|-----------|-------|--------|------------|
| RepoPilot (self) | 20 | 78 | ~10 seconds |
| Go (golang/go) | 10,360 | 89,031 | ~3-5 minutes |
| React (facebook/react) | 2,100 | 15,000 | ~1-2 minutes |
| Linux Kernel | 92,210 | 500,000+ | ~15-20 minutes |

**Subsequent queries**: Instant (embeddings cached on disk)

Indexes are stored in:
- `./.repopilot_index` for local repos
- `./analyzed_repo/.repopilot_index` for cloned repos

See `OPTIMIZATION_GUIDE.md` for GPU tuning and faster model alternatives.

---

## ✨ Current Capabilities

- ✅ **Semantic code search** using embeddings + FAISS vector store
- ✅ **Natural language Q&A** about repositories
- ✅ **Architecture summaries** and metadata queries
- ✅ **GitHub URL detection** and automatic cloning
- ✅ **Multi-language support** (Python, JavaScript, Java, C/C++, Go, Rust, TypeScript, etc.)
- ✅ **Cross-platform** (Linux, macOS, Windows via WSL2)
- ✅ **Fully offline** (no cloud APIs, no data leakage)

### 🚧 Limitations & Roadmap

**Current limitations:**
- **Read-only mode**: RepoPilot analyzes and explains code but doesn't edit files (intentional for safety)
- **CLI only**: No GUI yet (web UI coming soon)
- **Public repos only**: Private GitHub repos require SSH keys (planned)
- **No streaming**: Responses arrive all at once (streaming mode in progress)

**Coming next:**
- ✏️ **Safe code editing** with diff preview and approval
- 💬 **Streaming responses** for better UX
- 🌐 **Web UI** with React frontend
- 🐳 **Docker support** for one-command deployment
- 🔄 **Git integration** (diffs, blame, history analysis)
- 🔐 **Private repo support** via SSH/tokens

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Interface                        │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │   Agent (Orchestrator) │
         └───┬─────────────────┬──┘
             │                 │
    ┌────────▼────────┐  ┌─────▼──────────┐
    │  Planner        │  │ Verifier       │
    │ (Route queries) │  │ (Validate out) │
    └────────┬────────┘  └────────────────┘
             │
    ┌────────▼─────────────┐
    │    Executor          │
    │ (Run tools locally)  │
    └────┬────────┬────┬───┘
         │        │    │
    ┌────▼──┐  ┌──▼────▼─┐
    │Indexer│  │Repo     │
    │(FAISS)│  │Scanner  │
    └───────┘  └─────────┘
         │
    ┌────▼──────────┐
    │  Ollama       │
    │  - Embeddings │
    │  - LLM        │
    └───────────────┘
```

**Tech Stack:**
- **Python 3.10+** (core language)
- **Ollama** (local LLM inference)
- **LangChain** (LLM orchestration and embeddings)
- **FAISS** (Facebook AI Similarity Search for vector storage)
- **Git** (repository cloning)

---

## 📁 Project Structure

```
RepoPilot/
├── cli.py                    # CLI entry point
├── agent/
│   ├── planner.py            # Query classification and planning
│   ├── executor.py           # Tool execution (scan, search, clone)
│   └── verifier.py           # Output validation
├── core/
│   ├── indexer.py            # FAISS index builder and searcher
│   └── query_router.py       # Query type classification
├── llm/
│   └── local_llm_client.py   # Ollama LLM client
├── tools/
│   ├── repo_scanner.py       # File tree walker with language detection
│   ├── code_search.py        # Text search utilities
│   ├── file_io.py            # Safe file read/write
│   ├── diff_writer.py        # Diff generation (future edits)
│   └── github_helper.py      # GitHub cloning helpers
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── OPTIMIZATION_GUIDE.md     # GPU tuning and performance tips
└── assets/                   # Screenshots
```

---

## 🛠️ Development

### Contributing

```bash
# Fork and clone your fork
git clone https://github.com/Agastya910/RepoPilot.git
cd RepoPilot

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies (including dev tools)
pip install -r requirements.txt
pip install black pytest mypy  # Dev dependencies

# Run tests
pytest tests/

# Format code
black .

# Type checking
mypy .
```

### Running Tests

```bash
pytest tests/ -v
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋 FAQ

**Q: Does this send my code to the cloud?**  
A: No. Everything runs locally on your machine. Ollama runs on `localhost`, and your code never leaves your system.

**Q: Can I use this offline?**  
A: Yes, completely. Once Ollama models are downloaded, you can disconnect from the internet.

**Q: What if I don't have a GPU?**  
A: Ollama will automatically fall back to CPU. It will be slower but still works.

**Q: How much disk space do I need?**  
A: Models: ~20GB (Ollama cache). Indexes: ~100MB per 10k files. Cloned repos: varies.

**Q: Can I use different LLMs?**  
A: Yes! Any model available in Ollama works. Set via environment variable:
```bash
export REPOPILOT_LLM=mistral
```

**Q: Why is the first query slow?**  
A: The first run builds the embedding index (one-time cost). Subsequent queries load from cache and are instant.

**Q: Can it edit my code?**  
A: Not yet. RepoPilot is currently read-only for safety. Code editing with diff preview is planned.

---

## 🌟 Acknowledgments

- **Ollama** for making local LLM inference accessible
- **LangChain** for LLM orchestration primitives
- **FAISS** (Meta AI) for efficient vector search
- Inspired by GitHub Copilot, Cursor, and other AI coding assistants
- Built with privacy-first principles

---

## 👨‍💻 Author

**Agastya Todi**

- 🌐 GitHub: [@Agastya910](https://github.com/Agastya910)
- 💼 LinkedIn: [Agastya Todi](https://www.linkedin.com/in/agastya-todi)


---

## ⭐ Support

If you find RepoPilot useful, please consider:

- ⭐ **Starring this repository**
- 🐛 **Reporting bugs** via Issues
- 💡 **Suggesting features** via Discussions
- 📣 **Sharing** with other developers

---

**Built with ❤️ for developers who value privacy and local-first AI tools.**
