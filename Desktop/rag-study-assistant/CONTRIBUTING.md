# Contributing

Thank you for considering contributing to RAG Study Assistant!

## Development Setup

```bash
git clone https://github.com/rktm0604/RAG.git
cd RAG
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Code Style

- Use type hints for all function signatures
- Follow Google-style docstrings
- Run `python -m pytest tests/ -v` before submitting
- Keep all processing 100% local — no cloud API dependencies

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests for new functionality
3. Update README if adding dependencies or features
4. Ensure all existing tests pass
5. Open a PR with a clear description of changes
