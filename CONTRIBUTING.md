# Contributing to Banana Ripeness Detection

Thank you for your interest in contributing! 🎉

## How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/AYOCODEE/Banana-Ripeness-detection-with-knowledge-distillation/issues) first.
2. Open a new issue using the **Bug Report** template.
3. Include steps to reproduce, expected vs actual behaviour, and your environment details.

### Suggesting Features

Open a new issue using the **Feature Request** template and describe your idea clearly.

### Submitting Code

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes following the guidelines below.
4. Run the tests: `pytest tests/ -v`
5. Format your code: `black src/ tests/ app.py`
6. Lint your code: `flake8 src/ tests/ app.py --max-line-length=100`
7. Commit your changes: `git commit -m 'feat: describe your change'`
8. Push to your fork and open a Pull Request.

## Development Setup

```bash
git clone https://github.com/AYOCODEE/Banana-Ripeness-detection-with-knowledge-distillation.git
cd Banana-Ripeness-detection-with-knowledge-distillation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) with a max line length of 100.
- Use [black](https://black.readthedocs.io/) for formatting.
- Write docstrings for all public functions and classes.
- Add type hints where practical.

## Testing

All new code should include unit tests under `tests/`. Run:

```bash
pytest tests/ -v
```

## Commit Messages

Use the [Conventional Commits](https://www.conventionalcommits.org/) style:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test changes
- `refactor:` for refactoring

## Code of Conduct

Please be respectful and inclusive. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.
