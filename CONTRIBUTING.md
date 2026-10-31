# Contributing to Ceft

Thank you for your interest in contributing to **Ceft**! We welcome contributions, bug reports, feature requests, and documentation improvements.

---

## Code of Conduct

Please maintain a welcoming, respectful, and collaborative environment for all contributors.

---

## Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/Yug-dev0/CEFT.git
   cd CEFT
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # On Windows: .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Set Up Environment Variables:**
   ```bash
   cp .env.example .env
   ```

5. **Run the Test Suite:**
   ```bash
   python3 -m pytest tests -v
   ```

---

## Pull Request Guidelines

1. **Branch Naming**: Use descriptive branch names like `feat/video-subtitles` or `fix/ocr-bbox-parsing`.
2. **Tests**: Add unit or integration tests in `tests/` for any new feature or bugfix.
3. **Linting**: Ensure code adheres to standards:
   ```bash
   ruff check --select E9,F63,F7,F82 .
   ```
4. **Clean Commits**: Write clear and descriptive commit messages.

---

## Questions and Support

- **Author**: Yug Patel
- **GitHub**: [@Yug-dev0](https://github.com/Yug-dev0)
- **Email**: [yug.dev04@gmail.com](mailto:yug.dev04@gmail.com)
- **Issues**: [GitHub Issues](https://github.com/Yug-dev0/CEFT/issues)
