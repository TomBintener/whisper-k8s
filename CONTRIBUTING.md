# Contributing to whisper-k8s

Thank you for your interest in contributing to **whisper-k8s**! We welcome community contributions, bug fixes, performance improvements, and documentation enhancements.

---

## 1. Development Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/<your-username>/whisper-k8s.git
   cd whisper-k8s
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   # Install development and linting tools
   pip install -r requirements-dev.txt
   ```

4. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   ```

---

## 2. Running Tests & Benchmarks

`whisper-k8s` is architected with a **zero-dependency test framework**. All unit and regression tests run in standard Python without needing external services or database connections:

```bash
# Run the complete test suite (116 tests)
python -m unittest discover tests -v
```

### Running Performance Benchmarks
To measure P0 inference speedup, subtitle converter throughput, and disk I/O latency:
```bash
python -m unittest tests/test_performance_benchmark.py -v
```

### Validating Docker Compose Configurations
Ensure that all container orchestration profiles remain valid:
```bash
docker compose config
docker compose --profile gpu config
docker compose --profile redis config
```

---

## 3. Code Quality & Standards

- **Formatting & Linting**: We use [Ruff](https://astral.sh/ruff) targeting Python 3.10+ with a 120-character line length:
  ```bash
  ruff check app tests scripts
  ```
- **Docstrings & Comments**: Preserve in-code documentation and provide clear function docstrings describing parameters, return values, and edge case assumptions.
- **Security-First**: Any new HTTP endpoints or external URL handling must incorporate SSRF validation and path safety checks.

---

## 4. Git Commit Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` A new feature (e.g. `feat(queue): add RabbitMQ task queue provider`)
- `fix:` A bug fix (e.g. `fix(bridge): handle missing status files gracefully`)
- `perf:` Performance optimizations (e.g. `perf(vad): accelerate silence detection`)
- `docs:` Documentation changes (e.g. `docs(api): document webhook retry backoff`)
- `test:` Adding or updating tests (e.g. `test(e2e): add chunk boundary edge cases`)
- `ci:` Changes to GitHub Actions or build scripts (e.g. `ci: add matrix test on Python 3.12`)
- `chore:` Maintenance tasks or dependency updates

---

## 5. Pull Request Checklist

Before submitting a Pull Request:
- [ ] All 116 tests pass: `python -m unittest discover tests -v`.
- [ ] Linter checks pass: `ruff check app tests scripts`.
- [ ] Compose syntax validates: `docker compose config`.
- [ ] Relevant documentation in `docs/` and `README.md` is updated.
- [ ] PR description clearly explains the motivation, changes made, and verification steps.
