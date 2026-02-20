# Contributing to Virtual Try-On AI

Thank you for your interest in contributing to Virtual Try-On AI! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Follow best practices for code quality
- Test your changes before submitting
- Document your changes clearly

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- NVIDIA GPU (optional, for faster testing)

### Development Setup

1. **Fork the repository**
   ```bash
   # Go to https://github.com/yourusername/virtual-tryon-ai
   # Click "Fork"
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/virtual-tryon-ai.git
   cd virtual-tryon-ai
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```

## Making Changes

### Code Style

We follow PEP 8 with some modifications:

```python
# Good
def classify_garment(image: Image.Image) -> Dict[str, str]:
    """Classify clothing type from an image."""
    result = model.predict(image)
    return result

# Bad
def classificiation_garment(image):
    # classify clothes
    return model.predict(image)
```

### Formatting with Black

```bash
# Format all files
black backend frontend tests

# Check without changing
black --check backend frontend tests
```

### Linting with Flake8

```bash
# Check linting
flake8 backend frontend --max-line-length=100

# Ignore specific warnings
flake8 backend frontend --ignore=E203,W503
```

### Type Checking with Mypy

```bash
# Check type hints
mypy backend frontend

# Ignore in specific file
# type: ignore
```

## Creating Tests

Add tests for new features in `tests/`:

```python
import unittest
from backend.models import ClothingClassifier

class TestClothingClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = ClothingClassifier()

    def test_classify_returns_dict(self):
        result = self.classifier.classify(test_image)
        self.assertIsInstance(result, dict)
        self.assertIn('garment_type', result)

if __name__ == '__main__':
    unittest.main()
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_models.py::TestClothingClassifier

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=backend --cov=frontend
```

## Git Workflow

### Before Committing

```bash
# 1. Format code
black backend frontend tests

# 2. Lint code
flake8 backend frontend

# 3. Run type checking
mypy backend frontend

# 4. Run tests
pytest tests/ -v
```

### Commit Message Guidelines

```
[TYPE] Short description (50 chars)

Longer description explaining what changed and why.
Keep lines under 72 characters.

- Point 1
- Point 2

Fixes #123  # Link to issue if applicable
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Example**:
```
feat: Add semantic masking for garment placement

- Implement SemanticMasker class with region detection
- Add morphological operations for mask refinement
- Add tests for mask generation

Fixes #45
```

### Push and Create Pull Request

```bash
# Push your branch
git push origin feature/your-feature-name

# Go to GitHub and create a Pull Request
# - Write clear description
# - Link related issues
# - Ensure CI/CD passes
```

## Documentation

### Code Comments

```python
def generate_try_on(
    self,
    person_image: Image.Image,
    garment_image: Image.Image,
) -> Dict:
    """
    Generate virtual try-on result.

    This method processes both images through the ML pipeline,
    automatically detecting the garment type and generating
    a realistic rendering.

    Args:
        person_image: PIL Image of the person
        garment_image: PIL Image of the garment

    Returns:
        Dictionary with:
            - result_image: Generated try-on image
            - garment_detected: Detected garment type
            - confidence: Detection confidence score

    Raises:
        ValueError: If images are invalid
        RuntimeError: If model inference fails
    """
```

### README Updates

If adding new features, update README.md:

```markdown
## New Feature

Description of the feature.

Usage:
```python
# Example code
```

## Reporting Issues

### Bug Reports

Include:
1. Python version and OS
2. Steps to reproduce
3. Expected vs. actual behavior
4. Error traceback
5. Screenshots if applicable

**Template**:
```markdown
**Describe the bug**
A clear description of the bug.

**To Reproduce**
1. Step 1
2. Step 2
3. ...

**Expected behavior**
What should happen.

**Actual behavior**
What actually happens.

**Environment**
- OS: 
- Python: 
- GPU: 
```

### Feature Requests

**Template**:
```markdown
**Is your feature related to a problem?**
Describe the problem.

**Describe the solution you'd like**
How would you like it to work?

**Additional context**
Any other context.
```

## Review Process

1. **Automated Checks**
   - GitHub Actions tests must pass
   - Code coverage should be maintained

2. **Code Review**
   - At least one maintainer review required
   - Address comments and feedback

3. **Merge**
   - Squash commits if requested
   - Delete feature branch after merge

## Performance Benchmarking

When optimizing code, include benchmarks:

```python
import time

start = time.time()
# Code to benchmark
elapsed = time.time() - start
print(f"Time: {elapsed:.3f}s")
```

## Documentation Updates

Update docs in:
- `README.md` - Overview and usage
- `DEPLOYMENT_GUIDE.md` - Deployment instructions
- Code docstrings - Implementation details
- Issue templates - For consistent issue reporting

## Release Process

1. Update version in `config.py` and `setup.py`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v1.0.0`
4. Push tag: `git push origin v1.0.0`
5. Create GitHub Release with changelog

## Getting Help

- **Discussions**: Ask questions in GitHub Discussions
- **Issues**: Report bugs in GitHub Issues
- **Email**: Contact maintainers directly
- **Documentation**: Check README and DEPLOYMENT_GUIDE

## Thank You!

We appreciate all contributions, whether code, documentation, or bug reports!

---

**Last Updated**: February 2026
