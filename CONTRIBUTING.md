# Contributing to Booking.com Hotel Analytics

Thank you for your interest in contributing! This document provides guidelines and instructions.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Guidelines](#pull-request-guidelines)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the community

## Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/booking-hotel-analytics.git
   cd booking-hotel-analytics
   ```

3. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   pip install -e ".[dev]"
   pre-commit install
   ```

4. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### 1. Make Changes

- Write clean, readable code
- Add type hints
- Include docstrings (Google style)
- Write tests for new features

### 2. Run Quality Checks

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test
```

### 3. Commit Changes

Follow conventional commits:

```bash
git commit -m "feat: add new regression model"
git commit -m "fix: resolve data loading bug"
git commit -m "docs: update README installation section"
git commit -m "test: add unit tests for metrics"
git commit -m "refactor: improve model architecture"
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

## Coding Standards

### Python Style

- Follow PEP 8
- Use Black for formatting (line length: 120)
- Use isort for imports
- Use type hints

### Example

```python
from typing import List, Optional
import torch.nn as nn


class RegressionModel(nn.Module):
    """Base class for regression models."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """Initialize the model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.layers(x)
```

## Testing

### Writing Tests

```python
import pytest
from src.models.regression import RidgeRegression


def test_ridge_regression_initialization():
    """Test model initialization."""
    model = RidgeRegression(alpha=1.0)
    assert model.alpha == 1.0


@pytest.mark.parametrize("alpha,expected", [
    (0.1, 0.95),
    (1.0, 0.90),
    (10.0, 0.85),
])
def test_ridge_regression_performance(alpha, expected):
    """Test model performance with different alpha values."""
    model = RidgeRegression(alpha=alpha)
    # ... training and evaluation code
    assert score >= expected
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/unit/test_models.py

# With coverage
pytest --cov=src

# Parallel execution
pytest -n auto
```

## Pull Request Guidelines

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. **Automated Checks**: CI must pass
2. **Code Review**: At least one approval required
3. **Testing**: Verify functionality manually if needed
4. **Merge**: Squash and merge to main

## Questions?

Open an issue or contact maintainers at gausseuler159357@gmail.com
