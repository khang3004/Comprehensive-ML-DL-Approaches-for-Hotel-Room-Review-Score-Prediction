from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="booking-hotel-analytics",
    version="1.0.0",
    author="Khang et al.",
    author_email="gausseuler159357@gmail.com",
    description="Professional ML/DL System for Booking.com Hotel Analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khang3004/Comprehensive-ML-DL-Approaches-for-Hotel-Room-Review-Score-Prediction.git",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "mlflow": ["mlflow>=1.26.0"],
        "hydra": ["hydra-core>=1.2.0", "omegaconf>=2.2.0"],
        "all": [
            "mlflow>=1.26.0",
            "hydra-core>=1.2.0",
            "omegaconf>=2.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hotel-analytics=src.tasks.main:main",
        ],
    },
)
