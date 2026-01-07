#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QGAI Quantum Financial Modeling - TReDS MVP
Setup Script

This script allows the package to be installed via pip:
    pip install -e .

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "Hybrid Classical-Quantum TReDS Invoice Fraud Detection System"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

setup(
    name="qfm-treds-mvp",
    version="1.0.0",
    author="QGAI Quantum Financial Modeling Team",
    author_email="qfm@qgai.com",
    description="Hybrid Classical-Quantum TReDS Invoice Fraud Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qgai/qfm-treds-mvp",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "notebooks"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=23.0.0",
            "isort>=5.10.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "treds-fraud-detect=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
