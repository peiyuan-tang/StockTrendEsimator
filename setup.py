#!/usr/bin/env python3
"""
Setup configuration for Stock Trend Estimator Data Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stock-trend-estimator-pipeline",
    version="1.0.0",
    author="Stock Trend Estimator Team",
    author_email="dev@stocktrendestimator.com",
    description="Offline data collection pipeline using Flume Python for stock trend estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/peiyuan-tang/StockTrendEsimator",
    project_urls={
        "Bug Tracker": "https://github.com/peiyuan-tang/StockTrendEsimator/issues",
        "Documentation": "https://stocktrendestimator.readthedocs.io",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0",
        "yfinance>=0.2.0",
        "pandas>=1.5.0",
        "pandas-datareader>=0.10.0",
        "alpha-vantage>=2.3.1",
        "finnhub-python>=1.3.0",
        "newsapi>=0.1.1",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "feedparser>=6.0.0",
        "textblob>=0.17.0",
        "pyarrow>=10.0.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "apscheduler>=3.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "pylint>=2.15.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "database": [
            "psycopg2-binary>=2.9.0",
            "pymongo>=4.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-pipeline=data_pipeline.server.flume_server:main",
        ],
    },
)
