"""Setup script for metagenomic-assembly-optimization package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="metagenomic-assembly-optimization",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Tools for optimizing metagenomic assembly strategies through sample grouping analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/metagenomic-assembly-optimization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.6b0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sample-grouping-analysis=sample_grouping_analysis:main",
            "bias-aware-grouping-analysis=bias_aware_grouping_analysis:main",
            "scientific-grouping-analysis=scientific_grouping_analysis:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)