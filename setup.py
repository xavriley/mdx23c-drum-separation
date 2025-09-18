"""Setup script for MDX23C drum separation package."""

from setuptools import setup, find_packages
import os

# Read the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = """
# MDX23C Drum Separation

A pip installable package for drum separation using the MDX23C model.

## Installation

```bash
pip install mdx23c-drum-separation
```

## Usage

### Command Line Interface

```bash
mdx23c-separate --input_folder /path/to/audio/files --store_dir /path/to/output
```

### Python API

```python
from mdx23c import demix_audio

# Separate drums from an audio file
stems = demix_audio('path/to/audio.wav', output_dir='path/to/output')
```
"""

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="mdx23c-drum-separation",
    version="0.1.0",
    author="MDX23C Drum Separation",
    author_email="",
    description="A pip installable package for drum separation using the MDX23C model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/mdx23c-drum-separation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'mdx23c-separate=mdx23c.cli:main',
        ],
    },
    keywords="audio music source-separation drums ai machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/your-username/mdx23c-drum-separation/issues",
        "Source": "https://github.com/your-username/mdx23c-drum-separation",
    },
    include_package_data=True,
    zip_safe=False,
)
