# Microbiome Toolkit 

[![Build Status](https://github.com/bdsl-lab/mbiomekit/actions/workflows/ci.yml/badge.svg)](https://github.com/bdsl-lab/mbiomekit/actions)
[![Coverage Status](https://coveralls.io/repos/github/bdsl-lab/mbiomekit/badge.svg?branch=main)](https://coveralls.io/github/bdsl-lab/mbiomekit?branch=main)
[![PyPI version](https://badge.fury.io/py/mbiomekit.svg)](https://badge.fury.io/py/mbiomekit)

A collection of Python toolkits for microbiome analysis. 

## Installation 

### From PyPI

Not implemented yet

```bash
pip install mbiomekit
```

### From source 

git clone https://github.com/bdsl-lab/mbiomekit.git
cd mbiomekit
pip install -e .

## Usage 

### Command-line 

```bash 
mbiomekit --help 
```

### Python API

```python
from mbiomekit import main 

if __name__ == "__main__":
    main()
```

## Testing 

```bash
pytest --maxfail=1 --disable-warnings -v
```

## License

This project is licensed under the MIT License - see the [[LICENSE]] file for details. 

