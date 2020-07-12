<img src="static/logo.png" alt="alt text" width="33.333%" />


The simplest machine learning library for deploying prototypes, conducting quality assurance, and tracking production model performance. 

#### Preview
![demo](static/demo.gif)

## Usage

### Installation

Catacomb's Python library can be installed from the PyPi registry:

```
pip install catacomb-ai
```

To test installation, run `catacomb`:

```
Usage: catacomb [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  build
  push
  run
```

### Required Files

The only file Catacomb expects is a `system.py` file that implements a class containing the `__init__` and `output()` methods:

```python
import catacomb

class UppercaseModel:
    def __init__(self):
        """Initializing system and loading dependencies"""
        self.variable = True

    def output(self, text):
        """Performing inference and returning a prediction"""
        return text.upper()
        
if __name__ == "__main__":
    catacomb.connect(UppercaseModel, 'TEXT')
```

Implementing the system interface allows Catacomb to auto-generate a UI for the system/model from the command line tool. Model hosting will fail unless all dependencies are defined within the current directory (i.e. a `Pipfile` or `requirements.txt` file is required).

Running Catacomb locally can be done by running `python system.py`. 

### Deployment
Uploading to the Catacomb hosting platform can be done by running:

```
catacomb upload
```

and following the command-line prompts to configure meta-data and example test cases.

#### External Dependencies
Additional external dependencies can be installed by specifying a `catacomb.sh` bash file to run on the created image. This file is detected during the `catacomb build` process.

## License
MIT
