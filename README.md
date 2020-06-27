# Catacomb
Catacomb is the simplest machine learning platform for deploying prototypes, conducting quality assurance, and tracking production model performance. 

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

The only file Catacomb expects is a `system.py` file that implements the `System` class, including overriding the `__init__` and `output` method:

```python
import catacomb

class System(catacomb.System):
    def __init__(self):
        # Finish other system setup
        self.variable = 42

    def output(self, input_object):
        # Performing inference and returning a prediction
        return input_object * self.variable
```

Implementing the `System` interface allows Catacomb to auto-generate a UI for the system/model,
in addition to performing predictions over HTTP. Model hosting will fail unless all dependencies
are defined within the current directory (i.e. a `Pipfile` or `requirements.txt` file is required).

### Deployment

Containerization is at the core of Catacomb's hosting platform. 
Catacomb's build process requires that Docker is both already installed on the local machine, and 
the client is logged into a Docker account (for pushing images to the Docker registry). 

Deployment to Catacomb hosting can be done with the following commands:

## License
MIT