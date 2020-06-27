# Catacomb
Catacomb is a platform that hosts artificial intelligence systems so that anybody can interact with them in an intuitive way. We handle free deployment of machine learning models, from automatically generating a front-end application to handling analytics on model usage.

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
  delete
```

### Required Files

The only file Catacomb expects is a `system.py` file that implements the `System` class, including overriding
the `__init__` and `output` method:

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

Deployment to Catacomb hosting can be done in a single command: 

```
catacomb
```

The CLI will prompt for the required project metadata such as system name, 
and type annotations of input and output objects, before deploying:

```
$ catacomb

🤖 Image name: neural-network
🤖 Docker account username: hinton

🤖 Got it, deploying!

Your app is live at: https://catacomb.ai/hinton/neural-network/.

Enjoy!
```

## License
MIT
