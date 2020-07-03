# Catacomb
Catacomb is the simplest machine learning library for deploying prototypes, conducting quality assurance, and tracking production model performance. 

#### Preview
[![asciicast](https://asciinema.org/a/4q2OKzxrRKe2ql32BZwnqyKrE.svg)](https://asciinema.org/a/4q2OKzxrRKe2ql32BZwnqyKrE)

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
        """Initializing system and loading dependencies"""
        self.variable = 42

    def output(self, input_object):
        """Performing inference and returning a prediction"""
        return input_object * self.variable
        
if __name__ == "__main__":
    catacomb.start(System())
```

Implementing the `System` interface allows Catacomb to auto-generate a UI for the system/model,
in addition to performing predictions over HTTP. Model hosting will fail unless all dependencies
are defined within the current directory (i.e. a `Pipfile` or `requirements.txt` file is required).

Optionally, you can start a local REST API by running `catacomb.start(System())` when the script is called using `python system.py` (lines 12 to 13 in above example).

### Deployment
Catacomb's build process requires that Docker is both already installed on the local machine, and 
the client is logged into a Docker account (for pushing images to the Docker registry). 

Deployment to Catacomb hosting can be done with the following commands:

1. `catacomb build`
2. `catacomb push`

Include dependencies for `system.py` in a `Pipfile` and corresponding `Pipfile.lock` if using [Pipenv](https://pypi.org/project/pipenv/), or in a `requirements.txt` file. 

Run `catacomb build` to build a Docker image from the current directory. You will be prompted for an image name and you Dockerhub username.

```
➜ catacomb build
🤖 Image name: example
🤖 Docker hub username: mukundv7
🤖 Building your Docker image (this may take a while so you might wanna grab some coffee ☕)...
```

This will install any packages specified in the `Pipfile` or `requirements.txt` onto the Docker image and copy all contents of the current directory to the image.

One an image has been created with `catacomb build`, the `catacomb push` command can be run to push this image to Docker and produce an upload url. If you are logged into your account on [Catacomb](https://beta.catacomb.ai), you can simply go to this URL to finalize deployment.

#### External Dependencies
Additional external dependencies can be installed by specifying a `catacomb.sh` bash file to run on the created image. This file is detected during the `catacomb build` process.

## License
MIT
