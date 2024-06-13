# NOMAD's H5MD-NOMAD Parser Plugin
This is a plugin for [NOMAD](https://nomad-lab.eu) which contains a parser for the H5MD-NOMAD HDF5 file format.

## Getting started

### Install the dependencies

Clone the project and in the workspace folder, create a virtual environment (note this project uses Python 3.9):

```sh
git clone https://github.com/nomad-coe/nomad-parser-h5md.git
cd nomad-parser-h5md
python3.9 -m venv .pyenv
. .pyenv/bin/activate
```

There are 2 options for installation, while linking to the nomad-lab package:

1. Install with the current settings, which link to the develop branch of nomad-lab. In this case,
leave the `pyproject.toml` settings as is.

2. Install by linking to your local development version of nomad-lab. In this case, go into `pyproject.toml`
and replace:

```
"nomad-lab@git+https://github.com/nomad-coe/nomad.git@develop",
```

under the dependencies variable to:
```
"nomad-lab@file://<full_path_to_NOMAD>",
```

Now, install the plugin in development mode:

```sh
pip install --upgrade pip
pip install -e '.[dev]'
```



### Run the tests

You can run local tests using the `pytest` package:

```sh
python -m pytest -sv
```

where the `-s` and `-v` options toggle the output verbosity.

Our CI/CD pipeline produces a more comprehensive test report using `coverage` and `coveralls` packages.
To emulate this locally, perform:

```sh
pip install coverage coveralls
python -m coverage run -m pytest -sv
```

This setup should allow you to run and test the plugin as a "standalone" package, i.e., without explicitly adding it to the NOMAD package.
However, if there is some issue in NOMAD recognizing the package, you may also need to add the package folder to the `PYTHONPATH` of the Python environment of your local NOMAD installation:

```sh
export PYTHONPATH="$PYTHONPATH:<path-to-plugin-cloned-repo>/src"
```


### Run linting and auto-formatting

```sh
ruff check .
ruff format . --check
```

Ruff auto-formatting is also a part of the GitHub workflow actions. Make sure that before you make a Pull Request, `ruff format . --check` runs in your local without any errors otherwise the workflow action will fail.

### Debugging

For interactive debugging of the tests, use `pytest` with the `--pdb` flag.
We recommend using an IDE for debugging, e.g., _VSCode_.
If using VSCode, you can add the following snippet to your `.vscode/launch.json`:

```json
{
  "configurations": [
      {
        "name": "<descriptive tag>",
        "type": "debugpy",
        "request": "launch",
        "cwd": "${workspaceFolder}",
        "program": "${workspaceFolder}/.pyenv/bin/pytest",
        "justMyCode": true,
        "env": {
            "_PYTEST_RAISE": "1"
        },
        "args": [
            "-sv",
            "--pdb",
            "<path to plugin tests>",
        ]
    }
  ]
}
```

where `${workspaceFolder}` refers to the NOMAD root.

The settings configuration file `.vscode/settings.json` performs automatically applies the linting upon saving the file progress.
