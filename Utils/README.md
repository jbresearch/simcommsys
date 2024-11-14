# Simcommsys Utilities

This is a Python package that aids in building, running and plotting results from Simcommsys systems, timers and simulations.

We recommend that you use this package if you wish to run simulations using Simcommsys rather than directly interacting with the binaries.

## Prerequisites

This package requires Python (3.10 or later) and the [Poetry](https://python-poetry.org) build system in order to run.

The latter can be installed on Ubuntu (22.04 or later) using the following command:

``` bash
curl -sSL https://install.python-poetry.org | python3 -
```

## Installation

Before running this tool for the first time, or after pulling from the Git remote, you should install/refresh the tool:

``` bash
poetry install
```

This will create a virtual environment at `./.venv` (if one does not already exist), and install any project dependencies inside it.

## Usage

The tool can be run using the following (template) command:

``` bash
poetry run simcommsys-utils [subcommand ...]
```

In order to see the list of subcommands offered by the tool, run:

``` bash
poetry run simcommsys-utils --help
```

For full usage instructions of a `<subcommand>` offered by the tool, run:

``` bash
poetry run simcommsys-utils <subcommand> --help
```

## Examples

The `Examples/` directory within Simcommsys contains some examples of how this tool could be used.