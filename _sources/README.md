# Dask based Flood Mapping

![CI](https://github.com/TUW-GEO/dask-flood-mapper/actions/workflows/pytest.yml/badge.svg)

Map floods with Sentinel-1 radar images. We replicate in this package the work of Bauer-Marschallinger et al. (2022) on the TU Wien Bayesian-based flood mapping algorithm. This implementation is entirely based on `Dask` and data access via [STAC](https://stacspec.org/en). The algorithm requires three pre-processed input datasets stored and accessible via STAC at the Earth Observation Data Centre For Water Resources Monitoring (EODC). It is foreseen that future implementations can also use data from other STAC catalogues. This notebook explains how microwave backscattering can be used to map the extent of a flood. This workflow forms the backbone of this package.


## Installation

To install the package, do the following:

```
pip install git+https://github.com/TUW-GEO/dask-flood-mapper
```

## Usage

### Local Processing

### Distributed Processing

It is also possible to remote process the data at EODC with the added benefit that we can then process close to the data source without requiring rate-limiting file transfers over the internet.

## Development

## Contributing

Interested in contributing to this project.

### Environment management

For convenience one can use `pipenv` and the `Pipfile.lock` to deterministically install all packages used during development. 

```bash
pipenv install
```

Checkout the [documentation](https://pipenv.pypa.io/en/latest/) for more help with installing pipenv and reconstructing the development environment.

### Linter

Before each commiting, Ruff is actioned. To run Ruff without commiting, run:

```bash
pre-commit run --all-files
```

or

```bash
ruff check --fix --output-format concise
```

To fix the format, Ruff also offers this option with the command:

```bash
ruff format
```

## Credits

