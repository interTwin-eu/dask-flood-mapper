# Dask based Flood Mapping

![CI](https://github.com/TUW-GEO/dask-flood-mapper/actions/workflows/pytest.yml/badge.svg)

Map floods with Sentinel-1 radar images. We replicate in this package the work of Bauer-Marschallinger et al. (2022)<sup>1</sup> on the TU Wien Bayesian-based flood mapping algorithm. This implementation is entirely based on [`dask`](https://www.dask.org/) and data access via [STAC](https://stacspec.org/en) via [`odc-stac`](https://odc-stac.readthedocs.io/en/latest/). The algorithm requires three pre-processed input datasets stored and accessible via STAC at the Earth Observation Data Centre For Water Resources Monitoring (EODC). It is foreseen that future implementations can also use data from other STAC catalogues. This notebook explains how microwave backscattering can be used to map the extent of a flood. This workflow forms the backbone of this package.

## Installation

To install the package, do the following:

```
pip install git+https://github.com/TUW-GEO/dask-flood-mapper
```

## Usage

Storm Babet hit the Denmark and Northern coast of Germany at the 20th of October 2023 [Wikipedia](https://en.wikipedia.org/wiki/Storm_Babet). Here an area around Zingst at the Baltic coast of Northern Germany is selected as the study area.

### Local Processing

Define the time range and geographic region in which the event occurred.

```python 
time_range = "2022-10-11/2022-10-25"
bbox = [12.3, 54.3, 13.1, 54.6]
```
Use the flood module and calculate the flood extent with the Bayesian decision method applied tp Sentinel-1 radar images. The object returned is a [`xarray`](https://docs.xarray.dev/en/stable/) with lazy loaded Dask arrays. To get the data in memory use the `compute` method on the returned object.

```python
from dask_flood_mapper import flood

flood.decision(bbox=bbox, datetime=time_range).compute()
```

### Distributed Processing

It is also possible to remotely process the data at the EODC [Dask Gateway](https://gateway.dask.org/) with the added benefit that we can then process close to the data source without requiring rate-limiting file transfers over the internet.

For ease of usage of the Dask Gateway install the [`eodc`]() package besides the `dask-gateway` package. Also, see the [EODC documentation](https://github.com/eodcgmbh/eodc-examples/blob/main/demos/dask.ipynb). 

```
pip install dask-gateway eodc
```

Connect to the gateway (this requires an EODC account).

```python
from eodc.dask import EODCDaskGateway
from rich.prompt import Prompt
your_username = Prompt.ask(prompt="Enter your Username")
gateway = EODCDaskGateway(username=your_username)
```

Create a cluster.

```
cluster = gateway.new_cluster(cluster_options)
client = cluster.get_client()
cluster.adapt(minimum=2, maximum=5)
```

Map the flood the same way as we have done when processing locally.

```python
flood.decision(bbox=bbox, datetime=time_range).compute()
```

## Development

## Contributing

Interested in contributing to this project. Check the contributing guidelines for more information on how to contribute.

### Environment management

For convenience one can use `pipenv` and the `Pipfile.lock` to deterministically install all packages used during development. 

```bash
pipenv install
```

Checkout the [documentation](https://pipenv.pypa.io/en/latest/) for more help with installing pipenv and reconstructing the development environment.

### Testing

Running the requires `pytest` and is executed as follows:

```bash
# pipenv install pytest
pytest ./tests/
```

### Linting and formatting

The pre-commit hooks can be used to check whether you contribution follows the standards as adhered to in this project. Install and activate the `pre-commit` hooks, like so:

```bash
# pipenv install pre-commit
pre-commit install
```

Before each commiting, Ruff is actioned. To run Ruff without commiting, run:

```bash
pre-commit run --all-files
```

or

```bash
# pipenv install ruff
ruff check --fix --output-format concise
```

To fix the format, Ruff also offers this option with the command:

```bash
ruff format
```

To check whether the output of the notebook cells is removed one can do the following:

```bash
# pipenv install nbstripout
find . -name '*.ipynb' -exec nbstripout {} +
```

## Credits

Credits go to EODC ([https://eodc.eu](https://eodc.eu)) for developing the infrastructure and the management of the data required for this workflow. This work has been supported as part of the interTwin project ([https://www.intertwin.eu](https://www.intertwin.eu)). The interTwin project is funded by the European Union Horizon Europe Programme - Grant Agreement number 101058386.

Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union Horizon Europe/Horizon 2020 Programmes. Neither the European Union nor the granting authorities can be held responsible for them.

## Literature

1)  Bauer-Marschallinger, Bernhard, Senmao Cao, Mark Edwin Tupas, Florian Roth, Claudio Navacchi, Thomas Melzer, Vahid Freeman, and Wolfgang Wagner. â€œSatellite-Based Flood Mapping through Bayesian Inference from a Sentinel-1 SAR Datacube. Remote Sensing 14, no. 15 (January 2022): 3673. https://doi.org/10.3390/rs14153673.
