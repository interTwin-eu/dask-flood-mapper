# Dask based Flood Mapping

![CI](https://github.com/interTwin-eu/dask-flood-mapper/actions/workflows/pytest.yml/badge.svg)
[![DOI](https://zenodo.org/badge/859296745.svg)](https://doi.org/10.5281/zenodo.15004960)
![pypi](https://img.shields.io/pypi/v/dask_flood_mapper.svg)
[![GitHub Super-Linter](https://github.com/interTwin-eu/dask-flood-mapper/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/dask-flood-mapper/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)
[![SQAaaS source code](https://github.com/EOSC-synergy/dask-flood-mapper.assess.sqaaas/raw/main/.badge/status_shields.svg)](https://sqaaas.eosc-synergy.eu/#/full-assessment/report/https://raw.githubusercontent.com/eosc-synergy/dask-flood-mapper.assess.sqaaas/main/.report/assessment_output.json)

Map floods with Sentinel-1 radar images. We replicate in this package the work
of Bauer-Marschallinger et al. (2022)[^1] on the TU Wien Bayesian-based
flood mapping algorithm. This implementation is entirely based on
[`dask`](https://www.dask.org/) and data access via
[STAC](https://stacspec.org/en) with
[`odc-stac`](https://odc-stac.readthedocs.io/en/latest/). The algorithm requires
three pre-processed input datasets stored and accessible via STAC at the Earth
Observation Data Centre For Water Resources Monitoring (EODC). It is foreseen
that future implementations can also use data from other STAC catalogues. This
notebook explains how microwave backscattering can be used to map the extent of
a flood. The workflow detailed in this
[notebook](https://intertwin-eu.github.io/dask-flood-mapper/notebooks/03_flood_map.html)
forms the backbone of this package. For a short overview of the Bayesian decision
method for flood mapping see this
[ProjectPythia book](https://projectpythia.org/eo-datascience-cookbook/notebooks/tutorials/floodmapping.html).

## Installation

To install the package, do the following:

```bash
pip install dask-flood-mapper
```

## Usage

Storm Babet hit the Denmark and Northern coast of Germany at the 20th of October
2023 [Wikipedia](https://en.wikipedia.org/wiki/Storm_Babet). Here an area around
Zingst at the Baltic coast of Northern Germany is selected as the study area.

### Local Processing

Define the time range and geographic region in which the event occurred.

```python
time_range = "2022-10-11/2022-10-25"
bbox = [12.3, 54.3, 13.1, 54.6]
```

Use the flood module and calculate the flood extent with the Bayesian decision
method applied tp Sentinel-1 radar images. The object returned is a
[`xarray`](https://docs.xarray.dev/en/stable/) with lazy loaded Dask arrays. To
get the data in memory use the `compute` method on the returned object.

```python
from dask_flood_mapper import flood


flood.decision(bbox=bbox, datetime=time_range).compute()
```

### Distributed Processing

It is also possible to remotely process the data at the EODC
[Dask Gateway](https://gateway.dask.org/) with the added benefit that we can
then process close to the data source without requiring rate-limiting file
transfers over the internet.

For ease of usage of the Dask Gateway install the
[`eodc`](https://pypi.org/project/eodc/) package besides the `dask-gateway`
package. Also, see the
[EODC documentation](https://github.com/eodcgmbh/eodc-examples/blob/main/demos/dask.ipynb).

```bash
pip install dask-gateway eodc
# or use pipenv
# git clone https://github.com/interTwin-eu/dask-flood-mapper.git
# cd dask-flood-mapper
# pipenv sync -d
```

However differences in versions client- and server-side can cause problems.
Hence, the most convenient way to successively use the EODC Dask Gateway is
Docker. To do this clone the GitHub repository and use the docker-compose.yml.

```bash
git clone https://github.com/interTwin-eu/dask-flood-mapper.git
cd dask-flood-mapper
docker compose up
```

Copy and paste the generated URL to launch Jupyter Lab in your browser. Here one
can run the below code snippets or execute the
[notebook](https://intertwin-eu.github.io/dask-flood-mapper/notebooks/02_remote_dask.html)
about remote processing.

```python
from eodc.dask import EODCDaskGateway
from eodc import settings
from rich.prompt import Prompt


settings.DASK_URL = "http://dask.services.eodc.eu"
settings.DASK_URL_TCP = "tcp://dask.services.eodc.eu:10000/"
```

Connect to the gateway (this requires an EODC account).

```python
your_username = Prompt.ask(prompt="Enter your Username")
gateway = EODCDaskGateway(username=your_username)
```

Create a cluster.

> [!CAUTION]
> Per default no worker is spawned, therefore please use the widget to add/scale
> Dask workers in order to enable computations on the cluster.

```python
cluster_options = gateway.cluster_options()
cluster_options.image = "ghcr.io/eodcgmbh/cluster_image:2025.4.1"
cluster = gateway.new_cluster(cluster_options)
client = cluster.get_client()
cluster
```

Map the flood the same way as we have done when processing locally.

```python
flood.decision(bbox=bbox, datetime=time_range).compute()
```

### User Interface

It is also possible to run the workflow in an user-friendly interface, as shown
below:

![screenshot](docs/images/screenshot_floodmap_gui.png)

Firstly, install the extra packages with:

```bash
pip install dask-flood-mapper[app]
```

Then, to access it, simplify run the in terminal the command:

```bash
floodmap
```

It will open the GUI in the web browser.

## Authors

[Martin Schobben](https://github.com/martinschobben),
[Thais Beham](https://github.com/thaisbeham),
[Clay Harrison](https://github.com/claytharrison)

### Contributors

![https://github.com/interTwin-eu/dask-flood-mapper/graphs/contributors](https://contrib.rocks/image?repo=interTwin-eu/dask-flood-mapper)

## Contributing Guidelines

Please find the contributing guidelines in the specific file
[CONTRIBUTING.md](CONTRIBUTING.md).

## Automated Delivery

This repository holds a container image to be used for running Dask based flood
mapping on the EODC Dask Gateway. Use the URL
`ghcr.io/intertwin-eu/dask-flood-mapper:latest` to specify the image.

```bash
docker pull ghcr.io/intertwin-eu/dask-flood-mapper:latest
```

## Credits

Credits go to EODC ([https://eodc.eu](https://eodc.eu)) for developing the
infrastructure and the management of the data required for this workflow. This
work has been supported as part of the interTwin project
([https://www.intertwin.eu](https://www.intertwin.eu)). The interTwin project is
funded by the European Union Horizon Europe Programme - Grant Agreement number 101058386.

Views and opinions expressed are however those of the authors only and do not
necessarily reflect those of the European Union Horizon Europe/Horizon 2020
Programmes. Neither the European Union nor the granting authorities can be held
responsible for them.

## License

This repository is covered under the [MIT License](LICENSE.txt).

## Literature

[^1]:
    Bauer-Marschallinger, Bernhard, Senmao Cao, Mark Edwin Tupas, Florian
    Roth, Claudio Navacchi, Thomas Melzer, Vahid Freeman, and Wolfgang Wagner.
    Satellite-Based Flood Mapping through Bayesian Inference from a Sentinel-1
    SAR Datacube. Remote Sensing 14, no. 15 (January 2022): 3673.
    [https://doi.org/10.3390/rs14153673](https://doi.org/10.3390/rs14153673).
