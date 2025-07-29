FROM quay.io/jupyter/minimal-notebook:latest
FROM ghcr.io/eodcgmbh/cluster_image:latest

USER ubuntu
COPY --chown=1000:1000 . /app/dask_flood_mapper
WORKDIR /app/dask_flood_mapper

# hadolint ignore=DL3013,SC2102
RUN pip install .[all] --no-cache-dir
