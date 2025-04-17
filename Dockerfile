FROM ghcr.io/eodcgmbh/cluster_image:2025.4.1
USER ubuntu
RUN pip install jupyter
COPY --chown=1000:1000 . /app/dask_flood_mapper
WORKDIR /app/dask_flood_mapper
RUN pip install .[all]
