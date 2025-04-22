from dask.distributed import Client

client = Client(processes=False, threads_per_worker=2, n_workers=1, memory_limit="12GB")
print(client)
