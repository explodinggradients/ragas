import asyncio
import time

import pandas as pd
from llama_index import ServiceContext, StorageContext, load_index_from_storage


def load_query_engine():
    # CHANGE SERVICE_CONTEXT HERE!!!
    openai_sc = ServiceContext.from_defaults()
    service_context = openai_sc

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")

    # load index
    index = load_index_from_storage(storage_context=storage_context)

    return index.as_query_engine()


async def main(qs):
    await asyncio.gather(*[qe.aquery(i) for i in qs])


def sync_main(qs):
    return [qe.query(i) for i in qs]


if __name__ == "__main__":
    qe = load_query_engine()
    df = pd.read_csv("./ragas-wikiqa.csv")
    qs = df["question"][:8]
    start = time.perf_counter()
    # responses = asyncio.run(main(qs))
    responses = sync_main(qs)
    end = time.perf_counter() - start
    print(f"Finished in {end:0.4f} seconds")
