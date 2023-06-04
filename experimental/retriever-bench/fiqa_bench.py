import pickle

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.reranking.models import CrossEncoder
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from rich.console import Console
from rich.table import Table

console = Console()


def gen_table(exp_name: str, results: list[dict, dict, dict, dict]):
    table = Table(title=f"{exp_name} result")

    table.add_column("Metric Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="green")
    for metric in results:
        [table.add_row(s_at_k, str(metric[s_at_k])) for s_at_k in metric]

    return table


DATASET = "fiqa"
url = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        DATASET
    )
)
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


#### BI-Encoder #####
model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)
retriever = EvaluateRetrieval(
    model, score_function="dot"
)  # or "cos_sim" for cosine similarity

LOAD_BI_ENCODER_RESULTS = False
if not LOAD_BI_ENCODER_RESULTS:
    #### Load the SBERT model and retrieve using cosine-similarity
    results = retriever.retrieve(corpus, queries)

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
else:
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)

#### Rerank using Cross-Encoder ####

# Load CrossEncoder Models
reranker = {}
for model_name in [
    "cross-encoder/ms-marco-TinyBERT-L-6",
    # "cross-encoder/ms-marco-MiniLM-L-6-v2",
    # "cross-encoder/stsb-distilroberta-base",
    # "cross-encoder/ms-marco-electra-base",
]:
    reranker[model_name] = Rerank(CrossEncoder(model_name), batch_size=128)

# Rerank top-100 results using the reranker provided
rerank_results = {}
for model_name in reranker:
    rerank_results[model_name] = reranker[model_name].rerank(
        corpus, queries, results, top_k=100
    )

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
table = gen_table("before reranking", [ndcg, _map, recall, precision])
console.print(table)

for model_name in reranker:
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, rerank_results[model_name], retriever.k_values
    )
    table = gen_table(
        f"after reranking with {model_name}", [ndcg, _map, recall, precision]
    )
    console.print(table)
