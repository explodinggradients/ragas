from rich.console import Console
from rich.table import Table

result = (
    {
        "NDCG@1": 0.28395,
        "NDCG@3": 0.26503,
        "NDCG@5": 0.27639,
        "NDCG@10": 0.30024,
        "NDCG@100": 0.36041,
        "NDCG@1000": 0.39641,
    },
    {
        "MAP@1": 0.14865,
        "MAP@3": 0.20239,
        "MAP@5": 0.21875,
        "MAP@10": 0.23319,
        "MAP@100": 0.24772,
        "MAP@1000": 0.24961,
    },
    {
        "Recall@1": 0.14865,
        "Recall@3": 0.24294,
        "Recall@5": 0.29429,
        "Recall@10": 0.36756,
        "Recall@100": 0.59293,
        "Recall@1000": 0.80835,
    },
    {
        "P@1": 0.28395,
        "P@3": 0.17181,
        "P@5": 0.12716,
        "P@10": 0.08117,
        "P@100": 0.01434,
        "P@1000": 0.00208,
    },
)


# use type var
def gen_table(exp_name: str, results: tuple[dict, dict, dict, dict]):
    table = Table(title=f"{exp_name} result")

    table.add_column("Metric Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right", style="green")
    for metric in results:
        [table.add_row(s_at_k, str(metric[s_at_k])) for s_at_k in metric]

    return table


console = Console()
table = gen_table("test", result)
console.print(table)
