from ragas.metrics import numeric_metric


@numeric_metric(allowed_values=(0.0, 1.0))
async def rouge_score(
    reference: str,
    response: str,
    rouge_type: str = "rougeL",  # "rouge1" or "rougeL"
    mode: str = "fmeasure",  # "fmeasure", "precision", or "recall"
) -> float:
    """Calculate ROUGE score between reference and response.

    Parameters
    ----------
    reference : str
        The reference text to compare against
    response : str
        The response text to evaluate
    rouge_type : str, optional
        Type of ROUGE metric to use ("rouge1" or "rougeL"), by default "rougeL"
    mode : str, optional
        Score mode to return ("fmeasure", "precision", or "recall"), by default "fmeasure"

    Returns
    -------
    float
        ROUGE score in range [0.0, 1.0]
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError(
            f"{e.name} is required for rouge score. Please install it using `pip install {e.name}`"
        )

    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(reference, response)
    return getattr(scores[rouge_type], mode)
