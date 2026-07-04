def split_bleu_sentences(text: str) -> list[str]:
    """Split text into sentences for sacrebleu corpus_bleu."""
    if not text:
        return [text]
    if "\n" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines
    if ". " in text:
        return text.split(". ")
    return [text]


def fbeta_score(tp, fp, fn, beta=1.0):
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision == 0 and recall == 0:
        return 0.0

    beta_squared = beta**2
    fbeta = (
        (1 + beta_squared)
        * (precision * recall)
        / ((beta_squared * precision) + recall)
    )

    return fbeta
