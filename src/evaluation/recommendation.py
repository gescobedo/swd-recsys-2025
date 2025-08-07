from rmet import calculate


def calculate_recommendation_metrics(
    metrics,
    model_logits,
    model_targets,
    metrics_topk,
    flatten_results,
    return_individual=False,
):
    """Calculates recomendation metrics"""

    result_metrics = calculate(
        metrics=metrics,
        logits=model_logits,
        targets=model_targets,
        k=metrics_topk,
        flatten_results=flatten_results,
        return_individual=return_individual,
    )
    return result_metrics
