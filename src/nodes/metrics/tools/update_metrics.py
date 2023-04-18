from typing import Any, Dict, Optional


def update_metrics(metrics: Optional[Dict[str, Any]], new_data: Dict[str, Any]):
    if metrics is None:
        metrics = new_data
    else:
        metrics.update(new_data)
    return metrics
