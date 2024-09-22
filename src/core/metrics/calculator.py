class MetricCalculator:
    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, inputs, outputs, prefix=""):
        results = {}
        for metric in self.metrics:
            results[f"{prefix}_{metric.__name__}"] = metric(inputs, outputs)
        return results
