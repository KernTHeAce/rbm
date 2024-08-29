import time


def pipeline(model, trainer, epochs, metric_calculator, logger, model_initializer=None):
    start_run = time.time()
    start_initializing = time.time()
    if model_initializer:
        model = model_initializer(model)
        if model is None:
            logger.log_params({"errors": "exploding gradients"})
            return
    logger.log_params({"initializing time": time.time() - start_initializing})

    trainer.init_optimizer(model)
    for epoch in range(epochs):
        model, inputs, outputs = trainer.epoch(model)
        metrics = metric_calculator(inputs, outputs, "train")
        logger.log_metrics(metrics, epoch)

        inputs, outputs = trainer.test(model)
        metrics = metric_calculator(inputs, outputs, "test")
        logger.log_metrics(metrics, epoch)

    inputs, outputs = trainer.test(model)
    metrics = metric_calculator(inputs, outputs, "final_test_")
    logger.log_params(metrics)
    logger.log_params({"time": time.time() - start_run})
