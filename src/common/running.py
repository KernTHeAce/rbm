import time


def pipeline(model, trainer, epochs, metric_calculator, logger, model_initializer=None):
    if model_initializer:
        start_initializing = time.time()
        model = model_initializer(model)
        logger.log_params({"initializing time": time.time() - start_initializing})

    trainer.init_optimizer(model)
    for epoch in range(epochs):
        print(epoch)
        model, inputs, outputs = trainer.epoch(model)
        metrics = metric_calculator(inputs, outputs, "train")
        logger.log_metrics(metrics, epoch)

        inputs, outputs = trainer.test(model)
        metrics = metric_calculator(inputs, outputs, "test")
        logger.log_metrics(metrics, epoch)

    inputs, outputs = trainer.test(model)
    metrics = metric_calculator(inputs, outputs, "final_test_")
    logger.log_params(metrics)
