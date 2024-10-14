import time


def model_training_pipeline(model, trainer, epochs, metric_calculator, logger, model_initializer=None):
    start_run = time.time()
    if model_initializer:
        start_initializing = time.time()
        model = model_initializer(model)
        logger.log_params({"initializing time": time.time() - start_initializing})
        if model is None:
            logger.log_params({"errors": "exploding gradients"})
            return

    trainer.init_optimizer(model)
    for epoch in range(epochs):
        # print(epoch)
        model, targets, outputs, avg_loss = trainer.epoch(model)
        metrics = metric_calculator(targets, outputs, "train")
        metrics["train_avg_loss"] = avg_loss
        logger.log_metrics(metrics, epoch)

        targets, outputs, avg_loss = trainer.test(model)
        metrics = metric_calculator(targets, outputs, "test")
        metrics["test_avg_loss"] = avg_loss
        logger.log_metrics(metrics, epoch)

    inputs, outputs, avg_loss = trainer.test(model)
    metrics = metric_calculator(inputs, outputs, "final_test_")
    logger.log_params(metrics)
    logger.log_params({"time": time.time() - start_run})
    return model, trainer.optimizer
