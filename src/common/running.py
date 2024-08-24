def pipeline(model, trainer, epochs, metric_calculator, logger, model_initializer=None):
    if model_initializer:
        model = model_initializer(model)

    trainer.init_optimizer(model)
    for epoch in range(epochs):
        print(epoch)
        model, inputs, outputs = trainer.epoch(model)
        metrics = metric_calculator(inputs, outputs, "train")
        print(metrics)
        inputs, outputs = trainer.test(model)
        metrics = metric_calculator(inputs, outputs, "test")
        print(metrics)
        logger.log_metrics(metrics, epoch)

    inputs, outputs = trainer.test(model)
    metrics = metric_calculator(inputs, outputs)
    print(metrics)
    # logger.log_metrics(metrics)
