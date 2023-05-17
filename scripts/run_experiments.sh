#!/bin/bash

#poetry run python -m src.pipelines.soccer_ae --max_epoch 200 --prefix soccer_ae
#poetry run python -m src.pipelines.mnist_ae --max_epoch 200 --prefix mnist_ae
poetry run python -m src.pipelines.mnist_classifier --max_epoch 2 --prefix mnist_classifier
