#!/bin/bash

poetry run python -m src.pipelines.soccer_ae
poetry run python -m src.pipelines.mnist_ae
poetry run python -m src.pipelines.mnist_classifier
