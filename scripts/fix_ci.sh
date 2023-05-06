#!/bin/bash

poetry run isort ./src --line-length 120
poetry run black ./src --line-length 120
