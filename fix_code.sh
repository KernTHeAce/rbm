#!/bin/bash

isort() {
  echo "--------------------------------------ISORT--------------------------------------"
  poetry run isort src --line-length 120
  echo "---------------------------------------------------------------------------------"

}

black() {
  echo "--------------------------------------BLACK--------------------------------------"
  poetry run black src --line-length 120
  echo "---------------------------------------------------------------------------------"
}


help() {
  echo "usage: ./scripts/local_ci.sh [-h] [-t] [-c] [-i] [-b] [-p] [-a]"
  echo
  echo "Check changed code before pushing"
  echo
  echo "arguments:"
  echo "  -i     run isort"
  echo "  -b     run black"
  echo
}

if [ $# -lt 1 ]; then
  isort
  black
fi

while getopts "ib" option; do
  case $option in
  i)
    isort
    ;;
  b)
    black
    ;;
  h)
    help
    exit
    ;;
  \?) # incorrect option
    echo "Error: Invalid option"
    exit
    ;;
  esac
done
