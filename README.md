# gpt-mtg
This repository holds the code for generating mtg cards with gpt2.

# Getting Started

## Build Containers
To build containers capable of running the code on either the cpu or gpu use the command `make build-cpu` or
`make build-gpu`.

## Finetuning Model
To finetune a model on your dataset:
1. Create a dataset where the start of each text section begins with  `<|startoftext|>` and the end of the section ends with `<|endoftext|>
2. Create the appropriate docker container by running either `make build-cpu` or `make build-gpu`
3. Run `python3 run.py finetune` with the appropriate flags, with no adjustments this will run on the magic data in the data folder.

## Predicting Text
To predict text run:

# Contributing
To contribute to this repository please:
1. `cp dev/hooks/pre-commit .git/hooks` and `chmod +x .git/hooks/pre-commit`
