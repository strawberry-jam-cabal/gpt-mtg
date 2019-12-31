# gpt-mtg
This repository holds the code for generating mtg cards with gpt2.

# Getting Started

## Build Containers
To build containers capable of running the code on either the cpu or gpu use the command `make build-cpu` or
`make build-gpu`.

## Finetuning Model
To finetune a model on your dataset:
1. Create a dataset like output.txt in the wonderful <a href="https://github.com/billzorn/mtgencode">mtgencode<a> repository.  If you want to build your own dataset follow the instructions in the mtgencode repostiory.  For this project I just used output.txt
2. Create the appropriate docker container by running either `make build-cpu` or `make build-gpu`
3. Run `python3 run.py finetune` with the appropriate flags, with no adjustments this will run on the magic data in the data folder.

## Predicting Text
To predict text:
1. Download the models into the gpt-mtg base directory from <a href=https://drive.google.com/drive/u/1/folders/1_02y82VOEvR5OZ3e-s81swQwlJxN9rT8>here</a>
2. Unzip the files, the gpt model should unzip into a folder called `checkpoint`
3. open up a docker container with `make run-cpu`
4. run the text generation script `python3 run.py generate-text`

## Creating Cards
Follow the instructions in <a href="https://github.com/minimaxir/mtg-card-creator-api">this</a> repository to get setup.
Once you have all of the necessary tools to generate magic cards then:
1. Start docker with `make run-cpu`
2. Run the model to generate an encoding which should be parsable by mtgencode by running `python3 run.py generate-text --destination-path <filepath.txt>`
3. Use mtgencode to create a human readible version of the card `./decode.py -v <cardfile.txt> cards.pretty.txt -d`
4. Use mtgencode to create a magic set zeditor set file`decode.py -e rfields -mse encoded.txt MSE/card`
5. Use magic set editor to create sets from the cards

# Contributing
To contribute to this repository please:
1. `cp dev/hooks/pre-commit .git/hooks` and `chmod +x .git/hooks/pre-commit`



## TODO:
- figure out how to get allsets.json and parse it using mtgencode
- figure out how to use decode to mse to get mse parseable cards
- Figure out how to use mse to generate new cards.

