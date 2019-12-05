"""
This script has a few functionaities:

1. download-images:  This call will download all images necessary for training the gan
2. download-cards: This call will download the json representation of all of the cards data and process it into a single
    text file.
"""

import os
import sys
import gpt_2_simple as gpt2

if "linux" in sys.platform:
    os.environ["LC_AL"] = "C.UTF-8"
    os.environ["LANG"] = "C.UTF-8"
else:
    os.environ["LC_AL"] = "en_US.utf-8"
    os.environ["LANG"] = "en_US.utf-8"

import click

@click.group()
def main():
    pass


@main.command("style-transfer")
@click.argument("content-image-path")
@click.argument("style-image-path")
@click.option("--num-steps", default=300)
def download_images():
    pass


def download_cards():
    pass


@main.command("finetune")
@click.option("--model-name", type=str, default="124M", help="Can be 117M, 124M, or 355M")
@click.option("--text-path", type=str, default="./data/mtg_combined.txt")
@click.option("--num-steps", type=int, default=10000)
def finetune(model_name: str, text_path: str, num_steps) -> None:

    # Download the model if it is not present
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)


    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                  text_path,
                  model_name=model_name,
                  steps=num_steps)   # steps is max number of training steps

    gpt2.generate(sess)


if __name__ == "__main__":
    main()
