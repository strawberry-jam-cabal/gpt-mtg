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
@click.option("--text-path", type=str, default="/data/mtg_combined.txt")
@click.option("--num-steps", type=int, default=10000)
def finetune(text_path: str, num_steps) -> None:
    model_name = "124M"
    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                  text_path,
                  model_name=model_name,
                  steps=num_steps)   # steps is max number of training steps

    gpt2.generate(sess)


if __name__ == "__main__":
    main()