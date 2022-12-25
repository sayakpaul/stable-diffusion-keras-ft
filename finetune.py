"""
Adapted from  https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

# Usage
python finetune.py
"""

import warnings

warnings.filterwarnings("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse

import tensorflow as tf
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler

from datasets import DatasetUtils
from trainer import Trainer

MAX_PROMPT_LENGTH = 77


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to fine-tune a Stable Diffusion model."
    )
    parser.add_argument("--dataset_archive", default=None, type=str)
    parser.add_argument("--img_height", default=256, type=int)
    parser.add_argument("--img_width", default=256, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--wd", default=1e-2, type=float)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.999, type=float)
    parser.add_argument("--epsilon", default=1e-08, type=float)
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    return parser.parse_args()


def run(args):
    print("Initializing dataset...")
    data_utils = DatasetUtils(
        dataset_archive=args.dataset_archive,
        batch_size=args.batch_size,
        img_height=args.img_height,
        img_width=args.img_width,
    )
    training_dataset = data_utils.prepare_dataset()

    print("Initializing trainer...")
    diffusion_ft_trainer = Trainer(
        DiffusionModel(args.img_height, args.img_width, MAX_PROMPT_LENGTH),
        ImageEncoder(args.img_height, args.img_width),
        NoiseScheduler(),
    )

    print("Initializing optimizer...")
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=args.lr,
        weight_decay=args.wd,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        epsilon=args.epsilon,
    )

    print("Compiling trainer...")
    diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")

    print("Training...")
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        "ema_diffusion_model.h5",
        save_weights_only=True,
        monitor="loss",
        mode="min",
    )
    diffusion_ft_trainer.fit(
        training_dataset, epochs=args.num_epochs, callbacks=[ckpt_callback]
    )


if __name__ == "__main__":
    args = parse_args()
    run(args)
