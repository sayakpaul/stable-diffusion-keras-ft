"""
Adapted from  https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
and from https://github.com/huggingface/diffusers/pull/1884/files
and from https://github.com/keras-team/keras-io/pull/1388/files
# Usage
python finetune.py
"""
import warnings

from keras_cv.models.stable_diffusion.stable_diffusion import MAX_PROMPT_LENGTH

warnings.filterwarnings("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse

import tensorflow as tf
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from tensorflow.keras import mixed_precision

from datasets import DatasetUtils
from trainer import Trainer

CKPT_PREFIX = "ckpt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to fine-tune a Stable Diffusion model."
    )
    # Dataset related.
    parser.add_argument("--dataset_archive", default=None, type=str)
    parser.add_argument("--img_height", default=512, type=int)
    parser.add_argument("--img_width", default=512, type=int)
    parser.add_argument("--augmentation", action="store_true", help="Whether to do data augmentation.")

    # Optimization hyperparameters.
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--decay_steps", default=800, type=int)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--wd", default=1e-2, type=float)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.999, type=float)
    parser.add_argument("--epsilon", default=1e-08, type=float)
    parser.add_argument("--ema", default=0.9999, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    # Training hyperparameters.
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_epochs", default=70, type=int)
    parser.add_argument("--lora", action="store_true", help="Whether to load loRA layer.")
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--lora_alpha", default=4, type=float)

    # Others.
    parser.add_argument(
        "--mp", action="store_true", help="Whether to use mixed-precision."
    )
    parser.add_argument(
        "--pretrained_ckpt",
        default=None,
        type=str,
        help="Provide a local path to a diffusion model checkpoint in the `h5`"
             " format if you want to start over fine-tuning from this checkpoint.",
    )

    return parser.parse_args()


def run(args):
    if args.mp:
        print("Enabling mixed-precision...")
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        assert policy.compute_dtype == "float16"
        assert policy.variable_dtype == "float32"

    print("Initializing dataset...")
    data_utils = DatasetUtils(
        dataset_archive=args.dataset_archive,
        batch_size=args.batch_size,
        img_height=args.img_height,
        img_width=args.img_width,
    )
    training_dataset = data_utils.prepare_dataset(augmentation=args.augmentation)

    print("Initializing trainer...")
    ckpt_path = (
            CKPT_PREFIX
            + f"_epochs_{args.num_epochs}"
            + f"_res_{args.img_height}"
            + f"_mp_{args.mp}"
            + ".h5"
    )
    image_encoder = ImageEncoder(args.img_height, args.img_width)

    diffusion_ft_trainer = Trainer(
        diffusion_model=DiffusionModel(
            args.img_height, args.img_width, MAX_PROMPT_LENGTH,
        ),
        # Remove the top layer from the encoder, which cuts off the variance and only returns
        # the mean
        vae=tf.keras.Model(
            image_encoder.input,
            image_encoder.layers[-2].output,
        ),
        noise_scheduler=NoiseScheduler(),
        pretrained_ckpt=args.pretrained_ckpt,
        mp=args.mp,
        ema=args.ema,
        max_grad_norm=args.max_grad_norm,
        lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    print("Initializing optimizer...")
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        args.lr, decay_steps=args.decay_steps, alpha=args.alpha)
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=lr_decayed_fn,
        weight_decay=args.wd,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        epsilon=args.epsilon,
    )
    if args.mp:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    print("Compiling trainer...")
    diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")

    print("Training...")
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        save_weights_only=True,
        monitor="loss",
        mode="min",
    )
    diffusion_ft_trainer.fit(
        training_dataset,
        epochs=args.num_epochs,
        callbacks=[ckpt_callback]
    )


if __name__ == "__main__":
    args = parse_args()
    run(args)
