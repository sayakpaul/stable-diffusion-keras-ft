"""
Adapted from  https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

# Usage
python finetune.py
"""
import json
import warnings

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

MAX_PROMPT_LENGTH = 77


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to fine-tune a Stable Diffusion model."
    )
    # Dataset related.
    parser.add_argument("--dataset_archive", default=None, type=str)
    parser.add_argument("--img_height", default=256, type=int)
    parser.add_argument("--img_width", default=256, type=int)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--augmentation", type=bool, default=False)

    # Optimization hyperparameters.
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--wd", default=1e-2, type=float)
    parser.add_argument("--beta_1", default=0.9, type=float)
    parser.add_argument("--beta_2", default=0.999, type=float)
    parser.add_argument("--epsilon", default=1e-08, type=float)
    parser.add_argument("--ema", default=0.9999, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--exp_signature", type=str, help="Experiment signature")

    # Training hyperparameters.
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)

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

    print("Fetch data from HDFS")
    os.system("hdfs dfs -get {}/{} .".format(args.log_dir, args.dataset_archive))

    print("Saving config...")
    config = json.dumps(args.__dict__)
    with tf.io.gfile.GFile(os.path.join(args.log_dir, args.exp_signature, 'config.json'), 'w') as f:
        f.write(config)

    print("Initializing dataset...")
    data_utils = DatasetUtils(
        dataset_archive=args.dataset_archive,
        batch_size=args.batch_size,
        img_height=args.img_height,
        img_width=args.img_width,
    )
    training_dataset = data_utils.prepare_dataset(augmentation=args.augmentation)

    print("Initializing trainer...")
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
        lora=True
    )

    checkpoint_path = os.path.join(args.log_dir, args.exp_signature, "checkpoint")
    if tf.io.gfile.exists(checkpoint_path):
        print("Found existing checkpoints, begin loading checkpoint")
        latest = tf.train.latest_checkpoint(checkpoint_path)
        print("Latest checkpoint is {}".format(latest))
        diffusion_ft_trainer.diffusion_model.load_weights(latest)

    print("Initializing optimizer...")
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=args.lr,
        weight_decay=args.wd,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        epsilon=args.epsilon,
    )
    if args.mp:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    print("Compiling trainer...")
    diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")
    # print(diffusion_ft_trainer.diffusion_model.summary())
    # print(diffusion_ft_trainer.vae.summary())

    print("Training...")
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.log_dir, args.exp_signature, "checkpoint", "cp-{epoch:04d}.ckpt"),
        save_weights_only=True,
        monitor="loss",
        mode="min",
    )
    train_tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.log_dir, args.exp_signature),
        histogram_freq=1,
        profile_batch='1,20'
    )
    diffusion_ft_trainer.fit(
        training_dataset,
        epochs=args.num_epochs,
        callbacks=[ckpt_callback, train_tensorboard_callback]
    )


if __name__ == "__main__":
    args = parse_args()
    run(args)
