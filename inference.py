import argparse
import os

import keras_cv
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

from trainer import load_sd_lora_layer

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"


def preprocess_image(image):
    """ Loads image from path and preprocesses to make it model ready
        Args:
          image: image numpy value.
    """
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if image.shape[-1] == 4:
        image = image[..., :-1]
    hr_size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)


def save_image(image, filename):
    """
      Saves unscaled Tensor Images.
      Args:
        image: 3D image tensor. [height, width, channels]
        filename: Name of the file to save.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s" % filename)
    print("Saved as %s" % filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to do inference a Stable Diffusion model."
    )

    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--img_height", default=512, type=int)
    parser.add_argument("--img_width", default=512, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_steps", default=50, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--lora", action="store_true", help="Whether to load loRA layer.")
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--lora_alpha", default=4, type=int)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--exp", default=None, type=str)

    return parser.parse_args()


def run(args):
    model = keras_cv.models.StableDiffusion(
        img_height=args.img_height, img_width=args.img_width, jit_compile=True
    )
    upscale_model = hub.load(SAVED_MODEL_PATH)

    if args.lora:
        print("Loading LoRA layer")
        load_sd_lora_layer(
            diffusion_model=model.diffusion_model,
            img_height=args.img_height,
            img_width=args.img_width,
            rank=args.lora_rank,
            alpha=args.lora_alpha
        )
        # Just to make sure.
        model.diffusion_model.trainable = False

    checkpoint = os.path.join(args.log_dir, args.exp, "checkpoint", "cp-{}.ckpt".format(args.checkpoint.zfill(4)))
    model.diffusion_model.load_weights(checkpoint)

    print("Begin generating images")
    images = model.text_to_image(
        prompt=args.prompt,
        num_steps=args.num_steps,
        batch_size=args.batch_size
    )

    hdfs_path = os.path.join(args.log_dir, args.exp, "output/")
    for idx, img in enumerate(images):
        image_path = f"out-{idx}.png"
        upscale_path = f"upscaled-{idx}.png"
        Image.fromarray(img).save(image_path)
        hr_image = preprocess_image(img)
        upscale_image = upscale_model(hr_image)
        save_image(tf.squeeze(upscale_image), upscale_path)
        os.system("hdfs dfs -mkdir -p {}".format(hdfs_path))
        os.system("hdfs dfs -put {} {}".format(image_path, hdfs_path))
        os.system("hdfs dfs -put {} {}".format(upscale_path, hdfs_path))


if __name__ == "__main__":
    args = parse_args()
    run(args)
