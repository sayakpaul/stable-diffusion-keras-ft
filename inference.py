import argparse

import keras_cv
from PIL import Image

from trainer import load_sd_lora_layer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to do inference a Stable Diffusion model."
    )

    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--img_height", default=512, type=int)
    parser.add_argument("--img_width", default=512, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_steps", default=50, type=int)
    parser.add_argument("--checkpoint", default=None, type=str, help="Model checkpoint for loading model weights.")
    parser.add_argument("--lora", action="store_true", help="Whether to load loRA layer.")
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--lora_alpha", default=4, type=float)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--exp", default=None, type=str)

    return parser.parse_args()


def run(args):
    model = keras_cv.models.StableDiffusion(
        img_height=args.img_height, img_width=args.img_width, jit_compile=True
    )

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

    model.diffusion_model.load_weights(args.checkpoint)

    print("Begin generating images")
    images = model.text_to_image(
        prompt=args.prompt,
        num_steps=args.num_steps,
        batch_size=args.batch_size
    )

    for idx, img in enumerate(images):
        image_path = f"out-{idx}.png"
        Image.fromarray(img).save(image_path)


if __name__ == "__main__":
    args = parse_args()
    run(args)
