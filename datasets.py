"""
Adapted from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""

import os
from typing import Dict, Tuple

import keras_cv
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder

PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77
AUTO = tf.data.AUTOTUNE
POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
DEFAULT_DATA_ARCHIVE = "https://huggingface.co/datasets/sayakpaul/pokemon-blip-original-version/resolve/main/pokemon_dataset.tar.gz"


class DatasetUtils:
    def __init__(
            self,
            dataset_archive: str = None,
            batch_size: int = 4,
            img_height: int = 256,
            img_width: int = 256,
    ):
        self.tokenizer = SimpleTokenizer()
        self.text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
        self.augmenter = keras_cv.layers.Augmenter(
            layers=[
                keras_cv.layers.CenterCrop(img_height, img_width),
                keras_cv.layers.RandomFlip(),
                tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
            ]
        )

        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        if dataset_archive is None:
            data_path = tf.keras.utils.get_file(
                origin=DEFAULT_DATA_ARCHIVE,
                untar=True,
            )
            self.data_frame = pd.read_csv(os.path.join(data_path, "data.csv"))
        else:
            data_path = tf.keras.utils.get_file(
                origin="file://" + os.path.abspath('') + "/" + dataset_archive,
                untar=True,
            )
            # TODO: Remove sep tab for general usage
            self.data_frame = pd.read_csv(os.path.join(data_path, "data.csv"), sep="\t")

        self.data_frame["image_path"] = self.data_frame["image_path"].apply(
            lambda x: os.path.join(data_path, x)
        )

    def process_text(self, caption: str) -> np.ndarray:
        tokens = self.tokenizer.encode(caption)
        tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
        return np.array(tokens)

    def process_image(
            self, image_path: tf.Tensor, tokenized_text: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, 3)
        image = tf.image.resize(image, (self.img_height, self.img_width))
        return image, tokenized_text

    def apply_augmentation(
            self, image_batch: tf.Tensor, token_batch: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.augmenter(image_batch), token_batch

    def run_text_encoder(
            self, image_batch: tf.Tensor, token_batch: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Since the text encoder will remain frozen we can precompute it.
        return (
            image_batch,
            token_batch,
            self.text_encoder([token_batch, POS_IDS], training=False),
        )

    def prepare_dict(
            self,
            image_batch: tf.Tensor,
            token_batch: tf.Tensor,
            encoded_text_batch: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        return {
            "images": image_batch,
            "tokens": token_batch,
            "encoded_text": encoded_text_batch,
        }

    def prepare_dataset(self, augmentation=True) -> tf.data.Dataset:
        all_captions = list(self.data_frame["caption"].values)
        tokenized_texts = np.empty((len(self.data_frame), MAX_PROMPT_LENGTH))
        for i, caption in enumerate(all_captions):
            tokenized_texts[i] = self.process_text(caption)

        image_paths = np.array(self.data_frame["image_path"])

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, tokenized_texts))
        dataset = dataset.shuffle(self.batch_size * 10)
        dataset = dataset.map(self.process_image, num_parallel_calls=AUTO).batch(
            self.batch_size
        )
        if augmentation:
            dataset = dataset.map(self.apply_augmentation, num_parallel_calls=AUTO)
        dataset = dataset.map(self.run_text_encoder, num_parallel_calls=AUTO)
        dataset = dataset.map(self.prepare_dict, num_parallel_calls=AUTO)
        return dataset.prefetch(AUTO)
