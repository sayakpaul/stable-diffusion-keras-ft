# This script is for loading model checkpoint then save to h5 format
import tensorflow as tf
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler

from trainer import Trainer

MAX_PROMPT_LENGTH = 77

IMG_HEIGHT = IMG_WIDTH = 512
image_encoder = ImageEncoder(IMG_HEIGHT, IMG_WIDTH)
model = Trainer(
    diffusion_model=DiffusionModel(
        IMG_HEIGHT, IMG_WIDTH, MAX_PROMPT_LENGTH
    ),
    # Remove the top layer from the encoder, which cuts off the variance and only returns
    # the mean
    vae=tf.keras.Model(
        image_encoder.input,
        image_encoder.layers[-2].output,
    ),
    noise_scheduler=NoiseScheduler(),
    pretrained_ckpt=None,
    mp=True,
    ema=0.9999,
    max_grad_norm=1,
)

latest = tf.train.latest_checkpoint("checkpoint")
model.diffusion_model.load_weights(latest)
model.diffusion_model.save_weights("fine_tune_exp1.h5")
