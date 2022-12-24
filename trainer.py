import copy

import tensorflow as tf
import tensorflow.experimental.numpy as tnp


class Trainer(tf.keras.Model):
    # Reference: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

    def __init__(
        self,
        diffusion_model,
        vae,
        noise_scheduler,
        ema=0.9999,
        max_grad_norm=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.diffusion_model = diffusion_model
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.max_grad_norm = max_grad_norm

        self.ema = ema
        self.ema_diffusion_model = copy.deepcopy(self.diffusion_model)

        self.vae.trainable = False

    def train_step(self, inputs):
        images = inputs["images"]
        encoded_text = inputs["encoded_text"]
        bsz = tf.shape(images)[0]

        with tf.GradientTape() as tape:
            # Project image into the latent space.
            latents = self.vae(images, training=False)
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = tf.random.normal(tf.shape(latents))

            # Sample a random timestep for each image
            timesteps = tnp.random.randint(
                0, self.noise_scheduler.train_timesteps, (bsz,)
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the target for loss depending on the prediction type
            # just the sampled noise for now.
            target = noise  # noise_schedule.predict_epsilon == True

            # Can be implemented:
            # https://github.com/huggingface/diffusers/blob/9be94d9c6659f7a0a804874f445291e3a84d61d4/src/diffusers/schedulers/scheduling_ddpm.py#L352

            # Predict the noise residual and compute loss
            timestep_embedding = tf.map_fn(
                lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
            )
            timestep_embedding = tf.squeeze(timestep_embedding, 1)
            model_pred = self.diffusion_model(
                [noisy_latents, timestep_embedding, encoded_text], training=True
            )
            loss = self.compiled_loss(
                target, model_pred, regularization_losses=self.losses
            )

        # Update parameters of the diffusion model.
        trainable_vars = self.diffusion_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # EMA
        for weight, ema_weight in zip(
            self.diffusion_model.weights, self.ema_diffusion_model.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {m.name: m.result() for m in self.metrics}

    def get_timestep_embedding(self, timestep, dim=320, max_period=10000):
        half = dim // 2
        log_max_preiod = tf.math.log(tf.cast(max_period, tf.float32))
        freqs = tf.math.exp(-log_max_preiod * tf.range(0, half, dtype=tf.float32) / half)
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return embedding  # Excluding the repeat.

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.ema_diffusion_model.save_weights(
            filepath=filepath,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
