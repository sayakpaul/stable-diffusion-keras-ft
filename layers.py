from tensorflow import keras


class LoraLayer(keras.layers.Layer):
    def __init__(self, original_layer, rank=4, alpha=4., trainable=False, use_bias=False, **kwargs):
        # We want to keep the name of this layer the same as the original
        # dense layer.
        original_layer_config = original_layer.get_config()
        name = original_layer_config["name"]
        kwargs.pop("name", None)
        super().__init__(name=name, trainable=trainable, **kwargs)

        self.output_dim = original_layer_config["units"]
        if rank > self.output_dim:
            raise ValueError(f"LoRA rank {rank} must be less or equal than {self.output_dim}")
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.original_layer = original_layer
        self.original_layer.trainable = False

        self.down_layer = keras.layers.Dense(
            units=rank,
            use_bias=use_bias,
            kernel_initializer=keras.initializers.RandomNormal(stddev=1 / self.rank),
            trainable=trainable,
            name="lora_a"
        )

        self.up_layer = keras.layers.Dense(
            units=self.output_dim,
            use_bias=use_bias,
            kernel_initializer=keras.initializers.Zeros(),
            trainable=trainable,
            name="lora_b"
        )

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        lora_output = self.up_layer(self.down_layer(inputs)) * self.scale
        return original_output + lora_output
