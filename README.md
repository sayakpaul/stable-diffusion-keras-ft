# Fine-tuning Stable Diffusion using Keras

This repository provides code for fine-tuning [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) in Keras. It is adapted from this [script by Hugging Face](https://github.com/fchollet/stable-diffusion-tensorflow/blob/master/text2image.py). The pre-trained model used for fine-tuning comes from [KerasCV](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion). To know about the original model check out [this documentation](https://huggingface.co/CompVis/stable-diffusion-v1-4).  

**The code provided in this repository is for research purposes only**. Please check out [this section](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion#uses) to know more about the potential use cases and limitations.

By loading this model you accept the CreativeML Open RAIL-M license at https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE.

[Add results]

## Dataset 

Following the original script from Hugging Face, this repository also uses the [Pokemon dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions). But it was regenerated to suit this repository. The regenerated version of the dataset is hosted [here](https://huggingface.co/datasets/sayakpaul/pokemon-blip-original-version). Check out that link for more details.

## Training

Fine-tuning code is provided in `finetune.py`. Before running training, ensure you have the dependencies (refer to `requirements.txt`) installed.

You can launch training with the default arguments by running `python finetune.py`. Run `python finetune.py -h` to know about the supported command-line arguments.

For avoiding OOM and faster training, it's recommended to use a V100 GPU at least. We used an A100.

**Some details to note**:

* Only the diffusion model is fine-tuned.The image encoder and the text encoder are kept frozen. 
* Mixed-precision training is not yet supported. As a result, instead of 512x512 resolution, this repository uses 256x256.
* Distributed training is not yet supported. 
* One major difference from the Hugging Face implementation is that the EMA averaging of weights doesn't follow any schedule for the decay factor.

You can find the fine-tuned diffusion model weights [here](https://huggingface.co/sayakpaul/kerascv_sd_pokemon_finetuned/tree/main). 

### Training with custom data

The default Pokemon dataset used in this repository comes with the following structure:

```bash 
pokemon_dataset/
    data.csv
    image_24.png   
    image_3.png    
    image_550.png  
    image_700.png
    ...
```

`data.csv` looks like so:

![](https://i.imgur.com/AeRqWPH.png)

As long as your custom dataset follows this structure, you don't need to change anything in the current codebase except for the `dataset_archive`.

In case your dataset has multiple captions per image, you can randomly select one from the pool of captions per image during training.

Based on the dataset, you might have to tune the hyperparameters.

## Inference

```py
import keras_cv
import matplotlib.pyplot as plt
from tensorflow import keras

IMG_HEIGHT = IMG_WIDTH = 256


def plot_images(images, title):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.title(title)
        plt.imshow(images[i])
        plt.axis("off")


# We just have to load the fine-tuned weights into the diffusion model.
weights_path = keras.utils.get_file(
    origin="https://huggingface.co/sayakpaul/kerascv_sd_pokemon_finetuned/resolve/main/ema_diffusion_model.h5"
)
pokemon_model = keras_cv.models.StableDiffusion(
    img_height=IMG_HEIGHT, img_width=IMG_WIDTH
)
pokemon_model.diffusion_model.load_weights(weights_path)

# Generate images.
generated_images = pokemon_model.text_to_image("Yoda", batch_size=3)
plot_images(generated_images, "Fine-tuned on the Pokemon dataset")
```

You can check out this [Colab Notebook] (TODO) to play with the code.

## Results

Upcoming (there should be note on running hyperparameter tuning as the Hugging Face tutorial)

## Acknowledgements

* Thanks to Hugging Face for providing the fine-tuning script. It's quite readable.
* Thanks to the ML Developer Programs' team at Google for providing GCP credits.