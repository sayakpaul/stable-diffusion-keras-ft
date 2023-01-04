# Fine-tuning Stable Diffusion using Keras

This repository provides code for fine-tuning [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) in Keras. It is adapted from this [script by Hugging Face](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py). The pre-trained model used for fine-tuning comes from [KerasCV](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion). To know about the original model check out [this documentation](https://huggingface.co/CompVis/stable-diffusion-v1-4).  

**The code provided in this repository is for research purposes only**. Please check out [this section](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion#uses) to know more about the potential use cases and limitations.

By loading this model you accept the CreativeML Open RAIL-M license at https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE.

<div align="center">
<img src="https://i.imgur.com/X4m614M.png"/>
</div>

If you're just looking for the accompanying resources of this repository, here are the links:

* [Inference Colab Notebook](https://colab.research.google.com/github/sayakpaul/stable-diffusion-keras-ft/blob/main/notebooks/generate_images_with_finetuned_stable_diffusion.ipynb)
* Blog post (upcoming)
* [Interactive Hugging Face Space application](https://huggingface.co/spaces/sayakpaul/pokemon-sd-kerascv)
* [Fine-tuned model weights](https://huggingface.co/sayakpaul/kerascv_sd_pokemon_finetuned)

**Table of contents**:

* [Dataset](#dataset)
* [Training and additional details](#training)
* [Inference](#inference)
* [Results](#results)
* [Acknowledgements](#acknowledgements)

> This repository has a sister repository ([keras-sd-serving](https://github.com/deep-diver/keras-sd-serving)) that covers various deployment patterns for Stable Diffusion. 

## Dataset 

Following the original script from Hugging Face, this repository also uses the [Pokemon dataset](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions). But it was regenerated to work better with `tf.data`. The regenerated version of the dataset is hosted [here](https://huggingface.co/datasets/sayakpaul/pokemon-blip-original-version). Check out that link for more details.

## Training

Fine-tuning code is provided in `finetune.py`. Before running training, ensure you have the dependencies (refer to `requirements.txt`) installed.

You can launch training with the default arguments by running `python finetune.py`. Run `python finetune.py -h` to know about the supported command-line arguments. You can enable mixed-precision training by passing the `--mp` flag.

When you launch training, a diffusion model checkpoint will be generated epoch-wise only if the current loss is lower than the previous one.

For avoiding OOM and faster training, it's recommended to use a V100 GPU at least. We used an A100.   

**Some important details to note**:

* Distributed training is not yet supported. Gradient accumulation and gradient checkpointing are also not supported.
* Only the diffusion model is fine-tuned. The VAE and the text encoder are kept frozen. 


**Training details**:

We fine-tuned the model on two different resolutions: 256x256 and 512x512. We only varied the batch size and number of epochs for fine-tuning
with these two different resolutions. Since we didn't use gradient accumulation, we use [this code snippet](https://github.com/huggingface/diffusers/blob/b693aff7951c8562a2d11664dd78667c5a97640e/examples/text_to_image/train_text_to_image.py#L568-L572) to derive the number of epochs. 

* 256x256: `python finetune.py --batch_size 4 --num_epochs 577`
* 512x512: `python finetune.py --img_height 512 --img_width 512 --batch_size 1 --num_epochs 72 --mp`

For 256x256 resolution, we intentionally reduced the number of epochs to save compute time.

**Fine-tuned weights**:

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

IMG_HEIGHT = IMG_WIDTH = 512


def plot_images(images, title):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.title(title)
        plt.imshow(images[i])
        plt.axis("off")


# We just have to load the fine-tuned weights into the diffusion model.
weights_path = keras.utils.get_file(
    origin="https://huggingface.co/sayakpaul/kerascv_sd_pokemon_finetuned/resolve/main/ckpt_epochs_72_res_512_mp_True.h5"
)
pokemon_model = keras_cv.models.StableDiffusion(
    img_height=IMG_HEIGHT, img_width=IMG_WIDTH
)
pokemon_model.diffusion_model.load_weights(weights_path)

# Generate images.
generated_images = pokemon_model.text_to_image("Yoda", batch_size=3)
plot_images(generated_images, "Fine-tuned on the Pokemon dataset")
```

You can bring in your `weights_path` (should be compatible with the `diffusion_model`) and reuse the code snippet. 

Check out this [Colab Notebook](https://colab.research.google.com/github/sayakpaul/stable-diffusion-keras-ft/blob/main/notebooks/generate_images_with_finetuned_stable_diffusion.ipynb) to play with the inference code.

## Results

Initially, we fine-tuned the model on a resolution of 256x256. Here are some results along with comparisons to the results
of the original model. 

<div align="left">
<table>
  <tr>
    <th>Images</img></th>
    <th>Prompts</img></th>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/6eT1hPa.png"></img></td>
    <td>Yoda</td>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/q4tmr4a.png"></img></td>
    <td>robotic cat with wings</td>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/kaD8uRp.png> "></img></td>
    <td>Hello Kitty</td>
  </tr>
</table>
<sub><a href="https://huggingface.co/sayakpaul/kerascv_sd_pokemon_finetuned/resolve/main/ckpt_epochs_577_res_256_mp_False.h5">Weights</a></sub>
</div><br>

We can see that the fine-tuned model has more stable outputs than the original model. Even though the results can be aesthetically improved much more, the fine-tuning effects are visible. Also, we followed the same hyperparameters from [Hugging Face's script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) for the 256x256 resolution (apart from number of epochs and batch size). With 
better hyperparameters, the results will likely improve.

For the 512x512 resolution, we observe something similar. So, we experimented with the `unconditional_guidance_scale` parameter and noticed that when it's set to 40 (while keeping the other arguments fixed), the results came out better.

<div align="left">
<table>
  <tr>
    <th>Images</img></th>
    <th>Prompts</img></th>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/NfM3iOl.png"></img></td>
    <td>Yoda</td>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/ow8sKMH.png"></img></td>
    <td>robotic cat with wings</td>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/AG10vu5.png"></img></td>
    <td>Hello Kitty</td>
  </tr>
</table>
<sub><a href="https://huggingface.co/sayakpaul/kerascv_sd_pokemon_finetuned/resolve/main/ckpt_epochs_72_res_512_mp_True.h5">Weights</a></sub>
</div><br>

**Note**: Fine-tuning on the 512x512 is still in progress as of this writing. But it takes a lot of time to complete a single epoch without the presence of distributed training and gradient accumulation. The above results are from the checkpoint derived after 60th epoch. 

With a similar recipe (but trained for more optimization steps), Lambda Labs demonstrate [amazing results](https://github.com/LambdaLabsML/examples/tree/main/stable-diffusion-finetuning).

## Acknowledgements

* Thanks to Hugging Face for providing the fine-tuning script. It's very readable and easy to understand.
* Thanks to the ML Developer Programs' team at Google for providing GCP credits.
