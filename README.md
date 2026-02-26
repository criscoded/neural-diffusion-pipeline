# Neural Diffusion Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

A machine learning pipeline focused on synthetic image generation using Generative Adversarial Networks (GANs) and Stable Diffusion. This project explores the trade-offs between lightweight GAN-based models and high-fidelity diffusion-based models, culminating in a personalized Low-Rank Adaptation (LoRA) fine-tuned model.

## 📊 Dataset & Preprocessing

The training data relies on a curated dataset of approximately 819 Pokémon images  
Dataset: https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset

* **Resolution Scaling:** While initially scaled to 256x256, the images were dynamically resized during training depending on the model pipeline: 128x128 for the DCGAN and 512x512 for Stable Diffusion DreamBooth training
* **Note on Dataset Imbalance:** The dataset inherently exhibits a class imbalance, featuring a wide variety of characters but with Pikachu and its various evolutions appearing much more frequently (typically a 2-4:1 ratio compared to other Pokémon).

## 🧠 Modeling Approaches & Evolution

Throughout the project, multiple generative modeling strategies were progressively tested and evaluated:

### 1. DCGAN (Deep Convolutional GAN) 

Built using TensorFlow, the DCGAN was trained incrementally through three distinct versions to combat adversarial instability and mode collapse:
* **Version 1:** A basic architecture trained for 500 epochs with no stabilization techniques, which failed to produce usable images
* **Version 2:** Introduced label smoothing, noise injection into real images, and a slower discriminator learning rate. This stabilized training but ultimately suffered mode collapse around epoch 700
* **Version 3:** Added noise injection to both real and fake images, further lowered the generator learning rate, and implemented early stopping and periodic checkpointing. Training halted gracefully at epoch 217, preserving much better visual diversity without complete collapse

### 2. StyleGAN2 
* Attempted training with StyleGAN2-ADA on Apple M4 Max hardware utilizing PyTorch MPS (Metal Performance Shaders) 
* **Result:** Unsuccessful due to a lack of CUDA support and limitations with PyTorch MPS operators—specifically, the 2D grid sampler required by StyleGAN2 was unsupported and unworkable

### 3. Stable Diffusion + DreamBooth with LoRA (Low-Rank Adaptation)
Transitioned to Stable Diffusion v1.5, fine-tuning the model using DreamBooth and Low-Rank Adaptation (LoRA) for personalized prompt conditioning at a 512x512 resolution
* Trained over 1000 steps utilizing the Hugging Face `diffusers` library on an Apple Silicon MPS backend
* **Result:** Produced detailed, highly coherent Pokémon-inspired creatures, completely mitigating the blurry artifacts of the GAN models 

## ✍️ Prompt Engineering Strategy

To effectively harness the diffusion model and reject dataset noise, inference was conducted through a strict, progressive prompt engineering strategy:
1. **Initial Run:** `"a sks pokémon"` – Yielded target shapes but intermixed heavily with generalist diffusion traits
2. **Constrained Guidance:** `"a 2D sks creature in Pokémon style, full-body, centered, clean background"` paired with a heavy negative prompt (`"photo, realistic, laptop, weapon, human, text, signature, watermark, blurry, deformed"`) effectively isolated the subject
3. **Artistic Cohesion:** `"a 2D sks creature in Pokémon style, full-body, centered, forest background"` provided a consistent environmental context, resulting in highly convincing and artistically aligned outputs

## 📂 Project Structure

```
neural-diffusion-pipeline/
├── train_dreambooth_lora.py   # Main Hugging Face training script for LoRA fine-tuning
├── rosa_diffusion_pipeline.ipynb  # Jupyter Notebook containing DCGAN iterations and Diffusion eval
├── README.md                  # Project documentation
└── .gitignore                 # Specifies untracked data/weights
```

Note: To maintain repository health, the raw pokemon_jpg/ dataset and the generated pokemon_lora_output/ weights are intentionally excluded from version control

## ⚙️ Local Development & Training Setup

**Prerequisites:**

* Python 3.8+

* PyTorch (with MPS support for Apple Silicon)

* Hugging Face ecosystem (diffusers, transformers, accelerate, peft)

**Training the LoRA Model**

The repository utilizes train_dreambooth_lora.py for fine-tuning. Below is the execution command used to train the Stable Diffusion model for 1000 steps at a 512x512 resolution:

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="pokemon_lora/instance_images"
export OUTPUT_DIR="pokemon_lora_output"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks pokémon" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --checkpointing_steps=100 \
  --seed=42 \
  --mixed_precision="no" \
  --validation_prompt="a sks pokémon" \
  --validation_epochs=100 \
  --report_to="tensorboard"
```

## 🔮 Future Roadmap

* Advanced Prompting: Integrate retrieval-augmented prompts to further enhance generation diversity.
* Captioning Modules: Train custom captioning modules to actively reduce class bias and enable more specific character targeting.

## 📝 License

Distributed under the MIT License. See LICENSE for more information.

## 👨‍💻 Author

Cristofer Rosa
* GitHub: @criscoded
* LinkedIn: /in/criscoded
* Email: cristofercodes@gmail.com
