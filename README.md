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

## ![Results](https://img.shields.io/badge/Results-8A2BE2?style=for-the-badge&logo=googleanalytics&logoColor=white)

### Early GAN Outputs
![Early GAN Output](https://github.com/user-attachments/assets/ef6fd094-d43e-471f-b179-989f79cdc9c2)
![Early GAN Output 2](https://github.com/user-attachments/assets/d5b5bfaf-182f-42ec-bd3b-ab4864b39496)

### DCGAN Iterations
![DCGAN Version Comparison](https://github.com/user-attachments/assets/41f4e773-4fd6-48a8-be87-1f0b448a8376)
![DCGAN Generation Sample](https://github.com/user-attachments/assets/97566044-b2d1-4583-8030-33d965d1e942)

### Stable Diffusion (DreamBooth + LoRA)
![Stable Diffusion Training Steps Comparison](https://github.com/user-attachments/assets/b934d7cf-f3f5-4ed8-a413-d39f36424559)

GAN and DCGAN Performance
Throughout the project, multiple iterations of GAN architectures were evaluated for synthesizing Pokémon-style images. The foundational vanilla GAN served as a baseline but struggled significantly with stability and resolution, frequently exhibiting mode collapse, blurry results, and a failure to capture coherent shapes.

Transitioning to a Deep Convolutional GAN (DCGAN) noticeably improved the structure and quality of the generated images over the vanilla GAN. To optimize the DCGAN, the architecture was trained through three distinct iterations:

* Version 1: A basic DCGAN implementation lacking stabilization techniques. While initial training showed progress, the model failed to produce usable images, resulting in plain outputs.

* Version 2: Introduced stabilization methods including label smoothing for real samples, noise injection into real images, and a slower discriminator learning rate. These changes delayed mode collapse and stabilized training dynamics; however, extended training ultimately led to a loss of diversity, with images degrading into colored blobs around epoch 700.

* Version 3: Applied a comprehensive stabilization strategy by injecting noise into both real and fake images, further lowering the generator learning rate, and implementing early stopping with periodic checkpointing. This version successfully avoided severe mode collapse, yielding generated images that retained significantly more visual diversity and structure.

Despite the architectural improvements in Version 3, the DCGAN models remained limited to lower resolutions (typically 64x64 or 128x128) and lacked the fine anatomical detail required for high-quality generation.

**Stable Diffusion Performance**
To address the limitations of the adversarial models, a Stable Diffusion v1.5 model was fine-tuned using DreamBooth and Low-Rank Adaptation (LoRA). This approach yielded significantly more detailed and coherent images.

* Visual Fidelity and Control: The diffusion model successfully synthesized realistic, Pokémon-inspired creatures at a high resolution of 512x512. Image sharpness and structural coherence improved noticeably across training checkpoints. Furthermore, the model provided highly customizable, prompt-driven generation capabilities guided by specific fine-tuned tokens.

* Observed Limitations: The model's outputs heavily favored Pikachu-like features, reflecting a class imbalance in the training dataset. Additionally, some generations—particularly early in training or when given ambiguous prompts—exhibited object fusion artifacts where Pokémon were blended with unrelated shapes.

**Conclusion** 
While the GAN-based models offered faster training times and conceptual simplicity, they ultimately lacked the expressive power and controllability required for the task. The Stable Diffusion model, enhanced by DreamBooth and LoRA, allowed for the specialization of a general-purpose generator without retraining from scratch. It proved to be the most effective solution, serving as the clear winner in terms of visual fidelity, adaptability, and high-quality personalized image generation.

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
