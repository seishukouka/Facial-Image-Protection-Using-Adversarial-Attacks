# Facial-Image-Protection-Using-Adversarial-Attacks

This repository demonstrates an implementation of adversarial attacks (PGD + LPIPS) on facial images, aimed at **hindering personal identification** by modifying key facial features (eyes, nose, mouth). The modified faces are used as "real" data in DCGAN training, targeting **privacy protection** and **data augmentation** applications.

---

## Overview

- **PGD Attack + LPIPS**  
  We apply a targeted perturbation only to specified facial regions (e.g., eyes, nose, mouth). The perturbation makes the face embedding close to a specific “target embedding,” while using LPIPS (Learned Perceptual Image Patch Similarity) to limit unnatural distortion.

- **NonlinearEmbeddingMixer**  
  We combine embedding vectors from multiple faces into one target embedding, aiming for a more “diverse” alteration (i.e., to create a face embedding that is a mixture of several individuals).

- **DCGAN (Old Architecture) Re-training**  
  The adversarially modified faces are fed into a DCGAN as real data. We then observe how the attack influences the Generator and Discriminator during training.

- **CelebA-HQ Dataset via Hugging Face**  
  We use [CelebA-HQ](https://arxiv.org/abs/1710.10196), loaded with the Hugging Face `datasets` package.

---

<!-- You could insert a diagram or sample facial image modifications here, for better illustration (e.g.): -->
<!-- ![Overall Architecture](path/to/architecture.png) -->

---

## Demo & Generated Results

During training, the Generator outputs are saved periodically and visualized as a GIF animation.  
- `celeba_adv.gif` illustrates the progression of the generated samples over time.

<!-- Insert actual generated sample images or GIFs here (e.g.): -->
<!-- ![Sample Generation](path/to/generated_samples.gif) -->

---

## File Structure
. ├── model/ │ ├── model_no_noise.pth # Pre-trained DCGAN model │ └── model_adv0.4_epoch_x.pth # Model checkpoints after adversarial training (per epoch) ├── datasets/ │ └── (CelebA-HQ downloaded via Hugging Face API) ├── shape_predictor_68_face_landmarks.dat ├── requirements.txt └── main.py # Core implementation (includes all steps below)


- **main.py**  
  1. Loads and reads images (CelebA-HQ)  
  2. Performs face landmark detection (dlib)  
  3. Mixes multiple embeddings via NonlinearEmbeddingMixer  
  4. Applies PGD attack + LPIPS penalty to partially perturb the face  
  5. Trains a DCGAN using these perturbed images as real data  
  6. Includes a training loop plus visualization & saving of generated results  

---

## Setup

**Recommended Python version**: 3.8 or higher

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourname/adversarial-face-protection.git
   cd adversarial-face-protection
Install dependencies

bash
コードをコピーする
pip install -r requirements.txt
Note: Installing dlib may trigger a build process. You may need C++ build tools or similar, depending on your environment.

Prepare the dataset

CelebA-HQ will be automatically downloaded from Hugging Face.
Please be aware that downloading large datasets can take significant time.
Facial landmark predictor file

Download shape_predictor_68_face_landmarks.dat.bz2 from the dlib site, extract it, and place the .dat file in the repository’s root folder.
How to Run
bash
コードをコピーする
python main.py
By default, the script will take the first 10,000 images of CelebA-HQ, select up to 1,000 for training, and run for 20 epochs with batch size 32.
Model checkpoints will be saved in the model/ directory each epoch, and a final checkpoint named model_adv0.4_final.pth is produced at the end.
Periodically, we generate sample images using a fixed noise vector; these are gathered in img_list and converted into a GIF, celeba_adv.gif.
Results
During training, you will see plots of G_losses and D_losses. The file celeba_adv.gif shows the evolution of the generated samples.
Since we apply adversarial attacks to facial regions (eyes, nose, mouth), the final images include altered features that differ from the original identity. Meanwhile, LPIPS enforces a perceptual penalty to keep the modifications somewhat natural.
<!-- You can insert a loss plot or before/after images here for further clarity --> <!-- ![Loss Curve](path/to/loss_plot.png) --> <!-- ![Before & After Attack](path/to/before_after.png) -->
Notes
This code is intended for research purposes only; any unethical or illegal use is prohibited.
dlib’s face landmark detection may fail or mis-detect in some cases. The code skips images in such cases. Consider additional error handling if needed.
CelebA-HQ is high-resolution (1024×1024), so GPU memory usage is high. If you encounter out-of-memory errors, consider reducing max_samples or bsize further.
License
Released under the MIT License. Please see the LICENSE file for details.

Citation
FaceNet PyTorch (facenet_pytorch)
dlib (face landmark detection)
LPIPS (Learned Perceptual Image Patch Similarity)
CelebA-HQ Dataset
Contributing
Pull requests and issues are welcome. Feel free to open a PR or issue for bug reports, feature requests, or documentation improvements.

