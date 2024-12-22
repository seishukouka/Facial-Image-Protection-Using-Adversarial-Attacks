# Facial-Image-Protection-Using-Adversarial-Attacks

This repository demonstrates an implementation of adversarial attacks (PGD + LPIPS) on facial images, aimed at **hindering personal identification** by modifying key facial features (eyes, nose, mouth). The modified faces are used as "real" data in DCGAN training, targeting **privacy protection** and **data augmentation** applications.


## Overview

- **PGD Attack + LPIPS**  
  We apply a targeted perturbation only to specified facial regions (e.g., eyes, nose, mouth). The perturbation makes the face embedding close to a specific “target embedding,” while using LPIPS (Learned Perceptual Image Patch Similarity) to limit unnatural distortion.

- **NonlinearEmbeddingMixer**  
  We combine embedding vectors from multiple faces into one target embedding, aiming for a more “diverse” alteration (i.e., to create a face embedding that is a mixture of several individuals).

- **DCGAN (Old Architecture) Re-training**  
  The adversarially modified faces are fed into a DCGAN as real data. We then observe how the attack influences the Generator and Discriminator during training.


<!-- You could insert a diagram or sample facial image modifications here, for better illustration (e.g.): -->
<!-- ![Overall Architecture](path/to/architecture.png) -->


## Learning process
- **train_adversarial.py**  
  1. Loads and reads images (CelebA-HQ)  
  2. Performs face landmark detection (dlib)  
  3. Mixes multiple embeddings via NonlinearEmbeddingMixer  
  4. Applies PGD attack + LPIPS penalty to partially perturb the face  
  5. Trains a DCGAN using these perturbed images as real data  
  6. Includes a training loop plus visualization & saving of generated results  

---

# Adversarial Face Protection

Adversarial Face Protection is a research project focused on adversarial attacks and facial protection. This repository includes tools for training and testing adversarial models on CelebA-HQ dataset.


## Setup

**Recommended Python version**: 3.8 or higher

### 1. Clone the Repository

```bash
git clone https://github.com/yourname/adversarial-face-protection.git
cd adversarial-face-protection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: Installing `dlib` may trigger a build process. You might need C++ build tools or similar, depending on your environment.

### 3. Prepare the Dataset

- **CelebA-HQ**: The dataset will be automatically downloaded from Hugging Face.
  - Please note that downloading large datasets can take significant time.

### 4. Facial Landmark Predictor File

Download `shape_predictor_68_face_landmarks.dat.bz2` from the [github](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat), extract it, and place the `.dat` file in the repository’s root folder.

---

## How to Run

Run the following command to start the training:

```bash
python main.py
```

### Default Configuration

- Takes the first 10,000 images of CelebA-HQ
- Selects up to 1,000 samples for training
- Runs for 20 epochs with a batch size of 32

### Output

- Model checkpoints will be saved in the `model/` directory for each epoch.
- A final checkpoint named `model_adv0.4_final.pth` is produced at the end.
- Sample images are periodically generated using a fixed noise vector, saved in `img_list`, and converted into a GIF, `celeba_adv.gif`.

---

## Results

### During Training

- Plots of `G_losses` and `D_losses` are displayed.
- The generated GIF, `celeba_adv.gif`, visualizes the evolution of samples over epochs.

### Key Features

- Adversarial attacks are applied to facial regions (eyes, nose, mouth).
- LPIPS (Learned Perceptual Image Patch Similarity) enforces a perceptual penalty to keep modifications visually natural.

<!-- Optional: Add visual aids -->
<!-- ![Loss Curve](path/to/loss_plot.png) -->
<!-- ![Before & After Attack](path/to/before_after.png) -->

---

## Notes

1. **Ethical Use**: This code is for research purposes only. Any unethical or illegal use is strictly prohibited.
2. **Face Detection Limitations**: 
   - `dlib`'s face landmark detection may fail or mis-detect in some cases.
   - The code skips such images. Consider adding more robust error handling if needed.
3. **High GPU Memory Usage**:
   - CelebA-HQ images are high-resolution (1024×1024).
   - If you encounter memory issues, reduce `max_samples` or `bsize` in the configuration.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code, please cite the following resources:

- FaceNet PyTorch ([facenet_pytorch](https://github.com/timesler/facenet-pytorch))
- dlib (face landmark detection)
- LPIPS ([Learned Perceptual Image Patch Similarity](https://github.com/richzhang/PerceptualSimilarity))
- CelebA-HQ Dataset

---

## Contributing

Pull requests and issues are welcome! Feel free to open a PR or issue for:

- Bug reports
- Feature requests
- Documentation improvements

---
