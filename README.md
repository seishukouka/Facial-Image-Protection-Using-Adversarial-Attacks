# Facial Image Protection Using Adversarial Attacks

This repository provides an implementation of adversarial attacks (PGD + LPIPS) on facial images, specifically targeting key facial regions (eyes, nose, mouth). By applying adversarial perturbations that shift the face embedding toward a target embedding (while constraining image quality via LPIPS), we aim to hinder personal identification. These modified faces are then used as “real” data for a DCGAN (old architecture) to explore privacy-protecting data augmentation.

---

## Overview

- **PGD Attack + LPIPS**  
  Adversarial perturbations are confined to specific facial parts (e.g., eyes, nose, mouth). We steer the face embedding toward a certain target embedding while imposing an LPIPS-based penalty to preserve perceptual quality.

- **NonlinearEmbeddingMixer**  
  Multiple face embeddings are combined into one “mixed” target embedding, encouraging a more diverse and “non-identifiable” transformation.

- **DCGAN (Old Architecture) Training**  
  After applying adversarial modifications, the resulting images serve as real data for a DCGAN. We then observe how the Generator and Discriminator respond to adversarially altered faces.

- **CelebA-HQ Dataset via Hugging Face**  
  We utilize [CelebA-HQ](https://arxiv.org/abs/1710.10196) loaded through the Hugging Face `datasets` library.

---

# Adversarial Face Protection

Adversarial Face Protection is a research project focused on adversarial attacks and facial protection. This repository includes tools for training and testing adversarial models on CelebA-HQ dataset.

---

## Setup

**Recommended Python version**: 3.8 or higher

### 1. Clone the Repository

```bash
git clone https://github.com/seishukouka/adversarial-face-protection.git
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

Download `shape_predictor_68_face_landmarks.dat.bz2` from the [Github](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2/), extract it, and place the `.dat` file in the repository’s root folder.

---

## How to Run

Run the following command to start the training:

```bash
python3 train_adversarial.py
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


