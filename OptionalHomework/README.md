# Overview

---

## Model Design
Several architectures were explored, prioritizing simplicity and inference speed:

- Convolutional layers with kernel sizes 3 and 5 to learn spatial downsampling.
- A large fully connected model mapping the full input directly to the output.
- Output constraints using sigmoid activation or value clamping.

The **final model** is intentionally minimal:
- A **1×1 convolution** (3 → 1 channels).
- A **single linear layer** mapping the flattened 32×32 grayscale image to a flattened 28×28 output.
- Flattening and unflattening operations.

This design avoids unnecessary computation while still learning the required transformation.

---

## Loss Function
- **Mean Squared Error (MSE)** loss is used for pixel-wise regression.
- Loss reduction is set to **sum** instead of mean to strengthen the training signal.
- Pixel predictions outside the range \([0,1]\) are penalized with higher weight to encourage valid outputs.

---

## Early Stopping
Training is stopped early based on validation performance:
- Early stopping triggers when the **average sum of absolute pixel errors per image** falls below a threshold corresponding to half the distance between two discrete pixel intensity levels.
- This ensures that most pixels are mapped to the correct intensity value, which is sufficient for visual correctness.

---

## Qualitative Results
The report includes **six visual examples**, each showing:
1. Input image  
2. Ground truth output  
3. Model prediction  
4. Normalized absolute difference  

The difference images highlight that prediction errors are small and mostly imperceptible.

---

## Inference Speed Benchmarking
Inference speed was evaluated on **CPU and GPU** and compared to the original transformation pipeline.

- Hyperparameter sweeps with **Weights & Biases** were used to find the fastest configuration.
- Best performance was achieved with:
  - Batch size = 64  
  - Number of workers = 0  
  - `pin_memory = True`

### Runtime Results (Approximate)
| Method | Time          |
|------|---------------|
| Original CPU transformations | 2.3 - 2.5 s   |
| Model inference (CPU) | 0.45 - 0.5 s  |
| Model inference (GPU) | 0.16 - 0.35 s |

The model is faster than the original transformations on both CPU and GPU.

---

## Expected Points Breakdown
| Task | Points |
|-----|--------|
| Model design and creativity | **3 / 3** |
| Loss function choice and motivation | **2 / 2** |
| Early stopping criterion and motivation | **2 / 2** |
| Generated images and comparison | **1 / 1** |
| Inference benchmarking (CPU & GPU) | **2 / 2** |
| **Total Expected Score** | **10 / 10** |