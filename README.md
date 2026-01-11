# Universal Adversarial Perturbations on CIFAR-10  
### Machine Learning Security – Final Project

This project analyzes **Universal Adversarial Perturbations (UAPs)** in the context of CIFAR-10 image classification, with particular attention to **model robustness** and **transferability (black-box attacks)** across different neural network architectures.

The considered models are:
- `my_resnet18` (custom implementation)
- `tv_resnet18` (torchvision ResNet-18 adapted to CIFAR-10)
- `densenet_light` (compact DenseNet for CIFAR-10)

---

## Requirements

### Python
- Python ≥ 3.9

### External libraries
```
pip install torch torchvision torchaudio numpy matplotlib
```

---

## Project structure

```text
project_root/
│
├── run_clean_experiment.py      # Clean training + best checkpoint selection
├── load_models.py               # Load and freeze trained models
├── run_save_uap.py              # UAP generation and saving (pixel-space)
├── uap_load_view.py             # Load and visualize saved UAPs
├── eval_uap.py                  # Clean vs adversarial evaluation + fooling rate
├── config.py                    # Global configuration (hyperparameters, paths, eps)
├── data.py                      # CIFAR-10 DataLoader (clean and pixel-space)
├── uap.py                       # UAP generation utilities
│
├── models/
│   ├── my_resnet18.py           # Custom ResNet-18 implementation
│   ├── tv_resnet18.py           # torchvision ResNet-18 adapted to CIFAR-10
│   └── densenet.py              # Light DenseNet for CIFAR-10
│
├── checkpoints_compare/         # Best clean checkpoints
│   ├── my_resnet18_best.pth
│   ├── tv_resnet18_best.pth
│   └── densenet_light_best.pth
│
├── uaps/                        # Saved Universal Adversarial Perturbations (.pth)
│   ├── uap_my_resnet18_eps*.pth
│   ├── uap_tv_resnet18_eps*.pth
│   └── uap_densenet_light_eps*.pth
│
├── uap_img/                     # UAP visualization images
│   ├── uap_lambda_*.png
│   └── image_perturbation_*.png
│
├── data/                        # CIFAR-10 dataset
│   └── cifar-10-batches-py
│
├── output/                      # Example outputs from code runs
│   └── example_output_*.txt
│
└── README.md                    # Project documentation
```

---

## 1. Clean model training
Models are trained on CIFAR-10 under clean conditions.

For each model, the best checkpoint is selected based on test accuracy.

### A. Training all models
From the project root directory:: ```python run_clean_experiment.py```

### B. Training a single model
From the project root directory:: ```python run_clean_experiment.py --model <nome_modello>```

## 2. Loading and freezing models
Loads the best checkpoints and freezes the models (`eval mode` and `requires_grad=False`) for UAP generation.

### A. Load all models 
From the project root directory:: ```python load_models.py```

### B. Load a specific model
From the project root directory:: ```python load_models.py --model <nome_modello>```

## 3. Universal Adversarial Perturbation (UAP) generation
UAPs are computed in pixel-space and saved in the *uaps/* directory.

The perturbation budget ***ε*** is defined in config.py through the `EPS_PIX` variable.

### A. Generate UAP for a single model
From the project root directory:: ```python run_save_uap.py <model_name>```

### B. Generate UAPs for all models
From the project root directory:: ```python run_save_uap.py```

## 4. Loading and visualizing UAPs
The visualization includes:
- the universal perturbation ***δ***
- an example of original vs perturbed image
- all images are saved in *uap_img/*.

### A. Visualize a specific UAP
From the project root directory:: ```python uap_load_view.py <nome_file>```

### B. Visualize all saved UAPs
From the project root directory:: ```python uap_load_view.py```

## 5. Adversarial evaluation (clean vs UAP)
The following metrics are computed:
- Clean accuracy
- Adversarial accuracy
- Fooling rate

### A. Evaluate all Model × UAP combinations
From the project root directory:: ```python eval_uap.py```

### B. Evaluate a specific Model–UAP pair
From the project root directory:: ```python eval_uap.py <model_name> <uap_file>```

--- 

## Perturbation budget ***ε***
The value of ***ε*** can be modified directly in *config.py*.

Values used in the experiments:
- 4/255 ≈ 0.015686
- 8/255 ≈ 0.031373
- 16/255 ≈ 0.062745

## Results – Clean vs Adversarial (UAP)

### ε = 4/255 (0.015686)

| Model             | Clean Accuracy | ε (±ε)              | UAP Epoch | Model + UAP                          | Adversarial Accuracy | Fooling |
|-------------------|----------------|---------------------|-----------|--------------------------------------|----------------------|---------|
| My_ResNet-18      | 89.49%         | 4/255 (0.015686)    | 10        | my_model + uap_my                    | 64.04%               | 33.08%  |
|                   |                |                     |           | my_model + uap_tv                    | 66.17%               | 31.35%  |
|                   |                |                     |           | my_model + uap_densenet              | 82.23%               | 12.46%  |
| Tv_ResNet-18      | 90.37%         | 4/255 (0.015686)    | 10        | tv_model + uap_tv                    | 43.32%               | 54.40%  |
|                   |                |                     |           | tv_model + uap_my                    | 71.01%               | 26.09%  |
|                   |                |                     |           | tv_model + uap_densenet              | 82.32%               | 13.78%  |
| Densenet_Light    | 86.82%         | 4/255 (0.015686)    | 10        | densenet_light + uap_densenet        | 30.51%               | 68.38%  |
|                   |                |                     |           | densenet_light + uap_my              | 56.34%               | 40.95%  |
|                   |                |                     |           | densenet_light + uap_tv              | 57.89%               | 39.41%  |

---

### ε = 8/255 (0.031373)

| Model             | Clean Accuracy | ε (±ε)              | UAP Epoch | Model + UAP                          | Adversarial Accuracy | Fooling |
|-------------------|----------------|---------------------|-----------|--------------------------------------|----------------------|---------|
| My_ResNet-18      | 89.49%         | 8/255 (0.031373)    | 10        | my_model + uap_my                    | 19.65%               | 79.83%  |
|                   |                |                     |           | my_model + uap_tv                    | 34.54%               | 64.60%  |
|                   |                |                     |           | my_model + uap_densenet              | 47.58%               | 50.44%  |
| Tv_ResNet-18      | 90.37%         | 8/255 (0.031373)    | 10        | tv_model + uap_tv                    | 21.11%               | 74.49%  |
|                   |                |                     |           | tv_model + uap_my                    | 26.86%               | 72.65%  |
|                   |                |                     |           | tv_model + uap_densenet              | 46.14%               | 52.89%  |
| Densenet_Light    | 86.82%         | 8/255 (0.031373)    | 10        | densenet_light + uap_densenet        | 12.62%               | 86.92%  |
|                   |                |                     |           | densenet_light + uap_my              | 31.97%               | 67.07%  |
|                   |                |                     |           | densenet_light + uap_tv              | 38.95%               | 59.78%  |

---

### ε = 16/255 (0.062745)

| Model             | Clean Accuracy | ε (±ε)              | UAP Epoch | Model + UAP                          | Adversarial Accuracy | Fooling |
|-------------------|----------------|---------------------|-----------|--------------------------------------|----------------------|---------|
| My_ResNet-18      | 89.49%         | 16/255 (0.062745)   | 10        | my_model + uap_my                    | 10.16%               | 89.44%  |
|                   |                |                     |           | my_model + uap_tv                    | 10.61%               | 89.01%  |
|                   |                |                     |           | my_model + uap_densenet              | 11.85%               | 87.71%  |
| Tv_ResNet-18      | 90.37%         | 16/255 (0.062745)   | 10        | tv_model + uap_tv                    | 10.22%               | 89.56%  |
|                   |                |                     |           | tv_model + uap_my                    | 11.26%               | 88.50%  |
|                   |                |                     |           | tv_model + uap_densenet              | 13.02%               | 86.73%  |
| Densenet_Light    | 86.82%         | 16/255 (0.062745)   | 10        | densenet_light + uap_densenet        | 10.12%               | 89.92%  |
|                   |                |                     |           | densenet_light + uap_my              | 21.08%               | 78.36%  |
|                   |                |                     |           | densenet_light + uap_tv              | 16.40%               | 83.31%  |

---

## Final discussion of the results

The experimental results clearly demonstrate the effectiveness of **Universal Adversarial Perturbations (UAPs)** in degrading the performance of image classification models, both in **white-box** and **black-box** scenarios.

### Impact of the perturbation budget (ε)

As the perturbation budget ***ε*** increases (from 4/255 to 16/255), a consistent behavior is observed across all models:
- **adversarial accuracy** decreases sharply;
- the **fooling rate** increases, reaching values close to or above **85–90%**.

For *ε = 16/255*, all model–UAP combinations lead to adversarial accuracies in the range of **10–20%**, indicating that a sufficiently large universal perturbation is able to almost completely compromise the predictive capability of the models.

### White-box vs Black-box

As expected, **white-box attacks** (i.e., UAPs applied to the same model on which they were optimized) are consistently the most effective:
- for example, *my_resnet18 + uap_my* and *tv_resnet18 + uap_tv* exhibit the lowest adversarial accuracies and the highest fooling rates for every value of ***ε***.

However, the results also highlight a **remarkable transferability (black-box behavior)**:
- UAPs optimized on one model remain effective when applied to different architectures;
- in particular, UAPs generated on **DenseNet_Light** show strong transferability toward ResNet-based models, especially for *ε ≥ 8/255*.

This confirms that universal perturbations exploit vulnerabilities that are shared across models trained on the same dataset.

### Architectural comparison

Differences among architectures are more evident at lower values of ***ε***:
- for *ε = 4/255*, adversarial accuracies remain relatively high in some black-box cases, suggesting increased robustness to small perturbations;
- *DenseNet_Light*, despite having slightly lower clean accuracy, exhibits strong white-box vulnerability, with high fooling rates already at *ε = 8/255*.

Overall, none of the considered architectures proves to be truly robust against UAPs as the perturbation budget increases, confirming the severity of this type of attack.

### Conclusion

These experiments show that:
- UAPs represent a concrete and highly generalizable threat;
- model vulnerability is not limited to a single architecture;
- transferability makes UAP-based attacks particularly relevant in real-world scenarios, where the target model is unknown.

The obtained results are consistent with the existing literature on **Universal Adversarial Perturbations** and further emphasize the need for dedicated defense strategies against universal attacks.

---

## ✍️ Author
Name: Mattia Baldinetti

Course: Machine Learning Security (Cybersecurity)

University: La Sapienza University of Rome
