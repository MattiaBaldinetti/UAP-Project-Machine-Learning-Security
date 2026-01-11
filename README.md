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
Posizionarsi nella cartella principale del progetto: ```python run_clean_experiment.py```

### B. Allenamento di un singolo modello
Posizionarsi nella cartella principale del progetto: ```python run_clean_experiment.py --model <nome_modello>```

## 2. Caricamento e congelamento dei modelli
Ricarica i checkpoint migliori e congela i modelli (`modalità eval` e `requires_grad=False`) per l’uso nelle UAP.

### A. Per tutti i modelli: 
Posizionarsi nella cartella principale del progetto: ```python load_models.py```

### B. Per un modello specifico: 
Posizionarsi nella cartella principale del progetto: ```python load_models.py --model <nome_modello>```

## 3. Calcolo della Universal Adversarial Perturbation (UAP)
La UAP viene calcolata in pixel-space e salvata nella cartella *uaps/*.

Il budget della perturbazione ***ε*** è definito in config.py tramite la variabile `EPS_PIX`.

### A. Calcolo UAP per un singolo modello
Posizionarsi nella cartella principale del progetto: ```python run_save_uap.py <model_name>```

### B. Calcolo UAP per tutti i modelli
Posizionarsi nella cartella principale del progetto: ```python run_save_uap.py```

## 4. Caricamento e visualizzazione della UAP
Visualizza:
- la perturbazione universale ***δ***
- un esempio di immagine originale vs perturbata
- le immagini vengono salvate in *uap_img/*.

### A. Visualizzare una UAP specifica
Posizionarsi nella cartella principale del progetto: ```python uap_load_view.py <nome_file>```

### B. Visualizzare tutte le UAP salvate
Posizionarsi nella cartella principale del progetto: ```python uap_load_view.py```

## 5. Valutazione adversarial (clean vs UAP)
Vengono calcolate:
- Accuracy clean
- Accuracy adversarial
- Fooling rate

### A. Tutti le possibilità Modello × UAP
Posizionarsi nella cartella principale del progetto: ```python eval_uap.py```

### B. Valutazione di una coppia specifica
Posizionarsi nella cartella principale del progetto: ```python eval_uap.py <model_name> <uap_file>```

--- 

## Modifica del budget di perturbazione ***ε***
Il valore di ε può essere modificato direttamente in config.py.

Valori utilizzati negli esperimenti:
- 4/255 ≈ 0.015686
- 8/255 ≈ 0.031373
- 16/255 ≈ 0.062745

## Risultati – Clean vs Adversarial (UAP)

### ε = 4/255 (0.015686)

| Modello           | Accuracy Clean | ε (±ε)              | UAP Epoch | Model + UAP                          | Accuracy Adversarial | Fooling |
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

| Modello           | Accuracy Clean | ε (±ε)              | UAP Epoch | Model + UAP                          | Accuracy Adversarial | Fooling |
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

| Modello           | Accuracy Clean | ε (±ε)              | UAP Epoch | Model + UAP                          | Accuracy Adversarial | Fooling |
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

## Commento finale sui risultati

I risultati sperimentali mostrano in modo chiaro l’efficacia delle **Universal Adversarial Perturbations (UAP)** nel degradare le prestazioni dei modelli di classificazione, sia in **white-box** sia in **black-box**.

### Impatto del budget di perturbazione (ε)

All’aumentare del budget di perturbazione ***ε*** (da 4/255 a 16/255) si osserva un comportamento coerente su tutti i modelli:
- l’**accuracy adversarial** diminuisce drasticamente;
- il **fooling rate** cresce fino a valori prossimi o superiori all’85–90%.

Per *ε = 16/255*, tutte le combinazioni modello–UAP portano l’accuracy adversarial intorno al **10–20%**, indicando che una perturbazione universale sufficientemente ampia è in grado di compromettere quasi completamente la capacità predittiva dei modelli.

### White-box vs Black-box

Come atteso, l’attacco **white-box** (UAP applicata allo stesso modello su cui è stata ottimizzata) risulta sempre il più efficace:
- ad esempio, *my_resnet18 + uap_my* e *tv_resnet18 + uap_tv* mostrano i valori di accuracy adversarial più bassi e i fooling rate più elevati per ogni ***ε***.

Tuttavia, i risultati evidenziano anche una **notevole trasferibilità (black-box)**:
- le UAP ottimizzate su un modello risultano efficaci anche su architetture diverse;
- in particolare, le UAP generate su **DenseNet_Light** mostrano una buona capacità di trasferimento verso le ResNet, soprattutto per *ε ≥ 8/255*.

Questo conferma che le perturbazioni universali catturano vulnerabilità condivise tra modelli addestrati sullo stesso dataset.

### Confronto tra architetture

Le differenze tra le architetture emergono soprattutto a basso ***ε***:
- per *ε = 4/255*, le accuracy adversarial rimangono relativamente alte in alcuni casi black-box, indicando una maggiore robustezza a perturbazioni più piccole;
- *DenseNet_Light*, pur avendo una clean accuracy leggermente inferiore, mostra una forte vulnerabilità in white-box, con fooling rate elevati già per *ε = 8/255*.

Nel complesso, nessuna architettura risulta realmente robusta alle UAP quando il budget di perturbazione cresce, a conferma della pericolosità di questo tipo di attacco.

### Conclusione

Questi esperimenti dimostrano che:
- le UAP rappresentano una minaccia concreta e generalizzabile;
- la vulnerabilità non è limitata a una singola architettura;
- la trasferibilità rende l’attacco particolarmente rilevante in scenari reali, dove il modello target non è noto.

I risultati ottenuti sono coerenti con la letteratura sulle Universal Adversarial Perturbations e confermano la necessità di studiare strategie di difesa specifiche contro attacchi universali.

---

## ✍️ Author
Name: Mattia Baldinetti

Course: Machine Learning Security (Cybersecurity)

University: La Sapienza University of Rome
