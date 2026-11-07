# Kinase Inhibitor QSAR Machine Learning Pipeline

> A lightweight, end-to-end cheminformatics and machine learning pipeline designed to predict molecular potency (pIC50) using quantitative structure-activity relationship (QSAR) modeling.

---

## 🚀 Project Overview
In drug discovery, screening molecular potency early in the pipeline is critical for filtering candidate compounds. This project implements a machine learning workflow that:
1. Ingests molecular structures represented as **SMILES strings**.
2. Engineers structural features using **RDKit** (converting molecules into binary Morgan fingerprints).
3. Trains a supervised regression model (**Random Forest**) via **Scikit-Learn** to predict biological activity targets ($\text{pIC}_{50}$).
4. Evaluates model generalizability using metrics like $R^2$ Score and Root Mean Square Error (RMSE).

---

## 🛠️ Tech Stack & Requirements
* **Language:** Python 3.10+
* **Cheminformatics:** `rdkit` (via Conda Forge)
* **Machine Learning:** `scikit-learn`
* **Data Manipulation:** `pandas`, `numpy`
* **Model Serialization:** `joblib`

---

## 📂 Project Structure
```text
qsar-kinase-project/
│
├── data/                  # Contains raw/processed CSV datasets (train/test)
├── models/                # Serialized model checkpoints (.pkl files)
├── src/                   # Source code directory
│   ├── preprocess.py      # SMILES parsing & Morgan fingerprint generation
│   ├── train.py           # Model training and artifact serialization loop
│   └── evaluate.py        # Model evaluation metrics calculation
├── .gitignore             # Git exclusion rules (ignores data and model weights)
└── README.md              # Project documentation
