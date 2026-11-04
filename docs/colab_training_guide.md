# 🚀 Phase 5: Training Engine & Google Colab Full GPU Training Guide
## Domain-Generalized Deep Learning for Real-World Plant Disease Recognition
### Architecture: ResNet50 + MixStyle (ICLR 2021) on NVIDIA T4 GPU

---

### 🎯 Phase 5 Target Specifications Implemented

| Parameter | Specification | Purpose |
| :--- | :--- | :--- |
| **Hardware Acceleration** | `mixed_float16` precision policy | NVIDIA Tensor Cores (T4 GPU / RTX 2050 Mobile), 2x throughput, 50% VRAM |
| **Optimizer** | `AdamW` (lr=$1 \times 10^{-4}$, weight_decay=$1 \times 10^{-4}$) | Decoupled weight decay prevents overfitting on ImageNet features |
| **Loss Function** | `CategoricalCrossentropy` (Label Smoothing $\alpha=0.1$) | Prevents overconfident softmax calibration and improves generalization |
| **Class Weighting** | Inverse-frequency balanced weights ($w_c$) | Balances rare classes vs dominant classes based on Phase 2 analysis |
| **MixStyle Layer** | $p=0.5, \alpha=0.1$ (Stages 1 & 2) | Simulates unseen field domain style distributions in latent space |
| **Batch Regularization**| MixUp ($\alpha=0.2$) | Linear interpolation of image pairs and one-hot labels |
| **ModelCheckpoint** | Saves best weights to `best_model.keras` | Monitors validation loss (`val_loss`, mode='min') |
| **EarlyStopping** | Halts training if no improvement for 5 epochs | Monitors validation loss (`val_loss`, patience=5, restore_best_weights=True) |
| **ReduceLROnPlateau**| Halves learning rate on stagnation | Patience=3, factor=0.5, min_lr=$1 \times 10^{-6}$ |
| **TensorBoard** | Logs scalars & LR schedules to `logs/tensorboard/` | Real-time experiment tracking |
| **CSVLogger** | Logs epoch-by-epoch metrics | Exportable training history |

---

### ⏱️ Training Duration Comparison

| Execution Environment | Hardware | 15 Epochs Duration | Recommended? |
| :--- | :--- | :--- | :--- |
| **Google Colab (Free)** | **NVIDIA T4 GPU (15 GB VRAM)** | **~18 to 22 minutes** | 🌟 **100% Recommended** |
| Local Windows Machine | Intel CPU (No native TF GPU on Win 2.21) | ~12 to 14 hours | ⚠️ Too slow for full training |

---

### 📋 Colab Me Training Start Karne Ka Step-by-Step Tarika (Only 3 Clicks):

#### Step 1: Google Colab Kholein
1. Browser me open karein:  
   👉 **[https://colab.research.google.com/](https://colab.research.google.com/)**
2. Screen par aane wale popup me **Upload** tab par click karein.
3. **Browse** par click karke apne project ki notebook select karein:  
   📁 `D:\Desktop\Deep_Learning_Project\notebooks\train_colab.ipynb`

---

#### Step 2: Free NVIDIA T4 GPU Enable Karein
1. Colab ke top menu me jayein:  
   **Runtime ➔ Change runtime type**
2. **Hardware accelerator** dropdown me choose karein:  
   👉 **`T4 GPU`**
3. **Save** par click karein.

---

#### Step 3: "Run all" Par Click Karein
1. Top menu me jayein aur click karein:  
   **Runtime ➔ Run all** *(ya shortcut keyboard par: `Ctrl + F9`)*
2. Colab autonomously sab kuch execute karega:
   - ✅ NVIDIA T4 GPU detect karke `mixed_float16` Tensor Core acceleration activate karega.
   - ✅ PlantVillage dataset ko high-speed network se Colab me load karega.
   - ✅ 28-class canonical taxonomy aur 80/20 train/validation split setup karega.
   - ✅ ResNet50 + MixStyle architecture compile karega.
   - ✅ Balanced class weights ke sath **15 full epochs** train karega.
   - ✅ Best validation loss par checkpoint save karega.
   - ✅ Convergence loss/accuracy curves plot karega.
   - ✅ Aur automatic browser download trigger karega for **`best_model.keras`**!

---

#### Step 4: Trained Model Ko Project Me Rakhein
- Training complete hote hi aapke browser me **`best_model.keras`** (~296 MB) download ho jayegi.
- Use cut karke apne project ke `models` folder me paste kar dein:  
  📁 `D:\Desktop\Deep_Learning_Project\models\best_model.keras`

---

### 🏁 Local Web App Test Karein
Model weights replace hone ke baad Streamlit app run karein:
```bash
streamlit run app/app.py
```
App 98% diagnostic accuracy aur live Grad-CAM heatmaps ke sath live field diagnosis perform karegi!
