# Siamese-Network-for-Face-Verification
This project builds a Siamese face verification system trained with triplet loss. A shared CNN learns embeddings that pull same-person faces closer and push different identities apart. Using a Kaggle dataset and online triplet mining, the TensorFlow model is evaluated via face-pair verification using ROC and AUC.

## Key Concepts
- Siamese Neural Network  
- Triplet Loss  
- Online Triplet Mining  
- Face Embeddings  
- ROC Curve and AUC Evaluation  

---

## Dataset
The dataset contains face images organized by identity, with multiple images per person.  
Each image is preprocessed using OpenCV (resize and normalization) before being passed to the network.

---

## Model Architecture
- Shared CNN backbone for feature extraction  
- Outputs a **128-dimensional embedding** for each face  
- Optimized with **triplet loss** to minimize intra-class distance and maximize inter-class distance  

---

## Training
Training is performed using **online triplet mining**, where anchor, positive, and negative samples are generated dynamically during training.  
The model is trained in **TensorFlow 2.x** using the Adam optimizer.

All training steps, logs, and model saving are provided in the training notebook.

---

## Evaluation
Model performance is evaluated using **face pair verification**.  
Distances between embeddings are computed and used to generate a **ROC curve**, with **AUC** as the primary evaluation metric.

---

## Files Included

| File Name | Description |
|---------|------------|
| `train_siamese.ipynb` | Training notebook (preprocessing, training, model saving) |
| `siamese_model.keras` | Trained face embedding model |
| `evaluate.py` | Standalone evaluation script |
| `roc_curve.png` | ROC curve and AUC visualization |
| `README.md` | Project documentation |

---

## Tools & Libraries
- Python 3  
- TensorFlow 2.x  
- OpenCV  
- NumPy  
- scikit-learn  
- Matplotlib  

---

## How to Run

### Training
Open and run:
```
train_siamese.ipynb
```

### Evaluation
Run:
```bash
python evaluate.py
```

This will generate the ROC curve and save it as `roc_curve.png`.

---

## Output
- Trained face embedding model  
- ROC curve demonstrating face verification performance  

---

## Conclusion
This project demonstrates an end-to-end implementation of a Siamese network for face verification using triplet loss.  
The learned embeddings effectively separate genuine and impostor face pairs, making the approach scalable and suitable for real-world verification scenarios.
