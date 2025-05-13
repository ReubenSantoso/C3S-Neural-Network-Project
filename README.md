# CCS Prediction with Graph Neural Networks

This repository contains training code and experiments for predicting ion-neutral collision cross sections (CCS) of small molecules using Graph Neural Networks (GNNs). It extends and improves upon the models described in Ross *et al.*, *Analytical Chemistry* 2020:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

---

## 🚀 Features

- **Data preprocessing**:  
  - Load experimentally measured CCS values from IM–MS  
  - Compute graph features (atom/bond connectivity) with RDKit using molecular fingerprints  
  - Generate Molecular Quantum Numbers (MQNs) and mass/adduct descriptors  

- **Model zoo**:  
  - **VAE + MLP Regressor** (baseline — currently not implemented)  
  - **Node-level GNNs**  
    - GCN (mean pooling) + Dropout  
    - GraphSAGE + L2 regularization  
    - GAT with attention + L2
  - **Graph-level GNNs**  
    - Custom “C3SRegression” GCN  
    - GIN merged with molecular fingerprints + MLP  
  - **Other Models**  
    - Random Forest (clean vs. unclean data; with/without cyclic peptides)  

- **Training scripts**  
  - Configurable using pytorch and pytorch geometric

- **Key findings**  
  - **Dropout** harmed generalization in our GNNs (node-level)  
  - **Batch normalization** consistently improved convergence and test error  
