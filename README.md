# 🇮🇩 Indonesian Workers Welfare

This project applies Machine Learning techniques, specifically clustering and classification, to analyze and predict the welfare status of Indonesian workers based on key socioeconomic features. By leveraging publicly available datasets, the goal is to uncover patterns, segment worker populations, and build predictive models to support data-driven policy-making.

## 📋 Data Source

The dataset is sourced from a public Kaggle resource related to Indonesian workers’ welfare, featuring regional spending patterns and minimum wage indicators.<br>
[You can find the dataset here](https://www.kaggle.com/datasets/wowevan/dataset-kesejahteraan-pekerja-indonesia/code)

## 📁 Project Structure
- dataset/ – Raw and processed datasets (e.g., worker demographics, income, sector type, social protection).
- Clustering notebooks/ – Jupyter notebooks for preprocessing, clustering, model training, and clustering evaluation.
- Classification notebooks/ – Jupyter notebooks for preprocessing, clustering, model training, and classification evaluation.

## 🎯 Objectives
1. Clustering (Unsupervised Learning)
Group workers based on similar characteristics (e.g., income level, sector, social security access) using algorithms like K-Means.
2. Classification (Supervised Learning)
Predict a worker’s welfare category (e.g., “At Risk” vs. “Secure”) using models like Decision Tree, Random Forest, or Logistic Regression.
3. Policy-Oriented Insight
Translate ML outcomes into actionable insights for public policy and labor regulation improvement.

## 🔍 Features & Models
- Features: Age, gender, sector, region, wage, BPJS (social security) status, working hours.
- Clustering: K‑Means, DBSCAN, with Silhouette analysis.
- Classification: Decision Tree, Random Forest, Logistic Regression.
- Evaluation: Precision, recall, F1-score, confusion matrix, ROC-AUC.
- Visualization: Matplotlib, Seaborn for insightful data and model interpretations.

## Clustering Process
<img src="">

## 📊 Example Results
1. Identified meaningful clusters of workers based on socioeconomic variables.
2. Developed classification models with strong accuracy in predicting welfare categories.
3. Extracted key feature importances to guide policy interventions.
