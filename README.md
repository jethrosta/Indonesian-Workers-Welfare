# 🇮🇩 Indonesian Workers Welfare

This project applies Machine Learning techniques, specifically clustering and classification, to analyze and predict the welfare status of Indonesian workers based on key socioeconomic features. By leveraging publicly available datasets, the goal is to uncover patterns, segment worker populations, and build predictive models to support data-driven policy-making.

## 📋 Data Source

The dataset is sourced from a public Kaggle resource related to Indonesian workers’ welfare, featuring regional spending patterns and minimum wage indicators.<br>
[You can find the dataset here](https://www.kaggle.com/datasets/wowevan/dataset-kesejahteraan-pekerja-indonesia/code)

## 📁 Project Structure
- dataset/ – Raw and processed datasets (e.g., worker demographics, income, sector type, social protection).
- Clustering notebooks/ – Jupyter notebooks for preprocessing, clustering, model training, and clustering evaluation.
- Classification notebooks/ – Jupyter notebooks for preprocessing, classification, model training, and classification evaluation.

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

## Exploratory Data Analysis
### 📄 rataRataUpah.csv – Average Hourly Wage
<div align="center">
  
| Column Name | Description                  |
| ----------- | ---------------------------- |
| `Provinsi`  | Indonesian province name     |
| `Tahun`     | Year of data                 |
| `Upah_Jam`  | Average hourly wage (in IDR) |

</div>

### 📄 pengeluaran.csv – Per Capita Expenditure
<div align="center">
  
| Column Name         | Description                                    |
| ------------------- | ---------------------------------------------- |
| `Provinsi`          | Indonesian province                            |
| `Tahun`             | Year of data                                   |
| `Jenis_Pengeluaran` | Category: “Makanan”, “Non-makanan”, or “Total” |
| `Per_Kapita`        | Per capita monthly expenditure (in IDR)        |
| `Daerah`            | Region type (“Perkotaan” or “Perdesaan”)       |
</div>

### 📄 upahPengeluaran_Merged.csv (merged dataset)
Combines wage and expenditure data for clustering/classification:
<div align="center">
  
| Column Name              | Description                             |
| ------------------------ | --------------------------------------- |
| `Provinsi`               | Province name                           |
| `Tahun`                  | Year                                    |
| `Upah_Rata_rata`         | Average hourly wage (IDR)               |
| `Pengeluaran_Makanan`    | Per capita food expenditure             |
| `Pengeluaran_NonMakanan` | Per capita non-food expenditure         |
| `Pengeluaran_Total`      | Total per capita expenditure            |
| `Daerah`                 | Region classification (“Urban”/“Rural”) |
</div>

If we dive in into this branch of data we can find that:

<table align="center">
  <tr>
    <td><img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/GarisKemiskinanProvinsi.png" width="100%"></td>
    <td><img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/UpahRatarataProvini.png" width="100%"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/PengeluaranProvinsi.png" width="100%"></td>
    <td><img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/UMPProvinsi.png" width="100%"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/LivingPlaceDistribution.png" width="100%"></td>
    <td><img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/BoxplotDataNumerik.png" width="100%"></td>
  </tr>
</table>

The visualizations offer a multidimensional view of the welfare conditions of Indonesian workers, focusing on regional disparities in income, expenditure, and poverty indicators across provinces.

The first image (top-left) shows the Provincial Poverty Line across Indonesia. It is clear that there is a notable variation in the poverty threshold among provinces, with urbanized or more developed regions generally having a higher poverty line. This reflects the differing costs of living across Indonesia, where regions such as Jakarta and Papua tend to require higher minimum resources to escape poverty.

The second image (top-right) presents the Average Wage by Province. It shows that certain provinces benefit from significantly higher wages, which likely correlates with industrial concentration and economic development. However, some provinces show average wages that are worryingly close to, or even lower than, their respective poverty lines, indicating economic vulnerability.

The third image (middle-left) illustrates the Average Per Capita Expenditure by Province. This chart highlights the disparities in consumption patterns among Indonesian citizens. Provinces with higher expenditures often coincide with those with higher incomes and urbanization levels, but some regions show high spending relative to low income, potentially indicating financial strain or reliance on informal income sources.

The fourth image (middle-right) displays the Provincial Minimum Wage (UMP). While the UMP is intended as a protective floor for labor welfare, in several provinces it remains below the average expenditure or even near the poverty line. This suggests that the minimum wage in these areas may not be sufficient to sustain a decent standard of living.

The fifth image (bottom-left) shows the Distribution of Workers by Type of Residence. This graphic reveals a skewed concentration, likely indicating that the majority of the workforce is located in urban or semi-urban areas, aligning with industrial and service-based job availability. This spatial distribution is important for targeted policy interventions.

Finally, the sixth image (bottom-right) presents a Boxplot of Numerical Data used in the study. It provides a statistical overview of the range and distribution of key numeric variables such as wages, expenditure, and poverty thresholds. The presence of outliers and interquartile spreads highlights the existence of inequality and regional variation in labor welfare indicators.

In summary, the visualizations collectively illustrate a concerning mismatch in several provinces between income levels (both average and minimum), living costs, and poverty standards. They underline the importance of region-specific policy adjustments and wage reforms to better align worker income with living costs and ensure more equitable economic welfare across Indonesia.

## Clustering Process
This project explores the grouping of Indonesian provinces or regions based on three key features:
- UMP (Minimum Provincial Wage)
- Expenditure per Capita
- Average Wage
The objective is to understand how these socioeconomic factors form natural groupings that reflect welfare and economic disparity, particularly in relation to the poverty line.

### 🧪 Methodology
- Feature Selection:<br>
  Selected three primary indicators—UMP, Pengeluaran, and Upah_Rata_rata—as key socioeconomic variables related to the poverty line.
- Preprocessing:
  - Features were scaled using StandardScaler to normalize the values.
  - Handled missing or inconsistent data as needed.
- Elbow Method to Determine Optimal k:<br>
  Using distortion score (inertia), the elbow point was observed at k = 3, suggesting three optimal clusters.
- K-Means Clustering:
  - Applied KMeans with k=3
  - Visualized in 3D to represent how the clusters separate in feature space.
<div align="center">
<img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/ElbowMethodKMeansClustering.png">
</div>

<div align="center">
<img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/ClusteringUMPvsPengeluaran.png">
<img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/Clustering_3Fitur.png">
</div>

### 🔍 Supporting Analysis: Correlation Matrix
- Positive correlations between UMP, Pengeluaran, and Upah_Rata_rata confirm that these variables move in the same direction.
- Strong negative correlation between poverty line and wages confirms the relevance of these features for understanding welfare tiers.

<div align="center">
<img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/CorrelationMatrix.png">
</div>

### 🧩 Insights
The correlation matrix presents a heatmap of relationships between key socioeconomic features such as provincial minimum wage (UMP), average wage, poverty line, and expenditure types. It reveals a strong positive correlation between:
- Minimum Wage (UMP) and Average Wage
- Expenditure (total, food, and non-food) and Average Wage / UMP
This indicates that as wages increase, individuals tend to spend more, an expected economic pattern in welfare data. Conversely, negative correlations are observed between poverty indicators and income variables, which aligns with the assumption that higher wages reduce poverty risk.

In addition, the type of area (urban vs. rural) shows a moderate correlation with expenditure, reinforcing the idea that urban populations generally exhibit higher spending patterns and income levels compared to rural areas.

### 📌 Resulting Clusters
1. Cluster 1:
  - Average Provincial Minimum Wage (Rp): 1,796,418
  - Average Per Capita Expenditure (Rp): 516,870
  - Analysis: This cluster includes Indonesian workers with low average monthly income and low average expenditure. Workers in this cluster tend to have a frugal lifestyle due to the low salary amount.
2. Cluster 2:
  - Average Provincial Minimum Wage (Rp): 2,718,950
  - Average Per Capita Expenditure (Rp): 634,661
  - Analysis: This cluster includes Indonesian workers with a high average monthly income but low average expenditure. Workers in this cluster tend to have a frugal lifestyle despite having a high salary.
3. Cluster 3:
  - Average Provincial Minimum Wage (Rp): 2,289,149
  - Average Per Capita Expenditure (Rp): 1,091,833
  - Analysis: This cluster includes Indonesian workers with an average monthly income that is neither too high nor too low, but with an average expenditure amount that is nearly half of their salary. Workers in this cluster tend to have a high lifestyle, balanced by a relatively high salary.

<div align="center">
<img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/ClusteringDenganCentroid.png">
</div>

### 🤖 Classification Process
Next, the classification process was conducted by first creating a dataset from the clustering results, then splitting it into two parts: training data and testing data. The training dataset consisted of 3,718 samples, while the testing dataset contained 930 samples.

<div align="center">
  <img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/ConfMat_best_DT.png">
</div>

Following that, the classification model building phase was performed by selecting suitable classification algorithms such as Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN). The accuracy results for each classification model are summarized in the following table:
<div align="center">
  
| Algorithm           | Accuracy | F1-Score | Precision | Recall |
| ------------------- | -------- | -------- | --------- | ------ |
| Logistic Regression | 0.9796   | 0.9796   | 0.9797    | 0.9796 |
| Decision Tree       | 0.9935   | 0.9935   | 0.9936    | 0.9935 |
| Random Forest       | 0.9892   | 0.9892   | 0.9893    | 0.9892 |
| K-Nearest Neighbors | 0.9086   | 0.9081   | 0.9089    | 0.9086 |

</div>
Among all the available models, the Decision Tree produced the best results across evaluation metrics from Accuracy to Recall. Therefore, I chose to use the Decision Tree model.

To optimize the model, I applied hyperparameter tuning methods such as GridSearchCV and RandomizedSearchCV to find the best combination of hyperparameters. The best hyperparameters found were:
- `max_depth`: 20
- `min_samples_leaf`: 1
- `min_samples_split`: 2

The evaluation metrics with these hyperparameters are as follows:
<div align="center">
  
| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.9957 |
| F1-Score  | 0.9957 |
| Precision | 0.9957 |
| Recall    | 0.9957 |

<div align="center">
  <img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/learning_curve_overfit.png">
</div>

</div>

However, upon reviewing the learning curve graphs, it was observed that the model suffered from overfitting. Therefore, I searched for more generalized hyperparameters and obtained the following best parameters:
- max_depth: 10
- min_samples_leaf: 1
- min_samples_split: 2

<div align="center">
  <img src="https://github.com/jethrosta/Indonesian-Workers-Welfare/blob/main/images/learningcurve_best_dt.png">
</div>

### Insights and Recommendations:

Before tuning, the Decision Tree model yielded:
<div align="center">
  
| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.9935 |
| F1-Score  | 0.9935 |
| Precision | 0.9936 |
| Recall    | 0.9935 |

</div>

After hyperparameter tuning with the parameters (max_depth: 20, min_samples_leaf: 1, min_samples_split: 2), the results improved to:
<div align="center">
  
| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.9957 |
| F1-Score  | 0.9957 |
| Precision | 0.9957 |
| Recall    | 0.9957 |

</div>

Model weaknesses were identified, such as:
- Initial model parameters caused overfitting, leading to predictions skewed towards class 1 on the training data.
- After tuning, the model improved and became more generalizable, although some gap remained between the training and testing scores on the learning curve, it was narrower than before.
Recommendations for further actions include:
- Exploring other hyperparameters using methods like RandomizedSearchCV.
- Considering switching to other algorithms such as Random Forest for training.

## 📊 Results
1. Identified meaningful clusters of workers based on socioeconomic variables.
2. Developed clustering models with strong accuracy in knowing how many cluster based on selected features.
3. Developed classification models with strong accuracy in predicting welfare category.
4. Extracted key feature importances to guide policy interventions.
