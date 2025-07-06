# JWC

# Jio World Centre, Mumbai – Event Analytics Dashboard

Welcome to the open-source analytics dashboard for Jio World Centre, Mumbai!  
This dashboard uses advanced data science to extract actionable insights from visitor survey data for event strategy, marketing, and experience management.

## Features

- **Data Visualisation:** 10+ interactive charts, complex insights, and correlation heatmap
- **Classification:** Predict sponsor/exhibitor willingness using KNN, Decision Tree, Random Forest, GBRT (with all metrics, confusion matrix, ROC)
- **Clustering:** Discover customer personas with K-Means (elbow/silhouette charts, cluster download)
- **Association Rule Mining:** Find powerful associations in preferences & pain points (apriori algorithm, top 10 rules)
- **Regression:** Predict spend and satisfaction using multiple regression models, with insights
- **Filters, Upload, Download:** Flexible filtering, file upload/download for predictions
- **Branding:** Includes Jio World Centre logo and clear, color-coded metrics
- **Light/Dark Theme:** Toggle in sidebar

## How to Run

1. Clone/upload the repo to [Streamlit Cloud](https://streamlit.io/cloud) or run locally:
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
2. The app will load the dataset from `/data/JioWorldCentre_Survey_Synthetic.csv`
3. All charts/tables are interactive and downloadable.

## Data

- `/data/JioWorldCentre_Survey_Synthetic.csv` – fully synthetic but realistic convention centre visitor dataset.

## Requirements

- Python 3.9+
- See `requirements.txt` for all packages.

---

**Created by Subhayu for MBA analytics project**
