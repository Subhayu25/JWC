import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from pathlib import Path
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings("ignore")

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(
    page_title="Jio World Centre Analytics Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

# --- DISPLAY LOGO AT TOP (Local or from GitHub raw) ---
logo_path = "data/logo_JWC.png"
github_logo_url = "https://github.com/Subhayu25/JWC/blob/8f4f29b0abc9b332e00072750aff88470ea8a4cf/logo_jwc.png"

try:
    if logo_path.exists():
    logo_img = Image.open(logo_path)
    # adjust width to taste
    st.image(logo_img, width=200)
    else:
    st.warning(f"⚠️ Logo not found at {logo_path}")

# --- LOAD DATA LOCALLY OR FROM GITHUB RAW ---
@st.cache_data
def load_data():
    local_path = "data/JioWorldCentre_Survey_Synthetic.csv"
    github_csv_url = "https://raw.githubusercontent.com/Subhayu25/JWC/refs/heads/main/JioWorldCentre_Survey_Synthetic.csv"
    try:
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        else:
            return pd.read_csv(github_csv_url)
    except Exception as e:
        st.error("CSV file not found locally or on GitHub. Please check file location or repo.")
        st.stop()

df = load_data()

# --- SIDEBAR THEME SELECTION ---
st.sidebar.title("Settings")
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body{background:#121212;color:#ECEFF1;}</style>", unsafe_allow_html=True)

# --- EXECUTIVE SUMMARY ---
st.markdown("# Jio World Centre, Mumbai: Consumer Insights Dashboard")
st.markdown("""
<div style='background:#003366; padding:20px; border-radius:10px;'>
  <h2 style='color:#FFFFFF; margin:0;'>Executive Summary</h2>
  <p style='color:#FFD700; font-size:16px;'>
    This interactive dashboard uses a realistic synthetic survey dataset to uncover visitor personas,
    spending patterns, satisfaction drivers, clustering segments, classification of sponsor interest,
    regression insights, and association rules at Jio World Centre, Mumbai.
  </p>
</div>
""", unsafe_allow_html=True)

# --- MAIN TABS ---
tabs = st.tabs([
    "Data Visualisation",
    "Classification",
    "Clustering",
    "Association Rule Mining",
    "Regression",
    "Recommendations"
])

# ========== TAB 1: DATA VISUALISATION ==========
with tabs[0]:
    st.markdown("## :bar_chart: Data Visualisation & Insights")
    with st.expander("Filters"):
        gender = st.multiselect("Gender", df.Gender.unique(), df.Gender.unique())
        city = st.multiselect("City", df.City.unique(), df.City.unique())
        reason = st.multiselect("Visit Reason", df.VisitReason.unique(), df.VisitReason.unique())
        min_age, max_age = int(df.Age.min()), int(df.Age.max())
        age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age))
        df_viz = df[
            df.Gender.isin(gender) &
            df.City.isin(city) &
            df.VisitReason.isin(reason) &
            df.Age.between(age_range[0], age_range[1])
        ]
        st.caption(f"**{len(df_viz)} responses filtered**")
        st.markdown("*Use these filters to narrow down the segment of attendees based on demographic and event preferences.*")

    # Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Median Spend/Visit (INR)", f"{int(df_viz.SpendPerVisitINR.median()):,}")
    with c2:
        st.metric("Top Event Type", df_viz.VisitReason.mode()[0])
    with c3:
        st.metric("Top City", df_viz.City.mode()[0])
    st.markdown("*Key summary metrics give you a quick pulse on spending, popular event types, and top cities.*")

    # 1. Age Distribution
    st.markdown("### 1. Age Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df_viz.Age, bins=20, kde=True, ax=ax1, color="#4C72B0")
    ax1.set_xlabel("Age"); ax1.set_ylabel("Count")
    st.pyplot(fig1)
    st.markdown("""
    **Interpretation:**  
    - The majority of attendees are between 25–45 years old.  
    - Peak around age 35 suggests marketing focus on this age group.
    """)

    # 2. Spend per Visit by City & Event Type
    st.markdown("### 2. Spend per Visit by City & Event Type")
    fig2 = px.box(df_viz, x="City", y="SpendPerVisitINR", color="VisitReason", points="all")
    fig2.update_layout(xaxis_title="City", yaxis_title="Spend per Visit (INR)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    - Mumbai and Delhi show wider spend ranges and high-value outliers.  
    - Conferences drive higher median spend than other event types.
    """)

    # 3. Occupation vs. Networking Interest
    st.markdown("### 3. Occupation vs. Networking Interest")
    occ_net = pd.crosstab(df_viz.Occupation, df_viz.NetworkingInterest)
    fig3 = px.bar(occ_net, barmode='group', labels={'value':'Count'})
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    - Corporate attendees show highest networking interest.  
    - Students and artists less so, guiding targeted promotions.
    """)

    # 4. Attendance Frequency
    st.markdown("### 4. Event Attendance Frequency")
    fig4 = px.pie(df_viz, names="ParticipationFrequency", title="Attendance Frequency")
    fig4.update_traces(textinfo='percent+label')
    st.plotly_chart(fig4)
    st.markdown("""
    **Interpretation:**  
    - ~40% attend annually, indicating many first-time visitors.  
    - Quarterly attendees (~25%) represent a loyal segment.
    """)

    # 5. Food Preferences
    st.markdown("### 5. Food Preferences")
    food_counts = df_viz.FoodPreferences.str.get_dummies(sep=',').sum().sort_values(ascending=False)
    fig5 = px.bar(food_counts, labels={'value':'Count','index':'Food'})
    fig5.update_layout(xaxis_title="Food Type", yaxis_title="Count")
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    - Indian cuisine dominates, followed by Continental and Asian.  
    - Vegan/gluten-free have ~15–20% demand.
    """)

    # 6. Premium Seating vs. Spend
    st.markdown("### 6. Premium Seating Interest vs. Spend")
    fig6 = px.box(df_viz, x="PremiumSeatingInterest", y="SpendPerVisitINR", color="PremiumSeatingInterest")
    fig6.update_layout(xaxis_title="Premium Seating Interest", yaxis_title="Spend per Visit (INR)")
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    - “Yes” respondents spend more, confirming upsell potential.  
    - “Maybe” group may convert with targeted offers.
    """)

    # 7. Challenges Faced
    st.markdown("### 7. Challenges Faced at Events")
    chall_counts = df_viz.ChallengesFaced.str.get_dummies(sep=',').sum().sort_values(ascending=False)
    fig7 = px.bar(chall_counts, labels={'value':'Count','index':'Challenge'})
    fig7.update_layout(xaxis_title="Challenge", yaxis_title="Count")
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    - Parking and signage top complaints—immediate fixes.  
    - Food quality and tech issues also notable.
    """)

    # 8. Satisfaction by City
    st.markdown("### 8. Overall Satisfaction by City")
    fig8 = px.box(df_viz, x="City", y="OverallSatisfaction", color="City")
    fig8.update_layout(xaxis_title="City", yaxis_title="Satisfaction (1–10)")
    st.plotly_chart(fig8, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    - Pune and Mumbai show tight high satisfaction.  
    - Delhi’s wider spread indicates varied experiences.
    """)

    # 9. Correlation Heatmap
    st.markdown("### 9. Correlation Heatmap")
    corr_cols = [
        "Age","MonthlyIncomeINR","EventsAttendedAnnually",
        "SpendPerVisitINR","MaxFoodSpendINR","RecommendJioWorldCentre","OverallSatisfaction"
    ]
    fig9, ax9 = plt.subplots(figsize=(7,5))
    sns.heatmap(df_viz[corr_cols].corr(), annot=True, cmap="vlag", ax=ax9)
    ax9.set_title("Feature Correlations")
    st.pyplot(fig9)
    st.markdown("""
    **Interpretation:**  
    - Income vs. spend ~0.45 correlation.  
    - Satisfaction vs. recommendation ~0.72 correlation.
    """)

    # 10. Download Filtered Data
    st.markdown("### 10. Download Filtered Data")
    csv_data = df_viz.to_csv(index=False).encode()
    st.download_button("Download CSV", csv_data, "filtered_data.csv", "text/csv")
    st.markdown("*Download the filtered dataset for offline analysis.*")


# ========== TAB 2: CLASSIFICATION ==========
with tabs[1]:
    st.markdown("## :guardsman: Classification - Sponsor/Exhibitor Willingness")
    st.info("Target: SponsorOrExhibitorInterest == 'Yes'")

    clf_df = df.copy()
    clf_df['Sponsor_Label'] = (clf_df.SponsorOrExhibitorInterest == 'Yes').astype(int)
    features = [
        "Age","MonthlyIncomeINR","EventsAttendedAnnually",
        "SpendPerVisitINR","MaxFoodSpendINR","OverallSatisfaction"
    ]
    cat_cols = ['Gender','Education','Occupation','PreferredEventType']
    for col in cat_cols:
        le = LabelEncoder()
        clf_df[col] = le.fit_transform(clf_df[col])
        features.append(col)

    X = clf_df[features]
    y = clf_df['Sponsor_Label']
    Xtr, Xts, ytr, yts = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=0),
        'Random Forest': RandomForestClassifier(random_state=0),
        'Gradient Boosting': GradientBoostingClassifier(random_state=0)
    }
    results = []
    roc_data = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        preds = m.predict(Xts)
        acc = accuracy_score(yts, preds)
        prec = precision_score(yts, preds)
        rec = recall_score(yts, preds)
        f1 = f1_score(yts, preds)
        results.append([name, acc, prec, rec, f1])
        # ROC
        if hasattr(m, "predict_proba"):
            y_score = m.predict_proba(Xts)[:,1]
        else:
            y_score = m.decision_function(Xts)
        fpr, tpr, _ = roc_curve(yts, y_score)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))

    metrics_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1-Score"])
    st.dataframe(metrics_df.style.background_gradient(cmap="Blues"), use_container_width=True)
    st.markdown("**Interpretation:** Compare these to pick the best classifier.")

    # Feature importances via permutation for all classifiers
    st.markdown("### Feature Importances")
    sel_name = st.selectbox("Choose model", list(models.keys()), key="feature_importance_selector")
    sel_model = models[sel_name]
    with st.spinner(f"Computing permutation importances for {sel_name}..."):
        perm = permutation_importance(
            sel_model, Xts, yts,
            n_repeats=10, random_state=42, n_jobs=-1
        )
    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": perm.importances_mean
    }).sort_values("Importance", ascending=False)
    fig_imp, ax_imp = plt.subplots()
    sns.barplot(
        y="Feature", x="Importance",
        data=imp_df, ax=ax_imp, palette="viridis"
    )
    ax_imp.set_title(f"{sel_name} Permutation Importances")
    st.pyplot(fig_imp)
    st.markdown("""
    **Interpretation:**  
    Permutation importance measures how shuffling each feature affects model performance.  
    Larger drops mean the feature is more critical to accurate predictions.
    """)

    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    sel_cm = st.selectbox("Model (Confusion Matrix)", list(models.keys()), key="conf_matrix_selector")
    cm = confusion_matrix(yts, models[sel_cm].predict(Xts))
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=["No","Yes"], yticklabels=["No","Yes"],
        ax=ax_cm
    )
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
    st.markdown("**Interpretation:** TP/TN and FP/FN counts help assess model errors.")

    # ROC Curves
    st.markdown("#### ROC Curves")
    fig_roc, ax_roc = plt.subplots()
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    ax_roc.plot([0,1],[0,1],'k--')
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)
    st.markdown("**Interpretation:** Curves closer to the top-left indicate stronger models.")

    # Upload new data
    st.markdown("#### Predict on New Data")
    up = st.file_uploader("Upload CSV (no target)", type="csv")
    if up:
        new_df = pd.read_csv(up)
        for col in cat_cols:
            le = LabelEncoder()
            new_df[col] = le.fit_transform(new_df[col].astype(str))
        new_df["Predicted_Sponsor"] = sel_model.predict(new_df[features])
        new_df["Predicted_Sponsor"] = new_df["Predicted_Sponsor"].map({1:"Yes",0:"No"})
        st.dataframe(new_df)
        csv_out = new_df.to_csv(index=False).encode()
        st.download_button("Download Predictions", csv_out, "predictions.csv", "text/csv")
        st.markdown("**Interpretation:** Use this to score new leads for sponsorship interest.")

# ========== TAB 3: CLUSTERING ==========
with tabs[2]:
    st.markdown("## :busts_in_silhouette: Clustering - Customer Personas")
    cdf = df.copy()
    num_feats = [
        "Age","MonthlyIncomeINR","EventsAttendedAnnually",
        "SpendPerVisitINR","MaxFoodSpendINR","OverallSatisfaction"
    ]
    scaler = StandardScaler()
    Xc = scaler.fit_transform(cdf[num_feats])

    k = st.slider("Number of Clusters (k)", 2, 10, 3)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cdf["Cluster"] = km.fit_predict(Xc)

    inertias = []; sils = []
    for i in range(2,11):
        km2 = KMeans(n_clusters=i, random_state=42, n_init=10)
        labs = km2.fit_predict(Xc)
        inertias.append(km2.inertia_)
        sils.append(silhouette_score(Xc, labs))

    fig_elb = px.line(x=list(range(2,11)), y=inertias, markers=True, title="Elbow Method")
    st.plotly_chart(fig_elb)
    st.markdown("**Interpretation:** The 'elbow' point suggests an optimal k where adding more clusters yields diminishing returns.")

    fig_sil = px.line(x=list(range(2,11)), y=sils, markers=True, title="Silhouette Scores")
    st.plotly_chart(fig_sil)
    st.markdown("**Interpretation:** Higher silhouette scores indicate well-separated, dense clusters.")

    st.markdown("#### Cluster Personas")
    persona = cdf.groupby("Cluster")[num_feats].mean().round(1)
    st.dataframe(persona, use_container_width=True)
    st.markdown("**Interpretation:** Each row is the average profile for that cluster, guiding personalized marketing.")  

    # Download
    dl_csv = cdf.to_csv(index=False).encode()
    st.download_button("Download Clustered Data", dl_csv, "clustered_data.csv", "text/csv")

# ========== TAB 4: ASSOCIATION RULE MINING ==========
with tabs[3]:
    st.markdown("## :bulb: Association Rule Mining")
    cols = ['FoodPreferences','ChallengesFaced']
    selc = st.selectbox("Select column", cols)
    ms = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
    mc = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
    td = df[selc].str.get_dummies(sep=',')
    freq = apriori(td, min_support=ms, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=mc).sort_values("confidence", ascending=False).head(10)
    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])
    st.markdown("""
    **Interpretation:**  
    - Antecedents → Consequents: if the antecedents occur, consequents are likely.  
    - Support indicates frequency; confidence indicates reliability; lift indicates strength above chance.
    """)

# ========== TAB 5: REGRESSION ==========
with tabs[4]:
    st.markdown("## :chart_with_upwards_trend: Regression Insights")
    targets = ['SpendPerVisitINR','MaxFoodSpendINR','OverallSatisfaction']
    feats = ['Age','MonthlyIncomeINR','EventsAttendedAnnually','OverallSatisfaction']
    regs = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=0)
    }
    results = []
    for t in targets:
        y = df[t]; X = df[feats]
        Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)
        for name, model in regs.items():
            model.fit(Xtr, ytr)
            r2 = model.score(Xts, yts)
            results.append([name, t, round(r2, 3)])

    reg_df = pd.DataFrame(results, columns=["Model","Target","R2"]).pivot_table(
        index="Model", columns="Target", values="R2", aggfunc="first"
    )
    st.dataframe(reg_df, use_container_width=True)
    st.markdown("**Interpretation:** R² indicates how much variance in the target is explained by the features.")

    # Regression feature importance (Decision Tree)
    st.markdown("### Regression Feature Importances (Decision Tree for SpendPerVisitINR)")
    X_spend = df[feats]; y_spend = df["SpendPerVisitINR"]
    Xtr_s, Xts_s, ytr_s, yts_s = train_test_split(X_spend, y_spend, test_size=0.2, random_state=42)
    dt_spend = DecisionTreeRegressor(random_state=0).fit(Xtr_s, ytr_s)
    imp_spend = pd.DataFrame({
        "Feature": feats,
        "Importance": dt_spend.feature_importances_
    }).sort_values("Importance", ascending=False)
    fig_imp_reg = px.bar(imp_spend, x="Importance", y="Feature", orientation="h",
                         title="Decision Tree Feature Importances")
    st.plotly_chart(fig_imp_reg, use_container_width=True)
    st.markdown("""
    **Interpretation:**  
    Features with larger importance contribute more to predicting spend per visit.
    """)

    # Actual vs. Predicted (Linear)
    lin = LinearRegression().fit(Xtr_s, ytr_s)
    preds = lin.predict(Xts_s)
    fig_reg, ax_reg = plt.subplots()
    ax_reg.scatter(yts_s, preds, alpha=0.5)
    ax_reg.plot([yts_s.min(), yts_s.max()], [yts_s.min(), yts_s.max()], 'r--')
    ax_reg.set_xlabel("Actual Spend"); ax_reg.set_ylabel("Predicted Spend")
    st.pyplot(fig_reg)
    st.markdown("**Interpretation:** Points close to the red line indicate accurate predictions; outliers highlight errors.")

    st.markdown("#### Key Takeaways")
    st.write(
        "- Income & satisfaction are strong predictors of event spend.\n"
        "- Ridge regression and decision tree often yield highest test R².\n"
        "- Luxury spenders (outliers) remain challenging to model accurately."
    )

# ========== TAB 6: RECOMMENDATIONS ==========
with tabs[5]:
    st.markdown("## 📝 Final Recommendations")
    st.markdown("""
    **1. Demographic Targeting (Data Viz):**  
    - Prioritize ages 25–45, the core attendee group.  
    - Tailor high-value offers to outlier spenders.

    **2. Premium Upsells:**  
    - Offer premium seating to “Yes”/“Maybe” segments—high spend potential.  
    - Use classification model at signup to identify likely sponsors.

    **3. Operational Wins:**  
    - Improve parking & signage—top complaints.  
    - Boost staffing in cities with low satisfaction (Delhi).

    **4. Persona-based Messaging:**  
    - Cluster 0 (Young Professionals): emphasize networking & tech.  
    - Cluster 1 (Families): focus on entertainment & comfort.  
    - Cluster 2 (Academics/Students): highlight knowledge sessions.

    **5. F&B Bundles:**  
    - Bundle popular combos (e.g., Indian + Quick snacks).  
    - Expand vegan/gluten-free for ~20% niche demand.

    **6. Pricing & Forecasting:**  
    - Apply regression forecasts to set dynamic pricing by city/income.  
    - Monitor model R² quarterly to adjust strategies.

    **7. Ongoing Analytics:**  
    - Retrain models after major events to capture evolving behavior.  
    - Track key correlations (income↔spend, satisfaction↔recommendation) continuously.
    """)
