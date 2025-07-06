import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

import io

# Set page config
st.set_page_config(page_title="Jio World Centre Analytics Dashboard", page_icon=":bar_chart:", layout="wide")

# === SIDEBAR ===
st.sidebar.image("data/logo_jwc.png", width=180)
st.sidebar.title("Jio World Centre Dashboard")
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
        body { background-color: #212121; color: #F5F5F5; }
        </style>
        """, unsafe_allow_html=True)

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/JioWorldCentre_Survey_Synthetic.csv")
    return df

df = load_data()

# --- Executive Summary ---
st.markdown("""
# <span style='color:#002f6c'>Jio World Centre, Mumbai: Consumer Insights Dashboard</span>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:#ffe5b4; border-radius:10px; padding:18px; border: 1px solid #e3b05a'>
<b>Executive Summary:</b><br>
Welcome to the analytics platform for the Jio World Centre, Mumbai. This dashboard provides a deep dive into real-world event consumer behaviour, preferences, and business opportunities using state-of-the-art machine learning and data mining techniques.
<br><br>
<b>Key Features:</b>
<ul>
  <li>Explore complex, actionable insights into consumer personas, event trends, and spending patterns.</li>
  <li>Classify customer willingness and segment event visitors using advanced ML models.</li>
  <li>Identify customer personas and opportunities with clustering and association rule mining.</li>
  <li>Test and predict outcomes on new datasets, with interactive upload and download features.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# --- Main Tabs ---
tabs = st.tabs([
    "Data Visualisation",
    "Classification",
    "Clustering",
    "Association Rule Mining",
    "Regression"
])

# ========== TAB 1: DATA VISUALISATION ==========
with tabs[0]:
    st.markdown("## :bar_chart: Data Visualisation & Insights")

    # Filter Section
    with st.expander("**Filter Data**", expanded=False):
        gender = st.multiselect("Gender", options=df["Gender"].unique(), default=list(df["Gender"].unique()))
        city = st.multiselect("City", options=df["City"].unique(), default=list(df["City"].unique()))
        event_type = st.multiselect("Visit Reason", options=df["VisitReason"].unique(), default=list(df["VisitReason"].unique()))
        min_age, max_age = int(df.Age.min()), int(df.Age.max())
        age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age))
        df_viz = df[
            (df["Gender"].isin(gender)) &
            (df["City"].isin(city)) &
            (df["VisitReason"].isin(event_type)) &
            (df["Age"] >= age_range[0]) &
            (df["Age"] <= age_range[1])
        ]
        st.caption(f"**{len(df_viz)} responses filtered**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Median Spend per Visit (INR)", f"{int(df_viz['SpendPerVisitINR'].median()):,}")
        st.metric("Most Common Event Type", df_viz["VisitReason"].mode()[0])
    with col2:
        st.metric("Most Preferred City", df_viz["City"].mode()[0])
        st.metric("Median Age", int(df_viz["Age"].median()))
    with col3:
        st.metric("Highest Food Spend (INR)", f"{int(df_viz['MaxFoodSpendINR'].max()):,}")
        st.metric("Top Occupation", df_viz["Occupation"].mode()[0])

    st.markdown("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_viz["Age"], bins=20, kde=True, ax=ax, color="#3498db")
    ax.set_xlabel("Age")
    st.pyplot(fig)
    st.caption("**How to interpret:** Shows the age spread of event attendees. Skewness or spikes can reveal age-specific marketing targets.")

    st.markdown("### Spend per Visit by City & Event Type")
    fig2 = px.box(df_viz, x="City", y="SpendPerVisitINR", color="VisitReason", points="all")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("**How to interpret:** Helps compare spend patterns by city and event type. Outliers indicate high spenders.")

    st.markdown("### Occupation vs. Networking Interest")
    occ_net = pd.crosstab(df_viz["Occupation"], df_viz["NetworkingInterest"])
    fig3 = px.bar(occ_net, barmode='group')
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("**How to interpret:** Shows which occupations are most interested in networking.")

    st.markdown("### Event Attendance Frequency")
    fig4 = px.pie(df_viz, names="ParticipationFrequency", title="Event Attendance Frequency")
    st.plotly_chart(fig4)
    st.caption("**How to interpret:** Reveals customer loyalty and repeat visit potential.")

    st.markdown("### Food Preferences")
    food_counts = df_viz["FoodPreferences"].str.get_dummies(sep=',').sum().sort_values(ascending=False)
    fig5 = px.bar(food_counts, labels={'value':'Count', 'index':'Food'})
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("**How to interpret:** Indicates demand for each food type at events.")

    st.markdown("### Premium Seating Interest vs. Spend")
    fig6 = px.box(df_viz, x="PremiumSeatingInterest", y="SpendPerVisitINR", color="PremiumSeatingInterest")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("**How to interpret:** Higher spend among those interested in premium seating suggests upsell opportunities.")

    st.markdown("### Challenges Faced at Events")
    challenge_counts = df_viz["ChallengesFaced"].str.get_dummies(sep=',').sum().sort_values(ascending=False)
    fig7 = px.bar(challenge_counts, labels={'value':'Count', 'index':'Challenge'})
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("**How to interpret:** Top pain points for process improvement.")

    st.markdown("### Overall Satisfaction by City")
    fig8 = px.box(df_viz, x="City", y="OverallSatisfaction", color="City")
    st.plotly_chart(fig8)
    st.caption("**How to interpret:** Track city-wise satisfaction for targeted improvement.")

    st.markdown("### Correlation Heatmap")
    corr_cols = ["Age", "MonthlyIncomeINR", "EventsAttendedAnnually", "SpendPerVisitINR", "MaxFoodSpendINR",
                 "RecommendJioWorldCentre", "OverallSatisfaction"]
    corr = df_viz[corr_cols].corr()
    fig9, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig9)
    st.caption("**How to interpret:** Stronger positive/negative relationships help find drivers of spend or satisfaction.")

    st.markdown("### Download Data")
    csv = df_viz.to_csv(index=False).encode()
    st.download_button("Download Filtered Data as CSV", csv, "Filtered_JioWorldCentre_Data.csv", "text/csv")

# ========== TAB 2: CLASSIFICATION ==========
with tabs[1]:
    st.markdown("## :guardsman: Customer Willingness Classification")
    st.markdown("#### Predict who is willing to be a Sponsor/Exhibitor")
    st.info("Target: SponsorOrExhibitorInterest ('Yes' vs. others)")

    # Preprocessing for classification
    clf_df = df.copy()
    clf_df['Sponsor_Label'] = (clf_df['SponsorOrExhibitorInterest']=='Yes').astype(int)
    features = ["Age","MonthlyIncomeINR","EventsAttendedAnnually","SpendPerVisitINR","MaxFoodSpendINR","OverallSatisfaction"]
    # Add a few categorical encodings
    cat_features = ['Gender','Education','Occupation','PreferredEventType']
    for col in cat_features:
        le = LabelEncoder()
        clf_df[col] = le.fit_transform(clf_df[col])
        features.append(col)
    X = clf_df[features]
    y = clf_df['Sponsor_Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    # Model options
    classifiers = {
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=0),
        'Random Forest': RandomForestClassifier(random_state=0),
        'Gradient Boosting': GradientBoostingClassifier(random_state=0)
    }
    results = []
    roc_data = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append([name, acc, prec, rec, f1])
        # For ROC
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:,1]
        else:
            y_score = clf.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))

    res_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1-Score"])
    st.dataframe(res_df.style.background_gradient(cmap='YlGnBu'), use_container_width=True)
    st.caption("Shows key metrics on the test data for each algorithm. Compare to select best performer.")

    # Confusion matrix toggle
    st.markdown("#### Confusion Matrix")
    selected_model = st.selectbox("Select model for Confusion Matrix", list(classifiers.keys()))
    model = classifiers[selected_model]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Sponsor", "Sponsor"], yticklabels=["Not Sponsor", "Sponsor"], ax=ax_cm)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig_cm)
    st.caption("**How to interpret:** True Positive/Negative and False Positive/Negative counts shown.")

    # ROC Curve
    st.markdown("#### ROC Curve (All Models)")
    fig_roc, ax_roc = plt.subplots()
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    ax_roc.plot([0,1],[0,1],'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend()
    st.pyplot(fig_roc)
    st.caption("**How to interpret:** Higher curve = better model. Area under curve (AUC) closer to 1 is preferred.")

    # Upload new data for prediction
    st.markdown("#### Predict on New Uploaded Data")
    uploaded_file = st.file_uploader("Upload new customer CSV (no target variable)", type="csv")
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        for col in cat_features:
            le = LabelEncoder()
            new_data[col] = le.fit_transform(new_data[col].astype(str))
        pred = model.predict(new_data[features])
        new_data["Predicted_Sponsor"] = ["Yes" if p==1 else "No" for p in pred]
        st.dataframe(new_data)
        pred_csv = new_data.to_csv(index=False).encode()
        st.download_button("Download Predictions", pred_csv, "Predictions.csv", "text/csv")

# ========== TAB 3: CLUSTERING ==========
with tabs[2]:
    st.markdown("## :busts_in_silhouette: Clustering - Customer Personas (K-Means)")
    clust_df = df.copy()
    num_cols = ["Age","MonthlyIncomeINR","EventsAttendedAnnually","SpendPerVisitINR","MaxFoodSpendINR","OverallSatisfaction"]
    scaler = StandardScaler()
    clust_data = scaler.fit_transform(clust_df[num_cols])

    k = st.slider("Select number of clusters (k)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(clust_data)
    clust_df['Cluster'] = cluster_labels

    # Elbow Chart
    inertia = []
    for c in range(2, 11):
        km = KMeans(n_clusters=c, random_state=42, n_init=10)
        km.fit(clust_data)
        inertia.append(km.inertia_)
    fig_elbow = px.line(x=list(range(2, 11)), y=inertia, markers=True)
    fig_elbow.update_layout(xaxis_title="Number of Clusters", yaxis_title="Inertia (WCSS)", title="Elbow Method")
    st.plotly_chart(fig_elbow)
    st.caption("**How to interpret:** The 'elbow' point is a good choice for k.")

    # Silhouette Score Chart
    sil_scores = []
    for c in range(2, 11):
        km = KMeans(n_clusters=c, random_state=42, n_init=10)
        labels = km.fit_predict(clust_data)
        sil_scores.append(silhouette_score(clust_data, labels))
    fig_sil = px.line(x=list(range(2, 11)), y=sil_scores, markers=True)
    fig_sil.update_layout(xaxis_title="Number of Clusters", yaxis_title="Silhouette Score", title="Silhouette Scores")
    st.plotly_chart(fig_sil)
    st.caption("**How to interpret:** Higher silhouette score = better-defined clusters.")

    # Persona Table
    st.markdown("#### Cluster Personas")
    persona = clust_df.groupby('Cluster')[num_cols].mean().round(1)
    st.dataframe(persona.style.background_gradient(cmap="BuGn"), use_container_width=True)
    st.caption("Each row summarizes the average profile for customers in that cluster.")

    # Download cluster-labeled data
    csv_clust = clust_df.to_csv(index=False).encode()
    st.download_button("Download Data with Cluster Labels", csv_clust, "Clustered_Data.csv", "text/csv")

# ========== TAB 4: ASSOCIATION RULE MINING ==========
with tabs[3]:
    st.markdown("## :bulb: Association Rule Mining (Apriori)")

    st.markdown("### Choose Columns for Association Rule Mining")
    # Only columns with comma-separated values
    assoc_cols = ['FoodPreferences', 'ChallengesFaced']
    selected_col = st.selectbox("Select column", assoc_cols)
    min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05)

    # Transaction DataFrame
    assoc_df = df[selected_col].str.get_dummies(sep=',')
    freq_items = apriori(assoc_df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    rules = rules.sort_values("confidence", ascending=False).head(10)
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    st.caption(f"Showing top 10 association rules for {selected_col} (filtered by support/confidence).")
    st.markdown("""
        **How to interpret:** If {antecedents} is observed, then {consequents} is likely, with confidence/lift shown.
    """)

# ========== TAB 5: REGRESSION ==========
with tabs[4]:
    st.markdown("## :chart_with_upwards_trend: Regression & Predictive Insights")
    st.markdown("#### How income and event behaviour impact spending?")

    reg_df = df.copy()
    reg_targets = ['SpendPerVisitINR', 'MaxFoodSpendINR', 'OverallSatisfaction']
    input_features = ['Age','MonthlyIncomeINR','EventsAttendedAnnually','OverallSatisfaction']

    # Results Table
    reg_results = []
    regressors = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=0)
    }
    for target in reg_targets:
        y = reg_df[target]
        X = reg_df[input_features]
        Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)
        for name, model in regressors.items():
            model.fit(Xtr, ytr)
            r2 = model.score(Xts, yts)
            reg_results.append([name, target, round(r2, 3)])

    reg_resdf = pd.DataFrame(reg_results, columns=["Regressor","Target","Test R^2"])
    st.dataframe(reg_resdf.pivot(index="Regressor", columns="Target", values="Test R^2"))
    st.caption("Shows how well each regression model predicts each target variable.")

    # Chart example: Actual vs Predicted for best model/target
    st.markdown("#### Actual vs. Predicted Spend (Linear Regression)")
    y = reg_df["SpendPerVisitINR"]
    X = reg_df[input_features]
    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)
    linreg = LinearRegression().fit(Xtr, ytr)
    preds = linreg.predict(Xts)
    fig_reg, ax_reg = plt.subplots()
    ax_reg.scatter(yts, preds, alpha=0.5)
    ax_reg.plot([yts.min(), yts.max()], [yts.min(), yts.max()], 'r--')
    ax_reg.set_xlabel("Actual Spend")
    ax_reg.set_ylabel("Predicted Spend")
    st.pyplot(fig_reg)
    st.caption("**How to interpret:** Points near the red line indicate accurate predictions.")

    # Insights
    st.markdown("#### Regression Insights")
    st.write("""
    - **Income and satisfaction** are strong predictors of spend and food spend.
    - Decision Tree and Ridge often provide the highest R² for spend prediction.
    - Some outliers are not well predicted—possibly due to rare luxury spenders.
    """)

# END OF APP
