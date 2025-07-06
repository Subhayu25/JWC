import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

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

import warnings
warnings.filterwarnings("ignore")

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Jio World Centre Analytics Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

# === SIDEBAR WITH LOGO UPLOAD & GITHUB FALLBACK ===
st.sidebar.title("Jio World Centre Dashboard")
uploaded_logo = st.sidebar.file_uploader(
    "Upload Logo (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
    help="Optional: override default logo"
)
if uploaded_logo:
    st.sidebar.image(uploaded_logo, width=180)
elif os.path.exists("data/logo_jwc.png"):
    st.sidebar.image("data/logo_jwc.png", width=180)
else:
    # fallback to GitHub raw URL (replace placeholders accordingly)
    github_logo_url = (
        "https://raw.githubusercontent.com/USERNAME/REPO/BRANCH/"
        "data/logo_jwc.png"
    )
    st.sidebar.image(github_logo_url, width=180)

theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        "<style>body{background:#212121;color:#F5F5F5;} </style>",
        unsafe_allow_html=True
    )

# --- Load data safely, with GitHub fallback ---
@st.cache_data
def load_data():
    local_paths = [
        "data/JioWorldCentre_Survey_Synthetic.csv",
        "JioWorldCentre_Survey_Synthetic.csv"
    ]
    for path in local_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    # fallback to GitHub raw URL (replace placeholders accordingly)
    github_url = (
        "https://raw.githubusercontent.com/USERNAME/REPO/BRANCH/"
        "data/JioWorldCentre_Survey_Synthetic.csv"
    )
    df = pd.read_csv(github_url)
    st.sidebar.success("Loaded CSV from GitHub")
    return df

df = load_data()

# --- Executive Summary (dark blue background, white text) ---
st.markdown("# Jio World Centre, Mumbai: Consumer Insights Dashboard")
st.markdown(
    """
    <div style='background:#002f6c; padding:20px; border-radius:10px;'>
      <h3 style='color:#ffffff; margin:0;'>Executive Summary</h3>
      <p style='color:#ffffff; font-size:16px;'>
        This dashboard analyzes a synthetic event consumer dataset from Jio World Centre, Mumbai,
        using machine learning, clustering, regression, and association rule mining. Explore visitor
        personas, spending patterns, satisfaction scores, and business opportunities interactively.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

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

    # Filters
    with st.expander("Filter Data"):
        gender = st.multiselect("Gender", df["Gender"].unique(), df["Gender"].unique())
        city = st.multiselect("City", df["City"].unique(), df["City"].unique())
        reason = st.multiselect("Visit Reason", df["VisitReason"].unique(), df["VisitReason"].unique())
        min_age, max_age = int(df.Age.min()), int(df.Age.max())
        age_range = st.slider("Age Range", min_age, max_age, (min_age, max_age))
        df_viz = df[
            df.Gender.isin(gender) &
            df.City.isin(city) &
            df.VisitReason.isin(reason) &
            df.Age.between(age_range[0], age_range[1])
        ]
        st.caption(f"**{len(df_viz)} responses filtered**")

    # Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Median Spend per Visit (INR)", f"{int(df_viz.SpendPerVisitINR.median()):,}")
        st.metric("Most Common Event Type", df_viz.VisitReason.mode()[0])
    with c2:
        st.metric("Most Preferred City", df_viz.City.mode()[0])
        st.metric("Median Age", int(df_viz.Age.median()))
    with c3:
        st.metric("Highest Food Spend (INR)", f"{int(df_viz.MaxFoodSpendINR.max()):,}")
        st.metric("Top Occupation", df_viz.Occupation.mode()[0])

    # 1. Age Distribution
    st.markdown("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_viz.Age, bins=20, kde=True, ax=ax, color="#4C72B0")
    ax.set_xlabel("Age")
    st.pyplot(fig)
    st.caption("Shows the age spread of attendees.")

    # 2. Spend per Visit by City & Event Type
    st.markdown("### Spend per Visit by City & Event Type")
    fig2 = px.box(df_viz, x="City", y="SpendPerVisitINR", color="VisitReason", points="all")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Compare spend patterns; outliers show high spenders.")

    # 3. Occupation vs. Networking Interest
    st.markdown("### Occupation vs. Networking Interest")
    occ_net = pd.crosstab(df_viz.Occupation, df_viz.NetworkingInterest)
    fig3 = px.bar(occ_net, barmode='group')
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Which occupations are most interested in networking?")

    # 4. Event Attendance Frequency
    st.markdown("### Event Attendance Frequency")
    fig4 = px.pie(df_viz, names="ParticipationFrequency", title="Attendance Frequency")
    st.plotly_chart(fig4)
    st.caption("Loyalty and repeat visit potential.")

    # 5. Food Preferences
    st.markdown("### Food Preferences")
    food_counts = df_viz.FoodPreferences.str.get_dummies(sep=',').sum().sort_values(ascending=False)
    fig5 = px.bar(food_counts, labels={'value':'Count','index':'Food'})
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Demand for each food type at events.")

    # 6. Premium Seating vs. Spend
    st.markdown("### Premium Seating Interest vs. Spend")
    fig6 = px.box(df_viz, x="PremiumSeatingInterest", y="SpendPerVisitINR", color="PremiumSeatingInterest")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("Higher spend among premium seekers.")

    # 7. Challenges Faced
    st.markdown("### Challenges Faced at Events")
    chall_counts = df_viz.ChallengesFaced.str.get_dummies(sep=',').sum().sort_values(ascending=False)
    fig7 = px.bar(chall_counts, labels={'value':'Count','index':'Challenge'})
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("Top pain points for improvement.")

    # 8. Overall Satisfaction by City
    st.markdown("### Overall Satisfaction by City")
    fig8 = px.box(df_viz, x="City", y="OverallSatisfaction", color="City")
    st.plotly_chart(fig8, use_container_width=True)
    st.caption("City-wise satisfaction comparison.")

    # 9. Correlation Heatmap
    st.markdown("### Correlation Heatmap")
    corr_cols = ["Age","MonthlyIncomeINR","EventsAttendedAnnually","SpendPerVisitINR","MaxFoodSpendINR","RecommendJioWorldCentre","OverallSatisfaction"]
    corr = df_viz[corr_cols].corr()
    fig9, ax9 = plt.subplots(figsize=(7,5))
    sns.heatmap(corr, annot=True, cmap="vlag", ax=ax9)
    st.pyplot(fig9)
    st.caption("Correlations reveal drivers of spend & satisfaction.")

    # 10. Download Filtered Data
    st.markdown("### Download Filtered Data")
    csv = df_viz.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "Filtered_JWC.csv", "text/csv")

# ========== TAB 2: CLASSIFICATION ==========
with tabs[1]:
    st.markdown("## :guardsman: Classification - Sponsor/Exhibitor Willingness")
    st.info("Target: SponsorOrExhibitorInterest == 'Yes'")

    clf = df.copy()
    clf['Sponsor_Label'] = (clf.SponsorOrExhibitorInterest == 'Yes').astype(int)
    features = ["Age","MonthlyIncomeINR","EventsAttendedAnnually","SpendPerVisitINR","MaxFoodSpendINR","OverallSatisfaction"]
    cat_cols = ['Gender','Education','Occupation','PreferredEventType']
    for col in cat_cols:
        le = LabelEncoder()
        clf[col] = le.fit_transform(clf[col])
        features.append(col)

    X = clf[features]; y = clf['Sponsor_Label']
    Xtr, Xts, ytr, yts = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=0),
        'Random Forest': RandomForestClassifier(random_state=0),
        'Gradient Boosting': GradientBoostingClassifier(random_state=0)
    }
    metrics = []; roc_data = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        preds = m.predict(Xts)
        acc = accuracy_score(yts, preds)
        prec = precision_score(yts, preds)
        rec = recall_score(yts, preds)
        f1 = f1_score(yts, preds)
        metrics.append([name, acc, prec, rec, f1])
        if hasattr(m, "predict_proba"):
            scores = m.predict_proba(Xts)[:,1]
        else:
            scores = m.decision_function(Xts)
        fpr, tpr, _ = roc_curve(yts, scores)
        roc_data[name] = (fpr, tpr, auc(fpr, tpr))

    mdf = pd.DataFrame(metrics, columns=["Model","Accuracy","Precision","Recall","F1-Score"])
    st.dataframe(mdf.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.markdown("#### Confusion Matrix")
    sel = st.selectbox("Model", list(models.keys()))
    m = models[sel]
    cm = confusion_matrix(yts, m.predict(Xts))
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm,
                xticklabels=["No","Yes"], yticklabels=["No","Yes"])
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.markdown("#### ROC Curve")
    fig_roc, ax_roc = plt.subplots()
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    ax_roc.plot([0,1],[0,1],'k--')
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.legend()
    st.pyplot(fig_roc)

    st.markdown("#### Predict on New Data")
    up = st.file_uploader("Upload CSV (no target)", type="csv")
    if up:
        new = pd.read_csv(up)
        for col in cat_cols:
            le = LabelEncoder()
            new[col] = le.fit_transform(new[col].astype(str))
        new_preds = m.predict(new[features])
        new["Predicted_Sponsor"] = ["Yes" if p==1 else "No" for p in new_preds]
        st.dataframe(new)
        out = new.to_csv(index=False).encode()
        st.download_button("Download Predictions", out, "Preds.csv", "text/csv")

# ========== TAB 3: CLUSTERING ==========
with tabs[2]:
    st.markdown("## :busts_in_silhouette: Clustering - Customer Personas")
    cdf = df.copy()
    numf = ["Age","MonthlyIncomeINR","EventsAttendedAnnually","SpendPerVisitINR","MaxFoodSpendINR","OverallSatisfaction"]
    sc = StandardScaler()
    Xc = sc.fit_transform(cdf[numf])

    k = st.slider("Clusters (k)", 2, 10, 3)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cdf["Cluster"] = km.fit_predict(Xc)

    inert = []; sils = []
    for i in range(2,11):
        km2 = KMeans(n_clusters=i, random_state=42, n_init=10)
        labs = km2.fit_predict(Xc)
        inert.append(km2.inertia_)
        sils.append(silhouette_score(Xc, labs))

    fig_elb = px.line(x=list(range(2,11)), y=inert, markers=True, title="Elbow Method")
    st.plotly_chart(fig_elb)
    fig_sil = px.line(x=list(range(2,11)), y=sils, markers=True, title="Silhouette Scores")
    st.plotly_chart(fig_sil)

    st.markdown("#### Cluster Personas")
    pers = cdf.groupby("Cluster")[numf].mean().round(1)
    st.dataframe(pers, use_container_width=True)

    dl = cdf.to_csv(index=False).encode()
    st.download_button("Download Clustered Data", dl, "clustered.csv", "text/csv")

# ========== TAB 4: ASSOCIATION RULE MINING ==========
with tabs[3]:
    st.markdown("## :bulb: Association Rule Mining")
    cols = ['FoodPreferences','ChallengesFaced']
    selc = st.selectbox("Column", cols)
    ms = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
    mc = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
    td = df[selc].str.get_dummies(sep=',')
    fi = apriori(td, min_support=ms, use_colnames=True)
    rules = association_rules(fi, metric="confidence", min_threshold=mc).sort_values("confidence", ascending=False).head(10)
    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])
    st.markdown("**Interpretation:** Antecedents → Consequents with given confidence & lift.")

# ========== TAB 5: REGRESSION ==========
with tabs[4]:
    st.markdown("## :chart_with_upwards_trend: Regression & Predictive Insights")
    targets = ['SpendPerVisitINR','MaxFoodSpendINR','OverallSatisfaction']
    feats = ['Age','MonthlyIncomeINR','EventsAttendedAnnually','OverallSatisfaction']
    regs = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(random_state=0)
    }
    results = []
    for tgt in targets:
        y = df[tgt]; X = df[feats]
        Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)
        for name, model in regs.items():
            model.fit(Xtr, ytr)
            r2 = model.score(Xts, yts)
            results.append([name, tgt, round(r2,3)])
    rdf = pd.DataFrame(results, columns=["Model","Target","R2"]).pivot("Model","Target","R2")
    st.dataframe(rdf, use_container_width=True)

    lin = LinearRegression().fit(Xtr, ytr)
    preds = lin.predict(Xts)
    figr, axr = plt.subplots()
    axr.scatter(yts, preds, alpha=0.5)
    axr.plot([yts.min(),yts.max()],[yts.min(),yts.max()],'r--')
    axr.set_xlabel("Actual"); axr.set_ylabel("Predicted")
    st.pyplot(figr)

    st.markdown("#### Insights")
    st.write(
        "- Income & satisfaction are strong predictors of spend.\n"
        "- Ridge & Decision Tree often highest R².\n"
        "- Outliers (luxury spenders) remain hard to predict."
    )
