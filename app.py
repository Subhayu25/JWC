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

# === SIDEBAR WITH LOGO UPLOAD ===
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
    st.sidebar.info("No logo found. Upload a logo or add 'data/logo_jwc.png'.")

theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        "<style>body{background:#212121;color:#F5F5F5;}</style>",
        unsafe_allow_html=True
    )

# --- Load data safely, with GitHub fallback ---
@st.cache_data
def load_data():
    # Check local paths first:
    local_paths = [
        "data/JioWorldCentre_Survey_Synthetic.csv",
        "JioWorldCentre_Survey_Synthetic.csv"
    ]
    for path in local_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    # Fallback to raw GitHub URL:
    github_url = (
        "https://raw.githubusercontent.com/USERNAME/REPO/BRANCH/"
        "data/JioWorldCentre_Survey_Synthetic.csv"
    )
    try:
        df = pd.read_csv(github_url)
        st.sidebar.success("Loaded CSV from GitHub")
        return df
    except Exception as e:
        st.error(f"Could not load CSV. Tried local and GitHub.\n{e}")
        st.stop()

df = load_data()

# --- Executive Summary ---
st.markdown("# Jio World Centre, Mumbai: Consumer Insights Dashboard")
st.markdown(
    """
    <div style='background:#ffe5b4; padding:15px; border-radius:8px;'>
    <b>Executive Summary:</b><br>
    This dashboard analyzes a synthetic event consumer dataset from Jio World Centre, Mumbai 
    using machine learning, clustering, regression, and association rule mining. Explore visitor 
    personas, spending patterns, satisfaction scores, and business opportunities interactively.
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
        gender = st.multiselect("Gender", options=df["Gender"].unique(),
                                default=list(df["Gender"].unique()))
        city = st.multiselect("City", options=df["City"].unique(),
                              default=list(df["City"].unique()))
        event_type = st.multiselect("Visit Reason", options=df["VisitReason"].unique(),
                                    default=list(df["VisitReason"].unique()))
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

    # Key Metrics
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

    # 1. Age Distribution
    st.markdown("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_viz["Age"], bins=20, kde=True, ax=ax, color="#3498db")
    ax.set_xlabel("Age")
    st.pyplot(fig)
    st.caption("**How to interpret:** Shows the age spread of attendees.")

    # 2. Spend per Visit by City & Event Type
    st.markdown("### Spend per Visit by City & Event Type")
    fig2 = px.box(df_viz, x="City", y="SpendPerVisitINR",
                  color="VisitReason", points="all")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("**How to interpret:** Compare spend patterns; outliers show high spenders.")

    # 3. Occupation vs. Networking Interest
    st.markdown("### Occupation vs. Networking Interest")
    occ_net = pd.crosstab(df_viz["Occupation"], df_viz["NetworkingInterest"])
    fig3 = px.bar(occ_net, barmode='group')
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("**How to interpret:** Which occupations are most interested in networking.")

    # 4. Event Attendance Frequency
    st.markdown("### Event Attendance Frequency")
    fig4 = px.pie(df_viz, names="ParticipationFrequency", title="Attendance Frequency")
    st.plotly_chart(fig4)
    st.caption("**How to interpret:** Loyalty and repeat visit potential.")

    # 5. Food Preferences
    st.markdown("### Food Preferences")
    food_counts = df_viz["FoodPreferences"].str.get_dummies(sep=',').sum().sort_values(ascending=False)
    fig5 = px.bar(food_counts, labels={'value':'Count','index':'Food'})
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("**How to interpret:** Demand for each food type.")

    # 6. Premium Seating Interest vs. Spend
    st.markdown("### Premium Seating Interest vs. Spend")
    fig6 = px.box(df_viz, x="PremiumSeatingInterest", y="SpendPerVisitINR",
                  color="PremiumSeatingInterest")
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("**How to interpret:** Higher spend for premium seekers.")

    # 7. Challenges Faced at Events
    st.markdown("### Challenges Faced at Events")
    challenge_counts = df_viz["ChallengesFaced"].str.get_dummies(sep=',').sum().sort_values(ascending=False)
    fig7 = px.bar(challenge_counts, labels={'value':'Count','index':'Challenge'})
    st.plotly_chart(fig7, use_container_width=True)
    st.caption("**How to interpret:** Top pain points.")

    # 8. Overall Satisfaction by City
    st.markdown("### Overall Satisfaction by City")
    fig8 = px.box(df_viz, x="City", y="OverallSatisfaction", color="City")
    st.plotly_chart(fig8, use_container_width=True)
    st.caption("**How to interpret:** City-wise satisfaction.")

    # 9. Correlation Heatmap
    st.markdown("### Correlation Heatmap")
    corr_cols = [
        "Age","MonthlyIncomeINR","EventsAttendedAnnually",
        "SpendPerVisitINR","MaxFoodSpendINR",
        "RecommendJioWorldCentre","OverallSatisfaction"
    ]
    corr = df_viz[corr_cols].corr()
    fig9, ax9 = plt.subplots(figsize=(7,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax9)
    st.pyplot(fig9)
    st.caption("**How to interpret:** Correlations reveal drivers.")

    # 10. Download Filtered Data
    st.markdown("### Download Filtered Data")
    csv = df_viz.to_csv(index=False).encode()
    st.download_button(
        "Download Filtered CSV",
        data=csv,
        file_name="Filtered_JWC_Data.csv",
        mime="text/csv"
    )

# ========== TAB 2: CLASSIFICATION ==========
with tabs[1]:
    st.markdown("## :guardsman: Classification - Sponsor/Exhibitor Willingness")
    st.info("Target: SponsorOrExhibitorInterest == 'Yes'")

    # Prepare classification dataset
    clf_df = df.copy()
    clf_df['Sponsor_Label'] = (clf_df['SponsorOrExhibitorInterest']=='Yes').astype(int)
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
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, stratify=y, test_size=0.25, random_state=42
    )

    # Models
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
        acc = accuracy_score(y_test,y_pred)
        prec = precision_score(y_test,y_pred)
        rec = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        results.append([name,acc,prec,rec,f1])
        # ROC
        if hasattr(clf,'predict_proba'):
            y_score = clf.predict_proba(X_test)[:,1]
        else:
            y_score = clf.decision_function(X_test)
        fpr,tpr,_ = roc_curve(y_test,y_score)
        roc_data[name] = (fpr,tpr,auc(fpr,tpr))

    res_df = pd.DataFrame(results,columns=["Model","Accuracy","Precision","Recall","F1-Score"])
    st.dataframe(res_df.style.background_gradient(cmap='YlGnBu'),use_container_width=True)
    st.caption("Test metrics for each classifier.")

    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    sel = st.selectbox("Model",list(classifiers.keys()))
    m = classifiers[sel]
    y_pred = m.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    fig_cm,ax_cm = plt.subplots()
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                xticklabels=["No","Yes"],yticklabels=["No","Yes"],ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
    st.caption("TP, TN, FP, FN counts.")

    # ROC Curve
    st.markdown("#### ROC Curve (All Models)")
    fig_roc,ax_roc = plt.subplots()
    for name,(fpr,tpr,roc_auc) in roc_data.items():
        ax_roc.plot(fpr,tpr,label=f"{name} (AUC={roc_auc:.2f})")
    ax_roc.plot([0,1],[0,1],'k--')
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.legend()
    st.pyplot(fig_roc)
    st.caption("Higher curve = better model.")

    # Upload new for prediction
    st.markdown("#### Upload New Data for Prediction")
    up = st.file_uploader("CSV (no target)",type="csv")
    if up:
        new = pd.read_csv(up)
        for col in cat_cols:
            le = LabelEncoder()
            new[col] = le.fit_transform(new[col].astype(str))
        preds = m.predict(new[features])
        new["Predicted_Sponsor"] = ["Yes" if p==1 else "No" for p in preds]
        st.dataframe(new)
        out = new.to_csv(index=False).encode()
        st.download_button("Download Predictions",out,"Predictions.csv","text/csv")

# ========== TAB 3: CLUSTERING ==========
with tabs[2]:
    st.markdown("## :busts_in_silhouette: Clustering - Customer Personas")
    clust_df = df.copy()
    num_cols = [
        "Age","MonthlyIncomeINR","EventsAttendedAnnually",
        "SpendPerVisitINR","MaxFoodSpendINR","OverallSatisfaction"
    ]
    scaler = StandardScaler()
    X_clust = scaler.fit_transform(clust_df[num_cols])

    k = st.slider("Number of Clusters (k)",2,10,3)
    kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
    clust_df["Cluster"] = kmeans.fit_predict(X_clust)

    # Elbow
    inertias=[]
    for i in range(2,11):
        km=KMeans(n_clusters=i,random_state=42,n_init=10)
        km.fit(X_clust)
        inertias.append(km.inertia_)
    fig_elbow=px.line(x=list(range(2,11)),y=inertias,markers=True,
                      labels={'x':'k','y':'Inertia'},title="Elbow Method")
    st.plotly_chart(fig_elbow)
    st.caption("Elbow suggests optimal k.")

    # Silhouette
    sils=[]
    for i in range(2,11):
        km=KMeans(n_clusters=i,random_state=42,n_init=10)
        labs=km.fit_predict(X_clust)
        sils.append(silhouette_score(X_clust,labs))
    fig_sil=px.line(x=list(range(2,11)),y=sils,markers=True,
                    labels={'x':'k','y':'Silhouette'},title="Silhouette Scores")
    st.plotly_chart(fig_sil)
    st.caption("Higher silhouette = better clusters.")

    # Persona table
    st.markdown("#### Cluster Personas")
    persona=clust_df.groupby("Cluster")[num_cols].mean().round(1)
    st.dataframe(persona.style.background_gradient(cmap="BuGn"),use_container_width=True)
    st.caption("Average profile per cluster.")

    csv_cl=clust_df.to_csv(index=False).encode()
    st.download_button("Download Clustered Data",csv_cl,"clustered_data.csv","text/csv")

# ========== TAB 4: ASSOCIATION RULE MINING ==========
with tabs[3]:
    st.markdown("## :bulb: Association Rule Mining")
    cols=['FoodPreferences','ChallengesFaced']
    selcol=st.selectbox("Select column",cols)
    ms=st.slider("Min Support",0.01,0.5,0.05,0.01)
    mc=st.slider("Min Confidence",0.1,1.0,0.3,0.05)
    trans=df[selcol].str.get_dummies(sep=',')
    freq=apriori(trans,min_support=ms,use_colnames=True)
    rules=association_rules(freq,metric="confidence",min_threshold=mc)
    rules=rules.sort_values("confidence",ascending=False).head(10)
    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])
    st.markdown("**Interpretation:** If antecedents, then consequents likely.")

# ========== TAB 5: REGRESSION ==========
with tabs[4]:
    st.markdown("## :chart_with_upwards_trend: Regression Insights")
    targets=['SpendPerVisitINR','MaxFoodSpendINR','OverallSatisfaction']
    feats=['Age','MonthlyIncomeINR','EventsAttendedAnnually','OverallSatisfaction']
    regs={"Linear":LinearRegression(),"Ridge":Ridge(),"Lasso":Lasso(),"Decision Tree":DecisionTreeRegressor(random_state=0)}
    res=[]
    for t in targets:
        y=df[t]
        X=df[feats]
        Xtr,Xts,ytr,yts=train_test_split(X,y,test_size=0.2,random_state=42)
        for n,m in regs.items():
            m.fit(Xtr,ytr)
            r2=m.score(Xts,yts)
            res.append([n,t,round(r2,3)])
    rdf=pd.DataFrame(res,columns=["Model","Target","R2"]).pivot("Model","Target","R2")
    st.dataframe(rdf,use_container_width=True)
    st.caption("R2 scores for regressors.")

    # Actual vs Predicted for Linear
    lin=LinearRegression().fit(Xtr,ytr)
    preds=lin.predict(Xts)
    fig_reg,ax_reg=plt.subplots()
    ax_reg.scatter(yts,preds,alpha=0.5)
    ax_reg.plot([yts.min(),yts.max()],[yts.min(),yts.max()],'r--')
    ax_reg.set_xlabel("Actual")
    ax_reg.set_ylabel("Predicted")
    st.pyplot(fig_reg)
    st.caption("Close to line = accurate.")

    st.markdown("#### Regression Takeaways")
    st.write(
        "- Income & satisfaction strong spend predictors\n"
        "- Ridge & Decision Trees often best\n"
        "- Outliers (luxury spenders) harder to predict"
    )
