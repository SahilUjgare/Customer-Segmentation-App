import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

st.title("🛍️ Customer Segmentation App")

# Upload dataset
file = st.file_uploader("Upload your dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Raw Data")
    st.write(df.head())

    # ---------------- EDA ----------------
    st.subheader("📈 Exploratory Data Analysis")

    st.write("### Summary Statistics")
    st.write(df.describe())

    # Gender distribution
    st.write("### Gender Distribution")
    st.bar_chart(df['Gender'].value_counts())

    # Histograms
    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['Age'])
    st.pyplot(fig)

    st.write("### Income vs Spending Score")
    fig, ax = plt.subplots()
    ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
    ax.set_xlabel("Income")
    ax.set_ylabel("Spending Score")
    st.pyplot(fig)

    # ---------------- Preprocessing ----------------
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # ---------------- K-Means ----------------
    st.subheader("🎯 K-Means Clustering")

    k = st.slider("Select number of clusters", 2, 10, 5)

    kmeans = KMeans(n_clusters=k)
    df['Cluster'] = kmeans.fit_predict(X)

    # Plot clusters
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=df['Cluster']
    )
    ax.set_xlabel("Income")
    ax.set_ylabel("Spending Score")
    st.pyplot(fig)

    # Silhouette Score
    score = silhouette_score(X, df['Cluster'])
    st.write(f"### ✅ Silhouette Score: {score:.2f}")

    # ---------------- Insights ----------------
    st.subheader("💡 Business Insights")

    cluster_summary = df.groupby('Cluster').mean()
    st.write(cluster_summary)

    st.write("""
    🔍 Interpretation:
    - High income + high spending → VIP customers
    - High income + low spending → Target customers
    - Low income + high spending → Impulsive buyers
    """)
