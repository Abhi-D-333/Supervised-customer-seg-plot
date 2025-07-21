import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Supervised Segmentation", layout="wide")
st.title("🎯 Supervised Customer Segmentation Dashboard")
st.session_state.features = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency']

# ---------------------- HELPER ----------------------
def convert_review_to_text(val):
    try:
        val = float(val)
        if val >= 4.5:
            return "excellent"
        elif val >= 3.5:
            return "good"
        elif val >= 2.5:
            return "average"
        else:
            return "poor"
    except:
        return "unknown"

# ---------------------- STEP 1: TRAINING ----------------------
st.header("📥 Step 1: Upload Training Data")
train_file = st.file_uploader("Upload training CSV (must include Segment & Review column)", type=["csv"])

if train_file:
    df_train = pd.read_csv(train_file)
    st.write("📄 Training Data Preview", df_train.head())

    required_cols = st.session_state.features + ["Review", "Segment"]
    if all(col in df_train.columns for col in required_cols):
        df_train["ReviewText"] = df_train["Review"].fillna("").apply(convert_review_to_text)
        
        # TF-IDF Vectorizer
        st.session_state.tfidf = TfidfVectorizer(max_features=50)
        tfidf_matrix = st.session_state.tfidf.fit_transform(df_train["ReviewText"])
        tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        review_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names)

        numeric_features = df_train[st.session_state.features].fillna(0)
        X = pd.concat([review_df.reset_index(drop=True), numeric_features.reset_index(drop=True)], axis=1)
        y = df_train["Segment"]

        # Train classifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        st.session_state.model = model

        # Fit PCA for 2D projection
        st.session_state.pca = PCA(n_components=2)
        st.session_state.pca.fit(X)

        st.success("✅ Model trained successfully!")
    else:
        st.error(f"CSV must contain columns: {required_cols}")

# ---------------------- STEP 2: TESTING ----------------------
st.header("🔄 Step 2: Upload New Data to Predict Segments")
test_file = st.file_uploader("Upload new customer data (without Segment)", type=["csv"], key="test")

if test_file and st.session_state.get("model"):
    df_test = pd.read_csv(test_file)
    st.write("🧾 Test Data Preview", df_test.head())

    required_test_cols = st.session_state.features + ["Review"]
    if all(col in df_test.columns for col in required_test_cols):
        df_test["ReviewText"] = df_test["Review"].fillna("").apply(convert_review_to_text)
        tfidf_matrix = st.session_state.tfidf.transform(df_test["ReviewText"])
        review_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names)

        numeric_features = df_test[st.session_state.features].fillna(0)
        X_new = pd.concat([review_df.reset_index(drop=True), numeric_features.reset_index(drop=True)], axis=1)

        predictions = st.session_state.model.predict(X_new)
        df_test["Predicted Segment"] = predictions

        # PCA projection
        pca_components = st.session_state.pca.transform(X_new)
        df_test["PC1"] = pca_components[:, 0]
        df_test["PC2"] = pca_components[:, 1]

        st.success("🎯 Segments predicted successfully!")
        st.subheader("🔎 Prediction Result Sample")
        st.write(df_test.head())

        # Visualization
        st.subheader("🌀 Visualizing Predicted Segments")
        fig, ax = plt.subplots(figsize=(10, 6))

        segments = df_test["Predicted Segment"].unique()
        colors = cm.get_cmap("tab10", len(segments))

        for i, segment in enumerate(sorted(segments)):
            subset = df_test[df_test["Predicted Segment"] == segment]
            ax.scatter(
                subset["PC1"],
                subset["PC2"],
                label=f"Segment {segment}",
                alpha=0.7,
                s=80,
                color=colors(i)
            )

        # Add centroids
        centroids = df_test.groupby("Predicted Segment")[["PC1", "PC2"]].mean()
        ax.scatter(centroids["PC1"], centroids["PC2"], s=150, c='black', marker='X', label='Centroids')

        ax.set_title("📊 PCA Projection of Predicted Segments", fontsize=14)
        ax.set_xlabel("Principal Component 1", fontsize=12)
        ax.set_ylabel("Principal Component 2", fontsize=12)
        ax.legend(title="Segments", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

        st.pyplot(fig)

        # Segment-wise summary
        st.subheader("📈 Segment-wise Summary")
        st.write(df_test.groupby("Predicted Segment")[st.session_state.features].mean().round(2))
    else:
        st.error(f"Test CSV must contain: {required_test_cols}")
