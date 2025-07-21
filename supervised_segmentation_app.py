import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Supervised Customer Segmentation", layout="wide")
st.title("🎯 Supervised Customer Segmentation Dashboard")

st.session_state.features = ["Age", "Income", "SpendingScore", "PurchaseFrequency"]

st.header("📁 Step 1: Upload Labeled Training Data")
train_file = st.file_uploader("Upload labeled customer CSV file (with 'Segment' column)", type=["csv"], key="train")

if train_file:
    df_train = pd.read_csv(train_file)
    st.write(df_train.head())

    required_cols = st.session_state.features + ["Review", "Segment"]
    if all(col in df_train.columns for col in required_cols):
        # Preprocessing
        st.session_state.tfidf = TfidfVectorizer(max_features=50)
        review_features = st.session_state.tfidf.fit_transform(df_train["Review"].fillna("")).toarray()
        review_df = pd.DataFrame(review_features, columns=[f"review_tfidf_{i}" for i in range(review_features.shape[1])])

        numeric_features = df_train[st.session_state.features].fillna(0)
        X = pd.concat([review_df, numeric_features.reset_index(drop=True)], axis=1)
        y = df_train["Segment"]

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        st.session_state.model = model

        st.session_state.pca = PCA(n_components=2)
        _ = st.session_state.pca.fit(X)

        st.success("✅ Model trained successfully!")
    else:
        st.error(f"CSV must contain: {required_cols}")

st.header("🔍 Step 2: Upload New Data to Predict Segments")
test_file = st.file_uploader("Upload new customer data (without Segment)", type=["csv"], key="test")

if test_file and st.session_state.model:
    df_test = pd.read_csv(test_file)
    st.write(df_test.head())

    required_test_cols = st.session_state.features + ["Review"]
    if all(col in df_test.columns for col in required_test_cols):
        review_features = st.session_state.tfidf.transform(df_test["Review"].fillna("")).toarray()
        review_df = pd.DataFrame(review_features, columns=[f"review_tfidf_{i}" for i in range(review_features.shape[1])])

        numeric_features = df_test[st.session_state.features].fillna(0)
        X_new = pd.concat([review_df, numeric_features.reset_index(drop=True)], axis=1)

        predictions = st.session_state.model.predict(X_new)
        df_test["PredictedSegment"] = predictions

        components = st.session_state.pca.transform(X_new)
        df_test["PC1"] = components[:, 0]
        df_test["PC2"] = components[:, 1]

        st.subheader("📊 Predicted Segments")
        st.write(df_test[st.session_state.features + ["Review", "PredictedSegment"]].head())

        st.subheader("🌀 Visualization (PCA Projection)")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_test["PC1"], df_test["PC2"], c=df_test["PredictedSegment"], cmap="tab10", s=60)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Predicted Customer Segments")
        st.pyplot(fig)
    else:
        st.error(f"Test CSV must contain: {required_test_cols}")
