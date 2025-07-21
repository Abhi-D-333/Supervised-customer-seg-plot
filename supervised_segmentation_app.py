
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Supervised Customer Segmentation", layout="wide")
st.title("🎯 Supervised Customer Segmentation")

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None
if "pca" not in st.session_state:
    st.session_state.pca = None
if "features" not in st.session_state:
    st.session_state.features = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency']

st.header("📁 Step 1: Upload Training Data (with Segment)")
train_file = st.file_uploader("Upload training CSV", type=["csv"], key="train")

if train_file:
    df_train = pd.read_csv(train_file)
    st.write(df_train.head())

    required_cols = st.session_state.features + ["Review", "Segment"]
    if all(col in df_train.columns for col in required_cols):
        # Preprocessing
        st.session_state.tfidf = TfidfVectorizer(max_features=50)
        review_features = st.session_state.tfidf.fit_transform(df_train["Review"]).toarray()
        numeric_features = df_train[st.session_state.features].fillna(0)
        review_df = pd.DataFrame(review_features, columns=[f"tfidf_{i}" for i in range(review_features.shape[1])])
        X = pd.concat([review_df, numeric_features.reset_index(drop=True)], axis=1)

        y = df_train["Segment"]

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        st.session_state.model = model

        # Fit PCA for visualization
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

    if all(col in df_test.columns for col in st.session_state.features + ["Review"]):
        review_features = st.session_state.tfidf.transform(df_test["Review"]).toarray()
        numeric_features = df_test[st.session_state.features].fillna(0)
        X_new = pd.concat([pd.DataFrame(review_features), numeric_features.reset_index(drop=True)], axis=1)

        predictions = st.session_state.model.predict(X_new)
        df_test["PredictedSegment"] = predictions

        st.subheader("📊 Predicted Segments")
        st.write(df_test.head())

        # PCA Projection
        X_proj = st.session_state.pca.transform(X_new)
        df_test["PC1"] = X_proj[:, 0]
        df_test["PC2"] = X_proj[:, 1]

        fig, ax = plt.subplots()
        scatter = ax.scatter(df_test["PC1"], df_test["PC2"], c=df_test["PredictedSegment"].astype("category").cat.codes, cmap="tab10", s=60)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("PCA Projection of Predicted Segments")
        st.pyplot(fig)

        # Summary
        st.subheader("📈 Segment Statistics")
        summary = df_test.groupby("PredictedSegment")[st.session_state.features].mean().round(2)
        st.dataframe(summary)

        # Download option
        st.download_button("📥 Download Predicted Data", data=df_test.to_csv(index=False), file_name="predicted_segments.csv", mime="text/csv")
    else:
        st.error("Uploaded data must contain all required columns including 'Review'")
elif test_file:
    st.warning("Please train the model first by uploading training data.")
