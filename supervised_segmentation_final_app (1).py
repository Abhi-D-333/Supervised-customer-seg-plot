
# supervised_segmentation_app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Supervised Customer Segmentation", layout="wide")
st.title("🎯 Supervised Customer Segmentation Dashboard")

# Step 1: Upload labeled training data
st.header("📥 Step 1: Upload Labeled Customer Data (for Training)")
train_file = st.file_uploader("Upload labeled customer CSV file (with 'Segment' column)", type=["csv"], key="train")

if train_file:
    df_train = pd.read_csv(train_file)
    st.write(df_train.head())

    # Required columns
    st.session_state.features = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency']
    required_cols = st.session_state.features + ["Review", "Segment"]

    if all(col in df_train.columns for col in required_cols):
        # Preprocessing
        st.session_state.tfidf = TfidfVectorizer(max_features=50)
        review_features = st.session_state.tfidf.fit_transform(df_train["Review"]).toarray()
        numeric_features = df_train[st.session_state.features].fillna(0)
        X = pd.concat([pd.DataFrame(review_features), numeric_features.reset_index(drop=True)], axis=1)
        y = df_train["Segment"]

        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        st.session_state.model = model

        # Fit PCA for visualization
        st.session_state.pca = PCA(n_components=2)
        _ = st.session_state.pca.fit(X)

        st.success("✅ Model trained successfully!")
    else:
        st.error(f"CSV must contain: {required_cols}")

# Step 2: Upload new data for prediction
st.header("📥 Step 2: Upload New Data to Predict Segments")
test_file = st.file_uploader("Upload new customer data (without Segment)", type=["csv"], key="test")

if test_file and st.session_state.get("model"):
    df_test = pd.read_csv(test_file)

    if all(col in df_test.columns for col in st.session_state.features + ["Review"]):
        # Preprocess test data
        review_features_test = st.session_state.tfidf.transform(df_test["Review"]).toarray()
        numeric_features_test = df_test[st.session_state.features].fillna(0)
        X_new = pd.concat([pd.DataFrame(review_features_test), numeric_features_test.reset_index(drop=True)], axis=1)

        # Predict
        predictions = st.session_state.model.predict(X_new)
        df_test["PredictedSegment"] = predictions

        # PCA transform for visualization
        components = st.session_state.pca.transform(X_new)
        df_test["PC1"] = components[:, 0]
        df_test["PC2"] = components[:, 1]

        # Show results
        st.subheader("🔍 Predicted Segments")
        st.write(df_test.head())

        # Visualization
        st.subheader("🌀 Visualization (PCA Projection)")
        color_labels, _ = pd.factorize(df_test["PredictedSegment"])
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_test["PC1"], df_test["PC2"], c=color_labels, cmap="tab10", s=60)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Predicted Customer Segments")
        st.pyplot(fig)

    else:
        st.error("Test CSV must contain: Review + " + ", ".join(st.session_state.features))
