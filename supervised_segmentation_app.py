import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Supervised Segmentation", layout="wide")
st.title("🎯 Supervised Customer Segmentation Dashboard")

# Define features
st.session_state.features = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency']

# Step 1: Upload training data
st.header("📥 Step 1: Upload Training Data")
train_file = st.file_uploader("Upload training CSV (must include Segment & Review column)", type=["csv"])

if train_file:
    df_train = pd.read_csv(train_file)
    st.write("📄 Training Data Preview", df_train.head())

    required_cols = st.session_state.features + ["Review", "Segment"]
    if all(col in df_train.columns for col in required_cols):
        # Preprocess text and numerical data
        st.session_state.tfidf = TfidfVectorizer(max_features=50)
        review_features = st.session_state.tfidf.fit_transform(df_train["Review"].fillna("")).toarray()

        numeric_features = df_train[st.session_state.features].fillna(0)
        X = pd.concat([pd.DataFrame(review_features), numeric_features.reset_index(drop=True)], axis=1)
        y = df_train["Segment"]

        # Train classifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        st.session_state.model = model

        # Fit PCA
        st.session_state.pca = PCA(n_components=2)
        st.session_state.pca.fit(X)

        st.success("✅ Model trained successfully!")
    else:
        st.error(f"CSV must contain columns: {required_cols}")

# Step 2: Upload test data
st.header("🔄 Step 2: Upload New Data to Predict Segments")
test_file = st.file_uploader("Upload new customer data (without Segment)", type=["csv"], key="test")

if test_file and st.session_state.get("model"):
    df_test = pd.read_csv(test_file)
    st.write("🧾 Test Data Preview", df_test.head())

    # Check required columns
    required_test_cols = st.session_state.features + ["Review"]
    if all(col in df_test.columns for col in required_test_cols):
        review_features = st.session_state.tfidf.transform(df_test["Review"].fillna("")).toarray()
        numeric_features = df_test[st.session_state.features].fillna(0)
        X_new = pd.concat([pd.DataFrame(review_features), numeric_features.reset_index(drop=True)], axis=1)

        predictions = st.session_state.model.predict(X_new)
        df_test["Predicted Segment"] = predictions

        # Project PCA
        pca_components = st.session_state.pca.transform(X_new)
        df_test["PC1"] = pca_components[:, 0]
        df_test["PC2"] = pca_components[:, 1]

        st.success("🎯 Segments predicted successfully!")
        st.subheader("🔎 Prediction Result Sample")
        st.write(df_test.head())

        # Visualize
        st.subheader("🌀 Visualizing Predicted Segments")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_test["PC1"], df_test["PC2"], c=df_test["Predicted Segment"], cmap="tab10", s=60)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Segment Visualization (PCA Projection)")
        st.pyplot(fig)

        # Show segment summary
        st.subheader("📈 Segment-wise Summary")
        st.write(df_test.groupby("Predicted Segment")[st.session_state.features].mean().round(2))
    else:
        st.error(f"Test CSV must contain: {required_test_cols}")
