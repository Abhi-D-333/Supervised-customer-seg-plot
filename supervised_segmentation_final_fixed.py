
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🎯 Supervised Customer Segmentation Dashboard")

# Define features
numeric_features = ["Age", "Income", "SpendingScore", "PurchaseFrequency"]
text_feature = "Review"
target_column = "Segment"

# Upload and Train
st.header("🧠 Step 1: Upload Labeled Customer Data (With Segment column)")
train_file = st.file_uploader("Upload training CSV", type=["csv"], key="train")
if train_file:
    df_train = pd.read_csv(train_file)
    st.write(df_train.head())

    if all(col in df_train.columns for col in numeric_features + [text_feature, target_column]):
        # Pipeline for preprocessing
        preprocessor = ColumnTransformer(transformers=[
            ("text", TfidfVectorizer(max_features=50), text_feature),
            ("num", StandardScaler(), numeric_features)
        ])

        # Full pipeline with classifier
        model_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ])

        # Train
        model_pipeline.fit(df_train[[text_feature] + numeric_features], df_train[target_column])
        st.session_state.model = model_pipeline
        st.session_state.pca = PCA(n_components=2)
        transformed = preprocessor.fit_transform(df_train[[text_feature] + numeric_features])
        st.session_state.pca.fit(transformed)
        st.success("✅ Model trained successfully!")

    else:
        st.error(f"CSV must contain columns: {numeric_features + [text_feature, target_column]}")

# Upload test data
st.header("📂 Step 2: Upload New Data to Predict Segments")
test_file = st.file_uploader("Upload new customer data (without Segment)", type=["csv"], key="test")
if test_file and "model" in st.session_state:
    df_test = pd.read_csv(test_file)
    if all(col in df_test.columns for col in numeric_features + [text_feature]):
        predictions = st.session_state.model.predict(df_test[[text_feature] + numeric_features])
        df_test["PredictedSegment"] = predictions

        # PCA for visualization
        transformed_test = st.session_state.model.named_steps["preprocessor"].transform(df_test[[text_feature] + numeric_features])
        pca_test = st.session_state.pca.transform(transformed_test)
        df_test["PC1"] = pca_test[:, 0]
        df_test["PC2"] = pca_test[:, 1]

        st.subheader("📊 Visualization (PCA Projection)")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_test["PC1"], df_test["PC2"], c=pd.Categorical(df_test["PredictedSegment"]).codes, cmap="tab10", s=60)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Customer Segments (Predicted)")
        st.pyplot(fig)

        st.subheader("📁 Predicted Output")
        st.dataframe(df_test)
        csv = df_test.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", data=csv, file_name="predicted_segments.csv", mime="text/csv")
    else:
        st.error(f"CSV must contain: {numeric_features + [text_feature]}")
