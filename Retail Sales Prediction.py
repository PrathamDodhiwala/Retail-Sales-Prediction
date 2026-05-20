import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Retail Sales Prediction Dashboard", layout="wide")

st.title("🛒 Real-World Retail Sales Analysis & Prediction")

st.write(
    "Upload a retail sales dataset to analyze trends, visualize insights, and predict future sales."
)

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📂 Raw Dataset")

    st.dataframe(df.head())

    st.subheader("📌 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("🧹 Data Cleaning")

    duplicates = df.duplicated().sum()
    df.drop_duplicates(inplace=True)

    numeric_cols = df.select_dtypes(include=np.number).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    st.success(f"Removed {duplicates} duplicate rows")

    st.subheader("📊 Statistical Summary")

    st.dataframe(df.describe())

    st.subheader("📈 Sales Visualization")

    if len(numeric_cols) > 0:

        selected_column = st.selectbox("Select Numeric Column", numeric_cols)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(df[selected_column])

        ax.set_title(f"{selected_column} Trend")
        ax.set_xlabel("Index")
        ax.set_ylabel(selected_column)

        st.pyplot(fig)

    st.subheader("📉 Histogram")

    if len(numeric_cols) > 0:

        hist_column = st.selectbox(
            "Select Column for Histogram", numeric_cols, key="hist"
        )

        fig2, ax2 = plt.subplots(figsize=(8, 5))

        ax2.hist(df[hist_column], bins=20)

        ax2.set_title(f"Histogram of {hist_column}")
        ax2.set_xlabel(hist_column)
        ax2.set_ylabel("Frequency")

        st.pyplot(fig2)

    st.subheader("🔥 Correlation Heatmap")

    if len(numeric_cols) > 1:

        correlation = df[numeric_cols].corr()

        fig3, ax3 = plt.subplots(figsize=(10, 6))

        im = ax3.imshow(correlation)

        ax3.set_xticks(range(len(correlation.columns)))
        ax3.set_yticks(range(len(correlation.columns)))

        ax3.set_xticklabels(correlation.columns, rotation=90)
        ax3.set_yticklabels(correlation.columns)

        plt.colorbar(im)

        st.pyplot(fig3)

    st.subheader(" Sales Prediction Model")

    if len(numeric_cols) >= 2:

        target_column = st.selectbox("Select Target Column", numeric_cols, key="target")

        feature_columns = st.multiselect(
            "Select Feature Columns",
            [col for col in numeric_cols if col != target_column],
        )

        if len(feature_columns) > 0:

            X = df[feature_columns]
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = LinearRegression()

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            col4, col5 = st.columns(2)

            with col4:
                st.metric("MAE", round(mae, 2))

            with col5:
                st.metric("R² Score", round(r2, 2))

            st.subheader("📌 Actual vs Predicted")

            fig4, ax4 = plt.subplots(figsize=(10, 5))

            ax4.plot(y_test.values, label="Actual")
            ax4.plot(predictions, label="Predicted")

            ax4.legend()

            st.pyplot(fig4)

    st.subheader("⬇ Download Cleaned Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="cleaned_retail_data.csv",
        mime="text/csv",
    )

else:
    st.info(" Upload a CSV dataset to begin analysis.")
