"""
House Price Prediction - Streamlit Web Application
Interactive dashboard as described in the Mini Project Report
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths & load artifacts
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "housing.csv")


@st.cache_resource
def load_models():
    lr = joblib.load(os.path.join(MODELS_DIR, "linear_regression.pkl"))
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    meta = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))
    return lr, rf, meta


@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)


lr_model, rf_model, metadata = load_models()
raw_df = load_raw_data()

# ---------------------------------------------------------------------------
# Sidebar - User Inputs
# ---------------------------------------------------------------------------
st.sidebar.title("🏠 Property Details")
st.sidebar.markdown("Enter the features of the house to get a price prediction.")

# Numeric inputs (aligned with California housing features)
longitude = st.sidebar.slider("Longitude", -124.5, -114.0, -119.5, 0.01)
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 35.6, 0.01)
housing_median_age = st.sidebar.slider("Housing Median Age (years)", 1, 52, 28)
total_rooms = st.sidebar.number_input("Total Rooms", min_value=1, max_value=40000, value=50)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", min_value=1, max_value=7000, value=40)
population = st.sidebar.number_input("Population (block)", min_value=1, max_value=40000, value=100)
households = st.sidebar.number_input("Households", min_value=1, max_value=7000, value=40)
median_income = st.sidebar.slider("Median Income (× $10,000)", 0.5, 15.0, 3.5, 0.1)

ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    options=["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"],
    index=0,
)

model_choice = st.sidebar.radio(
    "Select Model",
    options=["Random Forest (Recommended)", "Linear Regression", "Both"],
    index=0,
)

predict_btn = st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("🏠 House Price Prediction ")
st.markdown(
    """
    This interactive application predicts median house values based on property and location attributes  
    using **Linear Regression** and **Random Forest Regressor** models trained on the California Housing dataset.
    """
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Prediction", "📈 Model Performance", "🔍 Data Explorer", "ℹ️ About"]
)

# ===================== TAB 1: Prediction =====================
with tab1:
    if predict_btn:
        # Build input DataFrame (must match training feature order / names)
        input_data = pd.DataFrame(
            {
                "longitude": [longitude],
                "latitude": [latitude],
                "housing_median_age": [housing_median_age],
                "total_rooms": [total_rooms],
                "total_bedrooms": [total_bedrooms],
                "population": [population],
                "households": [households],
                "median_income": [median_income],
                "ocean_proximity": [ocean_proximity],
            }
        )

        # Feature engineering (same as training)
        input_data["rooms_per_household"] = input_data["total_rooms"] / input_data["households"]
        input_data["bedrooms_per_room"] = input_data["total_bedrooms"] / input_data["total_rooms"]
        input_data["population_per_household"] = input_data["population"] / input_data["households"]

        # Ensure column order matches training
        feature_cols = metadata["numeric_features"] + metadata["categorical_features"]
        input_data = input_data[feature_cols]

        # Predictions
        col1, col2 = st.columns(2)

        if model_choice in ["Linear Regression", "Both"]:
            pred_lr = lr_model.predict(input_data)[0]
            with col1:
                st.metric(
                    label="Linear Regression Prediction",
                    value=f"${pred_lr:,.0f}",
                    help=f"R² on test set: {metadata['lr_metrics']['r2']:.3f}",
                )

        if model_choice in ["Random Forest (Recommended)", "Both"]:
            pred_rf = rf_model.predict(input_data)[0]
            with col2 if model_choice == "Both" else col1:
                st.metric(
                    label="Random Forest Prediction",
                    value=f"${pred_rf:,.0f}",
                    delta=f"R² {metadata['rf_metrics']['r2']:.3f}",
                    help="Usually more accurate for non-linear relationships",
                )

        st.success("Prediction completed! Adjust the sidebar inputs and click Predict again.")

        # Simple gauge-like visualization
        if model_choice != "Both":
            final_pred = pred_rf if "Random" in model_choice else pred_lr
        else:
            final_pred = (pred_lr + pred_rf) / 2
            st.info(f"Average of both models: **${final_pred:,.0f}**")

        # Comparison with dataset median
        dataset_median = raw_df["median_house_value"].median()
        st.markdown(f"**Dataset median house value:** ${dataset_median:,.0f}")

    else:
        st.info("👈 Enter property details in the sidebar and click **Predict Price**.")
        st.image(os.path.join(MODELS_DIR, "evaluation_plots.png"), caption="Model Evaluation Plots (from training)")

# ===================== TAB 2: Model Performance =====================
with tab2:
    st.header("Model Evaluation Metrics")

    metrics_df = pd.DataFrame(
        {
            "Metric": ["MAE ($)", "RMSE ($)", "R² Score"],
            "Linear Regression": [
                f"{metadata['lr_metrics']['mae']:,.2f}",
                f"{metadata['lr_metrics']['rmse']:,.2f}",
                f"{metadata['lr_metrics']['r2']:.4f}",
            ],
            "Random Forest": [
                f"{metadata['rf_metrics']['mae']:,.2f}",
                f"{metadata['rf_metrics']['rmse']:,.2f}",
                f"{metadata['rf_metrics']['r2']:.4f}",
            ],
        }
    )
    st.table(metrics_df)

    st.markdown(
        """
        - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices.  
        - **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily.  
        - **R² Score**: Proportion of variance explained by the model (closer to 1.0 is better).
        """
    )

    st.subheader("Evaluation Visualizations")
    if os.path.exists(os.path.join(MODELS_DIR, "evaluation_plots.png")):
        st.image(
            os.path.join(MODELS_DIR, "evaluation_plots.png"),
            use_container_width=True,
            caption="Actual vs Predicted | Residuals | Feature Importance",
        )

    st.markdown(
        """
        **Key Observations (from report):**
        - Random Forest consistently outperforms Linear Regression on non-linear price patterns.
        - Strongest predictors are typically median income, location (ocean proximity / coordinates), and rooms-related features.
        - Linear Regression remains a transparent and fast baseline.
        """
    )

# ===================== TAB 3: Data Explorer =====================
with tab3:
    st.header("Dataset Overview")
    st.markdown(
        f"""
        **California Housing Dataset** (1990 Census block groups)  
        - Rows: {len(raw_df):,}  
        - Features: longitude, latitude, housing_median_age, total_rooms, total_bedrooms,  
          population, households, median_income, ocean_proximity  
        - Target: median_house_value
        """
    )

    st.subheader("Sample Data")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("Price Distribution")
    fig = px.histogram(
        raw_df,
        x="median_house_value",
        nbins=50,
        title="Distribution of Median House Values",
        labels={"median_house_value": "Median House Value ($)"},
        color_discrete_sequence=["#1f77b4"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Median House Value by Ocean Proximity")
    fig2 = px.box(
        raw_df,
        x="ocean_proximity",
        y="median_house_value",
        title="House Value vs Ocean Proximity",
        color="ocean_proximity",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Heatmap (Numeric Features)")
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
    corr = raw_df[numeric_cols].corr()
    fig3 = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ===================== TAB 4: About =====================
with tab4:
    st.header("About This Project")
    st.markdown(
        """
        ### Mini Project Report Summary
        **Title:** House Price Prediction Using Machine Learning  
        **Submitted by:** Varun Mishra  
        **University:** Dr. A.P.J. Abdul Kalam Technical University, Lucknow  
        **College:** United College of Engineering & Research, Prayagraj  

        ### Objectives Achieved
        - ✅ Data collection from publicly available housing dataset  
        - ✅ Preprocessing: missing-value imputation, categorical encoding, feature scaling  
        - ✅ Feature engineering (rooms per household, bedrooms ratio, population density)  
        - ✅ Training & comparison of Linear Regression and Random Forest Regressor  
        - ✅ Evaluation with MAE, RMSE, and R²  
        - ✅ Visualizations of actual-vs-predicted, residuals, and feature importance  
        - ✅ Interactive Streamlit web application for end-user predictions  

        ### Tools & Technologies
        | Library / Tool       | Purpose                                      |
        |----------------------|----------------------------------------------|
        | Python 3             | Core language                                |
        | pandas / NumPy       | Data manipulation                            |
        | scikit-learn         | Models, preprocessing, metrics               |
        | matplotlib / Plotly  | Visualizations                               |
        | Streamlit            | Interactive web deployment                   |
        | joblib               | Model persistence                            |

        ### Future Scope (from report)
        - Gradient Boosting / XGBoost / LightGBM  
        - Neural network architectures  
        - Geospatial features & neighbourhood statistics  
        - Real-time market data integration  
        - Cloud deployment with authentication  

        ---
        *This application is a practical implementation of the end-to-end machine-learning pipeline described in the internship mini-project report.*
        """
    )

# Footer
st.markdown("---")
st.caption("House Price Prediction | Mini Project Internship Report Implementation | 2024-25")
