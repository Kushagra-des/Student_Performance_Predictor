from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.train_model import DATA_PATH, MODEL_PATH, PLOT_PATH, train_model, save_artifacts


st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="SP",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #f8fafc;
        }

        [data-testid="stHeader"] {
            background: rgba(248, 250, 252, 0.85);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        .hero {
            background: linear-gradient(135deg, #0f766e 0%, #2563eb 55%, #111827 100%);
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            font-size: 2.4rem;
            margin: 0 0 0.45rem;
            letter-spacing: 0;
        }

        .hero p {
            font-size: 1rem;
            margin: 0;
            max-width: 720px;
            color: #e0f2fe;
        }

        .result-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        }

        .score {
            font-size: 4rem;
            font-weight: 800;
            color: #0f766e;
            line-height: 1;
        }

        .score-label {
            color: #475569;
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        .insight {
            color: #334155;
            font-size: 1rem;
            line-height: 1.55;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def ensure_model_exists() -> None:
    if MODEL_PATH.exists():
        return

    model, metrics, results = train_model()
    save_artifacts(model, metrics, results)


@st.cache_resource
def load_model() -> dict:
    ensure_model_exists()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def get_study_tip(study_hours: float, sleep_hours: float, attendance: int, predicted_score: float) -> str:
    if attendance < 75:
        return "Attendance is the biggest risk here. Improving consistency in class should lift the score."
    if study_hours < 3:
        return "Study time is low. A steady daily revision habit can improve the prediction quickly."
    if sleep_hours < 6:
        return "Sleep is likely limiting performance. Better rest can support memory and focus."
    if predicted_score >= 85:
        return "This looks strong. Keep the routine consistent and focus on practice tests."
    return "The inputs are balanced. A little more focused study time should move the score upward."


bundle = load_model()
model = bundle["model"]
features = bundle["features"]
dataset = load_dataset()

st.markdown(
    """
    <div class="hero">
        <h1>Student Performance Predictor</h1>
        <p>Estimate exam scores from study hours, sleep hours, and attendance using a trained scikit-learn model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.95, 1.05], gap="large")

with left:
    st.subheader("Student Inputs")
    study_hours = st.slider("Study hours per day", 0.0, 10.0, 4.0, 0.5)
    sleep_hours = st.slider("Sleep hours per day", 3.0, 10.0, 7.0, 0.5)
    attendance = st.slider("Attendance percentage", 40, 100, 80, 1)

    input_data = pd.DataFrame(
        [[study_hours, sleep_hours, attendance]],
        columns=features,
    )

    predicted_score = float(model.predict(input_data)[0])
    predicted_score = max(0, min(100, predicted_score))
    tip = get_study_tip(study_hours, sleep_hours, attendance, predicted_score)

    st.markdown("### Input Summary")
    st.dataframe(input_data, use_container_width=True, hide_index=True)

with right:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="score">{predicted_score:.1f}</div>
            <div class="score-label">Predicted score out of 100</div>
            <hr>
            <p class="insight">{tip}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What-if Trend")
    study_range = pd.DataFrame(
        {
            "study_hours": [hour / 2 for hour in range(0, 21)],
            "sleep_hours": sleep_hours,
            "attendance": attendance,
        }
    )
    study_range["predicted_score"] = model.predict(study_range[features]).clip(0, 100)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(study_range["study_hours"], study_range["predicted_score"], color="#0f766e", linewidth=3)
    ax.scatter([study_hours], [predicted_score], color="#f97316", s=90, zorder=3)
    ax.set_xlabel("Study hours per day")
    ax.set_ylabel("Predicted score")
    ax.set_ylim(30, 100)
    ax.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)

st.divider()

tab_data, tab_model = st.tabs(["Dataset", "Model"])

with tab_data:
    st.dataframe(dataset, use_container_width=True, hide_index=True)

with tab_model:
    metric_cols = st.columns(3)
    metric_cols[0].metric("Training rows", len(dataset))
    metric_cols[1].metric("Average score", f"{dataset['score'].mean():.1f}")
    metric_cols[2].metric("Feature count", len(features))

    if Path(PLOT_PATH).exists():
        st.image(str(PLOT_PATH), caption="Actual vs predicted scores from the latest training run")
    else:
        st.info("Train the model to generate the evaluation plot.")
