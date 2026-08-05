# Student Performance Predictor

A machine Learning project that predicts a student's expected examination score based on academic and lifestyle factors using a **Random Forest Regressor**. The project includes a complete machine learning pipeline, model evaluation, and an interactive **Streamlit** web application for real-time predictions.

---

## Project Overview

Student academic performance depends on multiple factors beyond study time alone. This project leverages machine learning to estimate a student's expected score using:

- Study Hours per Day
- Sleep Hours per Day
- Attendance Percentage

The model is trained on a structured dataset and deployed through an intuitive Streamlit interface that allows users to experiment with different inputs and instantly view predicted scores.

---

## Features

- Data preprocessing and validation
- Random Forest Regression model
- Pipeline-based model training
- Automatic feature scaling
- Model evaluation using:
  - Mean Absolute Error (MAE)
  - R² Score
- Model serialization using Joblib
- Interactive Streamlit dashboard
- Automatic model generation if no trained model exists
- Prediction visualization
- Personalized study recommendations

---

## Project Structure

```text
Student_Performance_Predictor/
│
├── app.py                          # Streamlit Web Application
├── requirements.txt                # Project dependencies
├── README.md
│
├── data/
│   └── student_performance.csv     # Dataset
│
├── models/
│   ├── student_score_model.joblib  # Trained model
│   ├── model_metrics.txt           # Evaluation metrics
│   └── prediction_plot.png         # Actual vs Predicted plot
│
└── src/
    ├── train_model.py              # Model training pipeline
    └── __init__.py
```

---

## Machine Learning Workflow

```
Dataset
    │
    ▼
Data Validation
    │
    ▼
Feature Selection
    │
    ▼
Train-Test Split
    │
    ▼
StandardScaler
    │
    ▼
Random Forest Regressor
    │
    ▼
Model Evaluation
    │
    ▼
Save Model & Metrics
    │
    ▼
Streamlit Prediction App
```

---

## Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python 3 |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Web Application | Streamlit |
| Model Storage | Joblib |

---

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/StudentPerformancePredictor.git

cd StudentPerformancePredictor
```

### Create Virtual Environment (Recommended)

**Windows**

```bash
python -m venv .venv

.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run:

```bash
python src/train_model.py
```

This generates:

- `student_score_model.joblib`
- `model_metrics.txt`
- `prediction_plot.png`

inside the `models/` directory.

---

## Run the Streamlit Application

```bash
streamlit run app.py
```

After launching, open the local URL displayed in your terminal (typically `http://localhost:8501`).

---

## Input Features

| Feature | Description |
|----------|-------------|
| Study Hours | Average daily study time |
| Sleep Hours | Average daily sleep duration |
| Attendance | Attendance percentage |

---

## Output

The application predicts:

- Estimated Examination Score
- Input Summary
- Personalized Study Recommendation

---

## Model Performance

Current model evaluation:

| Metric | Score |
|---------|-------|
| Mean Absolute Error (MAE) | **1.82** |
| R² Score | **0.984** |

These metrics indicate excellent predictive performance on the available dataset.

---

## Model Used

**Random Forest Regressor**

Reasons for selection:

- Handles non-linear relationships effectively
- Robust against overfitting
- High prediction accuracy
- Works well on structured tabular datasets
- Requires minimal feature engineering

---

## Learning Outcomes

This project demonstrates practical implementation of:

- Machine Learning Pipeline
- Data Validation
- Feature Engineering
- Model Training
- Model Evaluation
- Model Serialization
- Interactive Dashboard Development
- Data Visualization
- Python Project Structuring

---

## Disclaimer

The dataset included in this project is synthetic and intended for educational purposes. Predictions should not be considered as real academic assessments. For production use, the model should be trained using real-world educational data.

---
