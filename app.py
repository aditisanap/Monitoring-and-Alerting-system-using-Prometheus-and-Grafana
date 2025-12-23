import streamlit as st 
import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp
from prometheus_client import start_http_server, Gauge, Histogram, Counter, CollectorRegistry
import joblib
import time

st.title("Simple Ml monitoring demo")

@st.cache_resource
def setup_metrics_server():
    registry = CollectorRegistry()

    try:
        start_http_server(8000, registry = registry)
    except OSError:
        pass
    m_acc = Gauge('model_accuracy', 'Model accuracy on recent data', registry=registry)
    m_drift = Gauge('data_drift_ks_score', 'KS statistic for data drift', registry=registry)
    m_lat = Histogram('prediction_latency_seconds','Prediction time', registry=registry)
    m_count = Counter('prediction_total', 'Total prediction made', registry=registry)

    return m_acc, m_drift, m_lat, m_count
model_accuracy, data_drift_ks, prediction_latency, predictions_total = setup_metrics_server()

@st.cache_resource
def load_model():
    iris = load_iris()
    X_train, y_train = iris.data, iris.target
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train.mean(axis=0), X_train.std(axis=0)

model, ref_mean, ref_std = load_model()

def generated_data(drift=False, n=50):
    if drift:
        data = np.random.normaL(ref_mean + [1.0,0,0,0], ref_std, (n,4))
    else:
        data = np.random.normal(ref_mean, ref_std, (n,4))
    labels = np.random.choice([0,1,2], n)
    return data, labels

st.info("Make prediction = See metrics update = Check drift. View in Grafana at http://localhost:3000")

drift_enabled = st.checkbox("Simulate Data Drift")

if st.button("**Generate & Predict** (50 samples)"):
    data, true_labels = generated_data(drift=drift_enabled)
    df = pd.DataFrame(data, columns=['sepal_l','sepal_w', 'petal_l', 'petal_w'])

    start_time = time.time()
    preds = model.predict(data)
    latency = time.time() - start_time

    acc = accuracy_score(true_labels, preds)

    ks_scores = [ks_2samp(ref_mean, col)[0] for col in data.T]
    avg_ks = np.mean(ks_scores) 

    model_accuracy.set(acc)
    data_drift_ks.set(avg_ks)
    prediction_latency.observe(latency)
    predictions_total.inc(len(preds))

    st.metric("Accuracy", f"{acc:.3f}")
    st.metric("Avg KS Drift", f"{avg_ks:.3f}")
    st.metric("Latency (s)", f"{latency:.4f}")

    st.success("Metrics Updated! Refresh http://localhost:8000/metrics to see values.")

if st.button("Auto Predict (5x)"):
    st.write("Running...")
    for i in range(5):
        data, true_labels = generated_data(drift=False)
        start_time = time.time()
        preds = model.predict(data)
        latency = time.time() - start_time

        acc = accuracy_score(true_labels, preds)
        ks_scores = [ks_2samp(ref_mean, col)[0] for col in data.T]
        avg_ks = np.mean(ks_scores)

        model_accuracy.set(acc)
        data_drift_ks.set(avg_ks)
        prediction_latency.observe(latency)
        predictions_total.inc(len(preds))

        time.sleep(0.2)
    st.success("Done! Metrics sent to Prometheus.")

st.markdown("---")
st.markdown("""
***Live Check:**
- [Streamlit](localhost:8501)
- [Metrics Raw] (http://localhost:8000/metrics)
- [Grafana] (http://localhost:3000)
""")