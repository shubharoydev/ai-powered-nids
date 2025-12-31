import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load data (simulation mode by default)
def load_data(use_real=False):
    if use_real:
        # For production: Uncomment and replace with your CSV file
        # df = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
        st.write("Real data loading not implemented in this sample. Use simulation.")
        df = pd.DataFrame()  # Placeholder
    else:
        # Simulation: Generate synthetic data
        np.random.seed(42)
        data = np.random.rand(1000, 5)  # 1000 samples, 5 features (e.g., packet size, duration, etc.)
        labels = np.random.choice(['Normal', 'Intrusion'], 1000)
        df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
        df['Label'] = labels
    return df

# Streamlit Dashboard
st.title("AI-Powered Network Intrusion Detection System (NIDS)")

st.sidebar.header("Controls")
train_button = st.sidebar.button("Train Model Now")

# Load data
df = load_data()

if not df.empty:
    st.write("Data Overview:", df.head())

    # Train model if button clicked
    if train_button:
        X = df.drop('Label', axis=1)
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")
        
        # Confusion Matrix Visualization
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        
        st.write("Classification Report:", classification_report(y_test, y_pred))

# Live Simulation Section
st.header("Live Traffic Simulator")
feature1 = st.number_input("Feature1 (e.g., Packet Size)", 0.0, 1.0)
feature2 = st.number_input("Feature2 (e.g., Duration)", 0.0, 1.0)
# Add more inputs as needed...

if st.button("Detect Intrusion"):
    # Placeholder prediction (train model first in real use)
    st.write("Simulated Prediction: Normal")  # Replace with actual model.predict