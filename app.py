import streamlit as st [cite: 11]
import numpy as np [cite: 13]
import matplotlib.pyplot as plt [cite: 15]
from sklearn.neighbors import KNeighborsClassifier [cite: 17]

st.set_page_config(page_title="OWN Weather Classification", layout="wide") [cite: 20]
st.title(" K-Nearest Neighbor Weather Classification") [cite: 22]


X = np.array([[25, 80], [27, 60], [31, 65], [23, 85], [20, 75]]) 
y = np.array([0, 1, 0, 0, 1, 1]) 
labels = {0: "Sunny", 1: "Rainy"} [cite: 27]

st.sidebar.header("Input Parameters") [cite: 37]
temp = st.sidebar.slider("Temperature (°C)", 18, 60, 26) [cite: 38]
hum = st.sidebar.slider("Humidity (%)", 58, 95, 78) [cite: 39]
k = st.sidebar.slider("K value", 1, 5, 3) [cite: 39]

model = KNeighborsClassifier(n_neighbors=k) [cite: 40]
model.fit(X, y) [cite: 40]
pred = model.predict([[temp, hum]])[0] [cite: 41]

st.sidebar.success(f"Prediction: {labels[pred]}") [cite: 44]

fig, ax = plt.subplots() [cite: 47]
ax.scatter(X[y==0, 0], X[y==0, 1], label="Sunny", s=100) [cite: 49, 50]
ax.scatter(X[y==1, 0], X[y==1, 1], label="Rainy", s=100) [cite: 52]
ax.scatter(temp, hum, marker="*", s=300, label="New Day") [cite: 54]
ax.set_xlabel("Temperature") [cite: 56]
ax.set_ylabel("Humidity") [cite: 58]
ax.legend() [cite: 60]
ax.grid(True) [cite: 62]
st.pyplot(fig) [cite: 65]
plt.close(fig) [cite: 67]
