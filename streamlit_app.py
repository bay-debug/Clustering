
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st
import folium
from streamlit_folium import folium_static

# Baca data
df = pd.read_csv('datasetgabunganbencanacsv.csv')

# Preprocessing
features = ['Tanah Longsor', 'Banjir', 'Gempa Bumi']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA untuk Reduksi Dimensi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Streamlit layout
st.title("Pengelompokan Daerah Rawan Bencana")
st.sidebar.title("Clustering Settings")

# Slider for number of clusters
n_clusters = st.sidebar.slider("Select number of clusters", 2, 5, 2)

# Clustering dengan K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_pca)

# Calculate silhouette score
silhouette_avg = silhouette_score(X_pca, df['Cluster'])
st.sidebar.write(f"Silhouette Score: {silhouette_avg}")

# Assigning Resiko categori to clusters
Resiko_categori = ['Rendah', 'Tinggi', 'Sedang', 'Sangat Tinggi', 'Parah']
df['Resiko Level'] = df['Cluster'].apply(lambda x: Resiko_categori[x])

# Display DataFrame
st.subheader("Data Terklaster dengan Tingkat Resiko")
st.dataframe(df)

# Visualisasi Hasil Clustering
# st.subheader("Hasil Pengelompokan PCA")
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clustering Hasil PCA')
plt.colorbar(label='Cluster')
# st.pyplot(plt)

# Create a map centered at the specified coordinates with zoom capabilities
map_center = [-7.150975, 110.1402594]
mymap = folium.Map(location=map_center, zoom_start=10, control_scale=True)

# Warna untuk setiap cluster
colors = {0: 'green', 1: 'red', 2: 'yellow', 3: 'blue', 4: 'orange'}

# Menambahkan titik ke peta
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=colors[row['Cluster']],
        fill=True,
        fill_color=colors[row['Cluster']],
        fill_opacity=0.6,
        popup=f"{row['Kabupaten / Kota']} - Cluster: {row['Cluster']} - Resiko Level: {row['Resiko Level']}"
    ).add_to(mymap)

# Menampilkan peta di Streamlit
st.subheader("Peta Wilayah Berkelompok dengan Tingkat Resiko")
folium_static(mymap)
