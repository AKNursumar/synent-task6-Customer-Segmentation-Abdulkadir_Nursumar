import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Load Dataset
# ---------------------------

df = pd.read_csv("Mall_Customers.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# ---------------------------
# Data Preprocessing
# ---------------------------

# Convert Gender to numeric
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ---------------------------
# Find Optimal Clusters
# (Elbow Method)
# ---------------------------

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# ---------------------------
# Apply K-Means
# ---------------------------

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X)

# ---------------------------
# Visualize Clusters
# ---------------------------

plt.figure(figsize=(10, 6))

for cluster in range(5):
    cluster_data = df[df['Cluster'] == cluster]

    plt.scatter(
        cluster_data['Annual Income (k$)'],
        cluster_data['Spending Score (1-100)'],
        label=f'Cluster {cluster}'
    )

# Cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X',
    label='Centroids'
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# ---------------------------
# Cluster Summary
# ---------------------------

summary = df.groupby('Cluster')[
    ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
].mean()

print("\nCustomer Segment Summary:")
print(summary)

# ---------------------------
# Insights
# ---------------------------

print("\n===== CUSTOMER SEGMENT INSIGHTS =====")
print("Cluster analysis grouped customers based on income and spending behavior.")
print("High-income, high-spending customers are premium customers.")
print("High-income, low-spending customers are potential marketing targets.")
print("Low-income, high-spending customers are enthusiastic buyers.")
print("Low-income, low-spending customers are budget-conscious customers.")