import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Simulate Hypothetical Logistics Dataset
np.random.seed(42)
n_samples = 100

data = {
    'Shipment_ID': [f'SHP_{i:03d}' for i in range(1, n_samples + 1)],
    'Route_Type': np.random.choice(['Urban', 'Suburban', 'Intercity'], size=n_samples, p=[0.5, 0.3, 0.2]),
    'Shipment_Volume_m3': np.random.uniform(5, 30, size=n_samples).round(1),
    'Distance_km': np.random.uniform(10, 250, size=n_samples).round(1),
    'Transportation_Cost': np.random.uniform(500, 4500, size=n_samples).round(2),
    'Delivery_Time_hrs': np.random.uniform(1, 12, size=n_samples).round(1)
}

df = pd.DataFrame(data)

# Set plot style
sns.set_theme(style='whitegrid')

# 2. Visualization 1: Cost vs Distance by Route Type (Scatter Plot)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Distance_km', y='Transportation_Cost', hue='Route_Type', palette='Set1', s=70)
plt.title('Transportation Cost vs Distance by Route Type')
plt.xlabel('Distance (km)')
plt.ylabel('Transportation Cost (INR)')
plt.tight_layout()
plt.show()

# 3. Visualization 2: Delivery Time Distribution across Route Types (Box Plot)
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x='Route_Type', y='Delivery_Time_hrs', palette='pastel')
plt.title('Delivery Time Distribution across Routes')
plt.xlabel('Route Type')
plt.ylabel('Delivery Duration (Hours)')
plt.tight_layout()
plt.show()

# 4. Visualization 3: Correlation Heatmap for Performance Metrics
plt.figure(figsize=(6, 4))
numeric_df = df[['Shipment_Volume_m3', 'Distance_km', 'Transportation_Cost', 'Delivery_Time_hrs']]
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title('Logistics Performance Metrics Correlation')
plt.tight_layout()
plt.show()