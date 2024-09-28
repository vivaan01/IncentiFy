import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('customer_data.csv')

# Sample features: order history, engagement, previous incentives, etc.
X = df[['order_count', 'total_spend', 'engagement_score', 'last_incentive']]
y = df['next_incentive']  # Target is the optimal incentive to offer next

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Normalizing the data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
