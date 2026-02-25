# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Loading and Preprocessing
2. Data Splitting
3. Model Initialization
4. Pipeline Creation and Model Training
5. Performance Evaluation and Visualization

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: Anisha a
RegisterNumber: 212225220009 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("encoded_car_data (1).csv")
data.head()
data = pd.get_dummies(data, drop_first=True)
x = data.drop('price', axis=1)
y = data['price']
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y.values.reshape(-1,1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}
results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', model)
    ])
  pipeline.fit(x_train, y_train)
 predictions = pipeline.predict(x_test)
 mse = mean_squared_error(y_test, predictions)
  r2 = r2_score(y_test, predictions)
 results[name] = {'MSE': mse, 'R² Score': r2}
print('Name:anisha A ')
print('Reg. No: 212225220009')
for model_name, metrics in results.items():
    print(f"{model_name} - Mean Squared Error: {metrics['MSE']:.2f}, "f"R² Score: {metrics['R² Score']:.2f}")
results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R² Score', data=results_df, palette='viridis')
plt.title('R² Score')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Output:
<img width="1364" height="288" alt="Screenshot 2026-02-25 094011" src="https://github.com/user-attachments/assets/6a9680b2-671a-4ebb-8898-ac2f964b3721" />
<img width="535" height="156" alt="Screenshot 2026-02-25 094034" src="https://github.com/user-attachments/assets/d982b8c1-2fa0-4bfc-aa8a-d98a62256256" />
<img width="656" height="86" alt="Screenshot 2026-02-25 094127" src="https://github.com/user-attachments/assets/10646802-08e3-468f-b956-6a2595aa543e" />
<img width="656" height="86" alt="Screenshot 2026-02-25 094127" src="https://github.com/user-attachments/assets/96ddf78b-c20c-4120-8fdd-085735f404c3" />
<img width="707" height="704" alt="Screenshot 2026-02-25 094210" src="https://github.com/user-attachments/assets/c3ed1d58-0952-4f54-a8d2-883f0270fe32" />
<img width="759" height="677" alt="Screenshot 2026-02-25 094228" src="https://github.com/user-attachments/assets/34a5cad6-3ca9-43a9-ac2b-b97b71badff2" />
<img width="380" height="25" alt="Screenshot 2026-02-25 094245" src="https://github.com/user-attachments/assets/e3d91f35-c390-4028-80b6-5f3ffc6d0d23" />



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
