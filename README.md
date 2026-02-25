# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Import required libraries (pandas, numpy, sklearn, matplotlib, seaborn).
3. Load the encoded car dataset from the CSV file.
4. Apply one-hot encoding to handle categorical variables.
5. Separate input features (X) and target variable (price).
6. Standardize features and target using StandardScaler.
7. Split the dataset into training and testing sets.
8. Create regression pipelines with PolynomialFeatures and 9.models (Ridge, Lasso, ElasticNet).
9. Train each model, generate predictions, and compute MSE, MAE, and R² Score.
10. Compare model performance and visualize results using bar plots.

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.pipeline import Pipeline 
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("encoded_car_data (1).csv")
print(data.head())

data=pd.get_dummies(data,drop_first=True)

x=data.drop('price',axis=1)
y=data['price']

scaler=StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(y.values.reshape(-1,1))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

models={
    "Ridge":Ridge(alpha=1.0),
    "Lasso":Lasso(alpha=1.0),
    "ElasticNet":ElasticNet(alpha=1.0,l1_ratio=0.5)
}

results={}

for name, model in models.items():
    pipeline=Pipeline([
        ('poly',PolynomialFeatures(degree=2)),
        ('regressor',model)
    ])

pipeline.fit(x_train,y_train)

predictions=pipeline.predict(x_test)

mse=mean_squared_error(y_test,predictions)
r2=r2_score(y_test,predictions)
mae=mean_absolute_error(y_test,predictions)

results[name]={'MSE':mse,'R^2 Score':r2,'MAE':mae}

print("Name: S R NIVEDHITHA")
print("Reg No: 212225240102")

for model_name, metrics in results.items():
    print(f"{model_name} - Mean Squared Error: {metrics ['MSE']:.2f}, R^2 Score: {metrics['R^2 Score']:.2f}")

results_df = pd.DataFrame (results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks (rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R^2 Score', data=results_df, palette='viridis')
plt.title('R^2 Score')
plt.ylabel('R^2 Score')
plt.xticks (rotation=45)


plt.tight_layout()
plt.show()
Developed by: S R NIVEDHITHA
RegisterNumber:  212225240102
*/

```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![alt text](<Screenshot 2026-02-24 200751.png>)
![alt text](<Screenshot 2026-02-24 200759.png>)
![alt text](<Screenshot 2026-02-24 200810.png>)
![alt text](<Screenshot 2026-02-24 200821.png>)
![alt text](<Screenshot 2026-02-24 200835.png>)

## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
