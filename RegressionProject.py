# This is a machine learning (unsupervised) project on road accident in Ghana
# dataset is sourced from OpenDataGhana
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

road_accident = pd.read_csv('C:/Users/Kwadwo Addo-Brako/Desktop/Database.csv')
road_accident_binary = road_accident[['Crashes', 'Killed']]
road_accident_binary.columns = ['Crashes', 'Killed']
road_accident_binary.head()
sns.lmplot(x='Crashes', y='Killed', data=road_accident_binary, order=2, ci=None)
X = np.array(road_accident_binary["Crashes"]).reshape(-1, 1)
y = np.array(road_accident_binary['Killed']).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
reg_acc = LinearRegression()
reg_acc.fit(X_train, y_train)
print(reg_acc.score(X_test, y_test))
y_pred = reg_acc.predict(X_test)
plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color="k")
plt.show(block=True)

mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
print('MAE:', mae)
print('RMSE:', rmse)
