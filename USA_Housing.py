# Importing Pandas, a data processing and CSV file I/O libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn is a Python data visualization library based on matplotlib.


df_USAhousing = pd.read_csv('USA_Housing_toy.csv')

df_USAhousing.head()

df_USAhousing.isnull().sum()

df_USAhousing.describe()

df_USAhousing.info()

sns.pairplot(df_USAhousing)

sns.displot(df_USAhousing['Price'])

sns.heatmap(df_USAhousing.corr())

df_USAhousing.corr()

X = df_USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = df_USAhousing['Price']

# Import train_test_split function from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Split up the data into a training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)

sns.displot((y_test-predictions),bins=50, rug_kws=dict(edgecolor="black", linewidth=1),color='Blue');



from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

