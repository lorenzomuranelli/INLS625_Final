# Import of necessary packages: excel, plots, data manipulation, regressions
import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import statsmodels.api as sm
import seaborn as sns
from seaborn import pairplot
from statsmodels.graphics.correlation import plot_corr

data = pd.read_excel("betting_data.xlsx")
pd.options.mode.chained_assignment = None  # default='warn'
print(data.shape)

current = data[(data['Season'] > 2005)]
current.isna().sum()

current = current.dropna(thresh=0.8*len(current), axis=1)
current.isna().sum()
current.info()

current.describe()
# Filter for the Buffalo Bills. The best team in football.
current['home_or_away'] = np.where(current['team_home'] == 'Buffalo Bills', 1, 0)
current2 = current.loc[(current["team_home"] == "Buffalo Bills") | (current["team_away"] == "Buffalo Bills")]
final = current2.filter(["team_home","team_away", "score_home","score_away" ,"home_or_away", "over_under_line", "spread_favorite", "team_favorite_id", "stadium_neutral"])
final['score'] = np.where(final['team_away'] == 'Buffalo Bills', final['score_away'], final['score_home'])
final = final.dropna()

final = final.fillna(final.mean())

df = final[['over_under_line','home_or_away', 'score', "spread_favorite", "stadium_neutral"]] #'team_favorite_id'
# New df
df.info()

df['home_or_away'] = df['home_or_away'].astype('float64')
df['over_under_line'] = df['over_under_line'].astype('float64')
df['score'] = df['score'].astype('float64')
#df['team_favorite_id'] = df["team_favorite_id"].astype('object')
df['spread_favorite'] = df["spread_favorite"].astype('float64')
df["stadium_neutral"] = df["stadium_neutral"].astype('float64')
df.info()

# Looking for +.5 - +1 correlation
pairplot(df)
plt.show()
corr = df.corr()
print(corr)


X = pd.DataFrame(df, columns = ['home_or_away', 'stadium_neutral', "over_under_line", "spread_favorite"])
y = pd.DataFrame(df, columns=['score'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#Regression model
lin_reg_mod = LinearRegression()
# Fit linear regression
lin_reg_mod.fit(X_train, y_train)
# Make prediction on the testing data
pred = lin_reg_mod.predict(X_test)
print(lin_reg_mod.intercept_)
print(lin_reg_mod.coef_)

#Calculating r-squared
test_set_r2 = r2_score(y_test, pred)
print(test_set_r2)

#Calculating RMSE
test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
print(test_set_rmse)

df_results = y_test
df_results['Predicted'] = pred.ravel()
df_results['Residuals'] = abs(df_results['score']) - abs(df_results['Predicted'])
print(df_results)

# Plotting the actual vs predicted values
sns.lmplot(x='score', y='Predicted', data=df_results, fit_reg=False)
line_coords = np.arange(df_results.score.min().min(), df_results.Predicted.max().max())
plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
plt.title('Actual vs. Predicted')
plt.show()

#Correlation matrix
df2 = df[['spread_favorite', 'stadium_neutral', 'home_or_away', 'score']]
corr2 = df2.corr()
# Create Correlation Matrix
fig= plot_corr(corr2,xnames=corr2.columns)
plt.show()


#Resource: https://www.kaggle.com/tobycrabtree/nfl-scores-and-betting-data
#Resource: https://medium.com/swlh/predicting-nfl-scores-in-python-3560ccd58cb1
#Resource: https://pandas.pydata.org/pandas-docs/stable/index.html
#Resource: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
#Resource: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
#Resource: https://www.geeksforgeeks.org/indexing-and-selecting-data-with-pandas/
#Resource: https://github.com/PlayingNumbers/NBASimulator/blob/master/NBAFinalsSimulation.ipynb
#Resource: https://stackoverflow.com/questions/54116693/modulenotfounderror-no-module-named-seaborn-in-python-ide
#Resource: https://pypi.org/project/scikit-learn/
#Resource: https://github.com/TyWalters/NFL-Prediction-Model/blob/master/NFLBettingModelTraining.py
