'''
This program is used to determine if the brain health of a patient is normal or not based on the health information and 
the exercise scores.

The dataset is simulated in this case study.
'''

# load the data
import pandas as pd
df = pd.read_csv('brain_health_data.csv')
# the first 15 columns are features
x = df.iloc[:,:-1].astype(float).values
# the last column is class label
y = df.iloc[:,-1].values

# feature selection based on feature importance using Random Forest
from sklearn.ensemble import RandomForestClassifier
rd = RandomForestClassifier()
from sklearn.feature_selection import SelectFromModel
# select the features with importance higher than the threshold
rd = SelectFromModel(rd, threshold = 0.10)
rd = rd.fit(x, y)
# selected features with a lower dimension
x_selected = rd.transform(x)

# split the data into training and test sets
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.20, random_state=1)

# pipeline: Scaler-->Logistic Regression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipeline_classifier = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(random_state=1)) ])

# tune the inverse regularization parameter C in logistic regression model
param_range = [0.01, 0.1, 1.0, 10.0]
# parameter grid [{para1:[value11, value12], para2:[value21, value22]}, {para1:[value], para2:[value2], para3:[value3]}]
param_grid=[{'clf__C': param_range} ]

# grid search with 4-fold cross validation
from sklearn.grid_search import GridSearchCV
gs = GridSearchCV(estimator=pipeline_classifier, param_grid=param_grid, scoring='accuracy', cv=6)
gs.fit(x_train, y_train)
print('The best accuracy: ', gs.best_score_)
print('The best parameters: ', gs.best_params_)

# the best configuration
best_model = gs.best_estimator_
# train the best model
best_model.fit(x_train, y_train)

# evaluation on test dataset
print ('Test accuracy: ', best_model.score(x_test, y_test))

