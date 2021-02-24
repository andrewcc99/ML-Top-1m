import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('wc1.csv')
pd.set_option('display.max_columns', None)
df.sample(30)

dummies = pd.get_dummies(df, columns=['type'])
df.sample(4)

X_train, X_test, y_train, y_test = train_test_split(
    dummies, df['type'],
    test_size=0.33, random_state=42)

# Initialize and train classifier model
model = LogisticRegression(max_iter=10000).fit(X_train, y_train)


# Make predictions on test set
y_pred = model.predict(X_test)
y_score2 = model.predict_proba(X_test)[:,1]

print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_test, y_pred))

cols = ['avg_external_script_block','avg_cyc_complexity']
normal = df

normal[cols]=(normal[cols]-normal[cols].min())/(normal[cols].max()-normal[cols].min())

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('type', axis=1), normal['type'],
    test_size=0.33, random_state=78)

model2 = LogisticRegression(max_iter=10000).fit(X_train, y_train)

y_pred = model2.predict(X_test)
y_score2 = model2.predict_proba(X_test)[:,1]

print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_test, y_pred))



