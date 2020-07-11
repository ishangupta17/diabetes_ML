#IMPORTING IMPORTANT FILES
import numpy as np
import pandas as pd
import pickle
df= pd.read_csv('diabetes.csv')
print(df.columns)
print(df.shape)


#DISTRIBUTING TARGET VALUES
X = np.array(df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']])
y= np.array(df['Outcome'])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


fr = RandomForestClassifier(n_estimators=50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
fr.fit(X_train,y_train)
pred = fr.predict(X_test)
print(classification_report(y_test,pred))



#final model
pickle.dump(fr,open('diabetes.pkl','wb'))
