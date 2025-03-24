
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

data = pd.read_csv("placedata v2.0 synthetic.csv")
print(list(data.columns))
data.head()

data.tail()

lb = LabelEncoder()
data['PlacementStatus'] = lb.fit_transform(data['PlacementStatus'])
data['PlacementTraining'] = lb.fit_transform(data['PlacementTraining'])
data['ExtracurricularActivities'] = lb.fit_transform(data['ExtracurricularActivities'])
data = data.drop(['StudentID'], axis = 1)
outputs = data['PlacementStatus']
inputs = data.drop(['PlacementStatus'], axis = 1)
std = StandardScaler()
inputs = pd.DataFrame(std.fit_transform(inputs))

data['PlacementTraining']

inputs

data.isna().sum()

data.describe()

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import math,random
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report

X_train,X_test,y_train,y_test=train_test_split(inputs,outputs,test_size=0.3,random_state=0)

from keras.layers import Dense, Dropout

model = keras.Sequential([
    Dense(64, input_shape=(10,), activation='relu'),
    Dropout(0.5),  # Adding dropout for regularization
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.2)


model.save("placement_model.keras")

# Save the scaler
import joblib
joblib.dump(std, "scaler.pkl")

