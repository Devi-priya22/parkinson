import pandas as pd
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

dframe = pd.read_csv('/content/parkinsons (2).data')

# Get the features and labels
features = dframe.loc[:, dframe.columns != 'status'].values[:, 1:]
labels = dframe.loc[:, 'status'].values

scaler = StandardScaler()
features = scaler.fit_transform(features)

pca = PCA(n_components=12)
principalComponents = pca.fit_transform(features)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12'])
finalPCADf = pd.concat([principalDf, dframe[['status']]], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train.ravel())
X_test = lda.transform(X_test)

LDADf = pd.DataFrame(data=X_train, columns=['LD1'])
finalLDADf = pd.concat([LDADf, dframe[['status']]], axis=1)

array = features
X = array[:, 0:12]
Y = dframe.loc[:, ['status']].values

model = XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=7, subsample=0.8, colsample_bytree=0.7, gamma=0.01)
rfe = RFE(estimator=model, n_features_to_select=5)
fit = rfe.fit(X, Y)

pca_feat_list = []
for i in range(len(fit.support_)):
    if fit.support_[i]:
        pca_feat_list.append('PC' + str(i + 1))
print(pca_feat_list)

selected_features = fit.support_
feature_names = dframe.columns[1:13]  # Assuming the last column is the target column
selected_feature_names = feature_names[selected_features]

print(selected_feature_names)


import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
dframe = pd.read_csv('/content/parkinsons (2).data')

# Get the features and labels
features = dframe.loc[:, dframe.columns != 'status'].values[:, 1:]
labels = dframe.loc[:, 'status'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=0)

# Train XGBoost model
model = XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=7, subsample=0.8, colsample_bytree=0.7, gamma=0.01)

# Define the selected feature indices
selected_feature_indices = [0, 2, 4, 7, 11]  # Modify this with the correct indices

# Fit the model on the training data using the selected features
model.fit(X_train[:, selected_feature_indices], y_train)

# Evaluate the model on the testing data
accuracy = model.score(X_test[:, selected_feature_indices], y_test)
print("Accuracy:", accuracy)
import numpy as np
input_data=(119.99200,74.99700,0.00007,0.00784,0.03130)


## changing input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

## reshape the numpy array
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)



prediction= model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("Person doesnot have parkinson disease")
else:
    print("Person has parkinson disease")