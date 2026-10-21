import joblib
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

if not os.path.exists('models'):
    os.mkdir('models')

print("Loading data...")
x_train, x_test, y_train, y_test = joblib.load(r'artifacts\data_splits.pkl')
print(f"Training set: {x_train.shape}, Testing set: {x_test.shape}")
print("Training Naive Bayes with PCA...")
# Added StandardScaler + PCA - Naive Bayes assumes feature independence,
# PCA decorrelates the features which helps satisfy this assumption
pipe_nb = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('nb', GaussianNB())])
pipe_nb.fit(x_train, y_train)
print("Training accuracy of the model is {:.2f}".format(pipe_nb.score(x_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(pipe_nb.score(x_test, y_test)))
joblib.dump(pipe_nb, r'models\naive_bayes.pkl')
print("Model saved to models\\naive_bayes.pkl")