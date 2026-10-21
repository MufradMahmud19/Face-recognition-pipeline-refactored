import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

if not os.path.exists('models'):
    os.mkdir('models')

print("Loading data...")
x_train, x_test, y_train, y_test = joblib.load(r'artifacts\data_splits.pkl')
print(f"Training set: {x_train.shape}, Testing set: {x_test.shape}")
print("Training SVC Sigmoid with GridSearchCV...")
pipe_sigmoid = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)),
('svc', SVC(kernel='sigmoid'))])

# Sigmoid kernel is very sensitive to C and gamma so need careful tuning
param_grid = {'svc__C': [0.01, 0.1, 1, 10], 'svc__gamma': ['scale', 0.001, 0.0001],}
grid_search = GridSearchCV(pipe_sigmoid, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(x_train, y_train)
print("\nBest Parameters Found:", grid_search.best_params_)
print("Best CV Score: {:.2f}".format(grid_search.best_score_))
print("SVC Sigmoid - Training accuracy of the model is {:.2f}".format(grid_search.score(x_train, y_train)))
print("SVC Sigmoid - Testing accuracy of the model is {:.2f}".format(grid_search.score(x_test, y_test)))
joblib.dump(grid_search.best_estimator_, r'models\svc_sigmoid.pkl')
print("Model saved to models\\svc_sigmoid.pkl")