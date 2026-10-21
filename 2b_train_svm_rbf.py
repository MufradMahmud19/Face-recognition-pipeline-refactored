import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

if not os.path.exists('models'):
    os.mkdir('models')

print("Loading augmented data...")
x_train, x_test, y_train, y_test = joblib.load(r'artifacts\data_splits.pkl')
print(f"Training set: {x_train.shape}, Testing set: {x_test.shape}")
print("Tuning and Training RBF SVM...")
# The base pipeline
pipe_rbf = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('svc', SVC(kernel='rbf'))])

# the grid of parameters to test
param_grid = {'svc__C': [1, 10, 50, 100],        # Tests different regularization strengths
    'svc__gamma': ['scale', 0.01, 0.001, 0.0001]} # Tests different kernel coefficients


# GridSearchCV will test all 16 combinations using 5-fold cross-validation
grid_search = GridSearchCV(pipe_rbf, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(x_train, y_train)
print("\nBest Parameters Found: ", grid_search.best_params_)
print("Best CV Score: {:.2f}".format(grid_search.best_score_))
print("RBF SVM - Training accuracy: {:.2f}".format(grid_search.score(x_train, y_train)))
print("RBF SVM - Testing accuracy: {:.2f}".format(grid_search.score(x_test, y_test)))
# Saving the mathematically optimized model
joblib.dump(grid_search.best_estimator_, r'models\svm_rbf.pkl')
print("Optimized model saved to models\\svm_rbf.pkl")