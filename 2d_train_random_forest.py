import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

if not os.path.exists('models'):
    os.mkdir('models')

print("Loading data...")
x_train, x_test, y_train, y_test = joblib.load(r'artifacts\data_splits.pkl')
print(f"Training set: {x_train.shape}, Testing set: {x_test.shape}")
print("Training Random Forest with GridSearchCV...")
pipe_rf = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)),
    ('rf', RandomForestClassifier(random_state=42))])

param_grid = {'rf__n_estimators': [200, 500], 'rf__max_depth': [None, 30, 50], 'rf__min_samples_split': [2, 5],}
grid_search = GridSearchCV(pipe_rf, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(x_train, y_train)
print("\nBest Parameters Found:", grid_search.best_params_)
print("Best CV Score: {:.2f}".format(grid_search.best_score_))
print("Training accuracy of the model is {:.2f}".format(grid_search.score(x_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(grid_search.score(x_test, y_test)))
joblib.dump(grid_search.best_estimator_, r'models\random_forest.pkl')
print("Model saved to models\\random_forest.pkl")