import joblib
import os
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

if not os.path.exists('models'):
    os.mkdir('models')

print("Loading data...")
x_train, x_test, y_train, y_test = joblib.load(r'artifacts\data_splits.pkl')
print(f"Training set: {x_train.shape}, Testing set: {x_test.shape}")
print("Training Voting Ensemble...")
# Recreating the top performing pipelines
pipe_linear = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)),
    ('svc', SVC(kernel='linear', C=10, probability=True))])

pipe_rbf = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)),
    ('svc', SVC(kernel='rbf', C=10, gamma='scale', probability=True))])

pipe_mlp = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu', solver='adam',
        alpha=0.001, max_iter=500,
        early_stopping=True, random_state=42
    ))
])

rfc = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)),
    ('rf', RandomForestClassifier(n_estimators=500, random_state=42))])

# Build the Voting Classifier over the top 4 models
ensemble = VotingClassifier( estimators=[('linear', pipe_linear), ('rbf', pipe_rbf), ('mlp', pipe_mlp), ('rf', rfc)],
    voting='soft', # Soft voting uses predicted probabilities for a more accurate consensus
    n_jobs=-1)

ensemble.fit(x_train, y_train)
print("Ensemble - Training accuracy of the model is {:.2f}".format(ensemble.score(x_train, y_train)))
print("Ensemble - Testing accuracy of the model is {:.2f}".format(ensemble.score(x_test, y_test)))
joblib.dump(ensemble, r'models\ensemble_voting.pkl')
print("Model saved to models\\ensemble_voting.pkl")