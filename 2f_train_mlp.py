import joblib
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

if not os.path.exists('models'):
    os.mkdir('models')

print("Loading data...")
x_train, x_test, y_train, y_test = joblib.load(r'artifacts\data_splits.pkl')
print(f"Training set: {x_train.shape}, Testing set: {x_test.shape}")
print("Training Neural Network (MLP)...")
# Proper pipeline with scaling + PCA before MLP
pipe_mlp = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),  # deeper network
        activation='relu',
        solver='adam',
        alpha=0.001,             # L2 regularization to prevent overfitting
        batch_size=64,
        learning_rate='adaptive', # reduces LR when loss plateaus
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,      # stop when validation loss stops improving
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    ))
])

#nnc = MLPClassifier(hidden_layer_sizes=(7,), activation='relu', max_iter=10000)
pipe_mlp.fit(x_train, y_train)
print("Training accuracy of the model is {:.2f}".format(pipe_mlp.score(x_train, y_train)))
print("Testing accuracy of the model is {:.2f}".format(pipe_mlp.score(x_test, y_test)))
joblib.dump(pipe_mlp, r'models\mlp.pkl')
print("Model saved to models\\mlp.pkl")