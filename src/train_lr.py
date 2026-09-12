from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluate import evaluate_classifier


def train_and_evaluate_lr(X_train, y_train, X_test, y_test):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_classifier(y_test, y_pred, y_prob)
    return model, metrics
