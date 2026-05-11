python id="x3ru0o"
models = {
    "Logistic Regression": 0.82,
    "KNN": 0.85,
    "Decision Tree": 0.80
}
for model, score in models.items():
    print(model, "Accuracy:", score)

python id="vzy0t2"
best_model = max(models, key=models.get)