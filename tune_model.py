import pandas as pd
import os
import mlflow
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

data_dir = "data"
X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

mlflow.set_experiment("Asthma hyperparameter tuning for gb")
model_to_tune = GradientBoostingClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
parameter_combinations = list(ParameterGrid(param_grid))

best_score = -1
best_params = None

# Запуск родительского эксперимента для hyperparameter tuning
with mlflow.start_run(run_name="gb_search") as parent_run:
    mlflow.log_params(param_grid) 
    
    # Перебор всех комбинаций гиперпараметров
    for i, params in enumerate(parameter_combinations):
        with mlflow.start_run(run_name=f"run_{i+1}", nested=True) as child_run:
            
            mlflow.log_params(params)
            model = GradientBoostingClassifier(**params, random_state=42)
            
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_macro', n_jobs=-1)   
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            mlflow.log_metric("cv_f1_macro_mean", mean_score)
            mlflow.log_metric("cv_f1_macro_std", std_score)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

    print(f"Лучший f1-macro: {best_score:.4f}")
    best_params_for_logging = {f"best_{key}": value for key, value in best_params.items()}
    mlflow.log_params(best_params_for_logging) 
    print(f"Лучшие параметры: {best_params}")
    mlflow.log_metric("best_f1_macro_score", best_score)

# Сохранение и логирование лучшей модели
with mlflow.start_run(run_name="Best_gb_model_tuned"):
    mlflow.log_params(best_params)

    input_example_df = X_train.head().copy()
    int_cols = input_example_df.select_dtypes(include=['int', 'int64']).columns
    input_example_df[int_cols] = input_example_df[int_cols].astype(float)

    best_model = GradientBoostingClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    mlflow.log_metric("final_train_f1_macro", best_score)
    mlflow.sklearn.log_model(best_model, "best_model", input_example=input_example_df)
    print("Лучшая модель сохранена в MLflow")