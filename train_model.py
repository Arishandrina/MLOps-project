import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import pickle
import time 
import shap
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

data_dir = "data"

X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

mlflow.set_experiment("Asthma prediction models")

models = {
    "logistic_regression": LogisticRegression(
        penalty='l1', 
        C=1.0, 
        solver='liblinear', 
        max_iter=1000, 
        random_state=42
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.01,
        max_depth=3,
        random_state=42
    ),
    "svm": SVC(
        kernel='linear', 
        C=1.0,
        probability=True,
        random_state=42
    ),
    "k_nearest_neighbors": KNeighborsClassifier(
        n_neighbors=21,
        weights='distance'
    )
}

input_example_df = X_train.head().copy()
int_cols = input_example_df.select_dtypes(include=['int64']).columns
input_example_df[int_cols] = input_example_df[int_cols].astype(float)

# Обучение и логирование моделей
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"Обучение модели: {name}")

        mlflow.log_params(model.get_params())
        mlflow.set_tag("validation_method", "cross-validation")

        start_time_cv = time.time()
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
        end_time_cv = time.time()
        mlflow.log_metric("cv_f1_macro_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_macro_std", cv_scores.std())
        mlflow.log_metric("cv_training_time_sec", end_time_cv - start_time_cv)

        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        
        # Feature importance (permutation importance)
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        sorted_idx_10 = perm_importance.importances_mean.argsort()[-10:]
        perm_df_top10 = pd.DataFrame(
            perm_importance.importances[sorted_idx_10].T,
            columns=X_test.columns[sorted_idx_10]
        )

        plt.figure(figsize=(10, 8))
        perm_df_top10.boxplot(vert=False)
        plt.title(f'Permutation importance для {name}')
        plt.xlabel("Уменьшение accuracy score")
        plt.tight_layout()

        plt.savefig("permutation_importance.png")
        mlflow.log_artifact("permutation_importance.png", "feature_importance_plots")
        plt.close()

        #Feature importance (SHAP values)
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X_train)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))

        if name == "k_nearest_neighbors":
            X_test_shap = shap.sample(X_test, 100)
        else:
            X_test_shap = X_test

        shap_values = explainer.shap_values(X_test_shap)
        values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

        try: # Попытка построить beeswarm plot
            shap.summary_plot(values_for_plot, X_test_shap, show=False, max_display=15)
            plot_type_used = "beeswarm"

        except TypeError as e: # Если не получилось, строим bar plot
            plt.close()
            shap.summary_plot(values_for_plot, X_test_shap, plot_type="bar", show=False, max_display=15)
            plot_type_used = "bar"

        fig = plt.gcf()
        fig.set_figwidth(10)
        fig.set_figheight(8)
        plt.title(f'SHAP summary plot ({plot_type_used}) для {name}')
        fig.tight_layout()
        
        shap_plot_path = f"shap_summary_plot_{name}.png"
        fig.savefig(shap_plot_path)
        mlflow.log_artifact(shap_plot_path, "feature_importance_plots")
        plt.close(fig)
        os.remove(shap_plot_path)
        print(f"SHAP plot saved for {name}")
  
        # Оценка на тестовой выборке
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"Final test accuracy: {accuracy:.4f}\n")

        mlflow.log_metric("final_test_accuracy", accuracy)
        mlflow.log_metric("final_test_precision_class_1", report['1']['precision'])
        mlflow.log_metric("final_test_recall_class_1", report['1']['recall'])
        mlflow.log_metric("final_test_f1_class_1", report['1']['f1-score'])
        mlflow.log_metric("final_test_training_time_sec", training_time)
        
        mlflow.sklearn.log_model(model,
                                artifact_path=name,
                                input_example=input_example_df
                            )
        
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("train_size", len(X_train))
        mlflow.set_tag("test_size", len(X_test))
        
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"model_{name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
