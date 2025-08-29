import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
import os

experiments = "Asthma prediction models"
runs_df = mlflow.search_runs(experiment_names=[experiments])

print(f"Найдено {len(runs_df)} запусков")

# Сравнение результатов на cv
cols_to_use = [
    "tags.mlflow.runName",
    "metrics.cv_f1_macro_mean",
    "metrics.cv_f1_macro_std",
    "metrics.cv_training_time_sec"
]

runs_filtered = runs_df.dropna(subset=["metrics.cv_f1_macro_mean"])

runs_filtered = runs_filtered.rename(columns={
    "tags.mlflow.runName": "Model",
    "metrics.cv_f1_macro_mean": "Mean f1 macro (cv)",
    "metrics.cv_f1_macro_std": "Std f1 macro (cv)",
    "metrics.cv_training_time_sec": "Cv duration (sec)"
})

runs_filtered = runs_filtered.sort_values("Mean f1 macro (cv)", ascending=False)

os.makedirs("visualizations", exist_ok=True)

# Сравнение моделей по качеству
plt.figure(figsize=(10, 6))
sns.barplot(x="Mean f1 macro (cv)", y="Model", data=runs_filtered, palette="viridis")
plt.title("Сравнение моделей по средней f1 macro (cv)")
plt.xlabel("Средний f1 macro (cv)")
plt.ylabel("Модель")
plt.xlim(left=runs_filtered["Mean f1 macro (cv)"].min() - 0.01) 
plt.tight_layout()
plt.savefig("visualizations/model_quality_comparison.png")
print("График сравнения по качеству сохранен")

# Сравнение по качеству и скорости
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Cv duration (sec)",
    y="Mean f1 macro (cv)",
    hue="Model",
    s=200,
    data=runs_filtered,
    palette="magma"
)
plt.title("Сравнение моделей: cкорость vs качество")
plt.xlabel("Время cv (сек)")
plt.ylabel("Средний f1 macro (cv)")
plt.legend(title="Модели")
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/model_speed_vs_quality_comparison.png")
print("График сравнения по скорости и качеству сохранен")