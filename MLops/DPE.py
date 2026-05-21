
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("DPE_stageStudent")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


file_path = Path(r"C:\Users\yayaa\MLops\data\DPE.csv")
if not file_path.exists():
    raise FileNotFoundError(f"Fichier non trouvé : {file_path}")

df = pd.read_csv(file_path)
print("Dimensions du dataset :", df.shape)


TARGET = "conso_finale"

y = pd.to_numeric(df[TARGET], errors="coerce")

leak_cols = [TARGET, "code_postal", "type_energie_principale"]
leak_cols = [c for c in leak_cols if c in df.columns]

X = df.drop(columns=leak_cols, errors="ignore")

mask = y.notna()
X = X.loc[mask].copy()
y = y.loc[mask].copy()


X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42
)


num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()


num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),  # 🔧 Moins sensible aux outliers
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])


y_train_log = np.log1p(y_train)
y_val_log   = np.log1p(y_val)
y_test_log  = np.log1p(y_test)


print("\n" + "="*70)
print(" TEST DE DIFFÉRENTS MODÈLES")
print("="*70)

models = {
    "Ridge": Ridge(alpha=10.0),
    "Lasso": Lasso(alpha=1.0, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, 
        max_depth=15, 
        min_samples_split=10, 
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, 
        max_depth=5,
        learning_rate=0.1, 
        random_state=42
    )
}

results = {}
best_score = -np.inf
best_model_name = None
best_pipeline = None

for name, model in models.items():
    
    print(f"\n{'─'*70}")
    print(f" Modèle : {name}")
    print(f"{'─'*70}")

    log_model = Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

    # validation croisée
    cv_scores = cross_val_score(
        log_model,
        X_train,
        y_train_log,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print(" Entraînement...")
    log_model.fit(X_train, y_train_log)

    # prédictions
    y_pred_log_test = log_model.predict(X_test)
    y_pred_test = np.expm1(y_pred_log_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    results[name] = {
    "CV_R2_mean": cv_mean,
    "CV_R2_std": cv_std,
    "TEST_MAE": mae_test,
    "TEST_R2": r2_test,
    "TEST_MAPE": np.mean(np.abs(y_test - y_pred_test) / (y_test + 1e-9)) * 100
        }

    #  MLflow tracking
    with mlflow.start_run(run_name=name):

        mlflow.log_param("model", name)

        mlflow.log_metric("CV_R2_mean", cv_mean)
        mlflow.log_metric("CV_R2_std", cv_std)
        mlflow.log_metric("TEST_R2", r2_test)
        mlflow.log_metric("TEST_MAE", mae_test)

        mlflow.sklearn.log_model(
        sk_model=log_model,
        artifact_path="model",
        registered_model_name="dpe_model"
    )

    # garder le meilleur modèle
    if r2_test > best_score:
        best_score = r2_test
        best_model_name = name
        best_pipeline = log_model

# Résumé final

print("\n" + "="*70)
print(" TABLEAU RÉCAPITULATIF DES MODÈLES")
print("="*70)
print(f"{'Modèle':<20} {'CV R²':<15} {'TEST MAE':<12} {'TEST R²':<10} {'TEST MAPE':<10}")
print("─"*70)

for name, metrics in results.items():
    cv_r2_str = f"{metrics['CV_R2_mean']:.3f} ± {metrics['CV_R2_std']:.3f}"
    print(f"{name:<20} {cv_r2_str:<15} {metrics['TEST_MAE']:>10.2f}  {metrics['TEST_R2']:>8.3f}  {metrics['TEST_MAPE']:>8.2f}%")

print("="*70)
print(f" MEILLEUR MODÈLE : {best_model_name}")
print(f"   → R² TEST = {best_score:.3f}")
if best_model_name in results:
    print(f"    MAE TEST = {results[best_model_name]['TEST_MAE']:.2f}")
    print(f"    MAPE TEST = {results[best_model_name]['TEST_MAPE']:.2f}%")
print("="*70)


y_pred_train_final = np.expm1(best_pipeline.predict(X_train))
y_pred_val_final   = np.expm1(best_pipeline.predict(X_val))
y_pred_test_final  = np.expm1(best_pipeline.predict(X_test))

print("\n Modèle entraîné et prêt pour les prédictions !")
print(f" Pipeline du meilleur modèle stocké dans : best_pipeline")



idx = y_test.sample(1).index[0]
x_row = X.loc[[idx]]
y_true = y.loc[idx]


y_pred_log = best_pipeline.predict(x_row)[0]  
y_pred = np.expm1(y_pred_log)

print("\n" + "="*60)
print(" NOUVELLE PRÉDICTION ALÉATOIRE")
print("="*60)
print(f" Modèle utilisé    : {best_model_name}")
print(f" Index du test     : {idx}")
print(f" Valeur réelle     : {y_true:.2f} kWh/an")
print(f" Valeur prédite    : {y_pred:.2f} kWh/an")
print(f" Différence        : {y_pred - y_true:.2f} kWh/an")

print(f" Erreur absolue    : {abs(y_true - y_pred):.2f} kWh/an")
print(f" Erreur %          : {abs(y_true - y_pred) / max(y_true, 1e-9) * 100:.2f}%")
print("="*60)


# Visualisation : Prédictions Réelles vs Prédites (Meilleur Modèle)

import matplotlib.pyplot as plt

y_pred_test_final = np.expm1(best_pipeline.predict(X_test))

plt.figure(figsize=(10, 8))

plt.scatter(y_test, y_pred_test_final, alpha=0.6, s=50, color='#3498db', 
            edgecolors='black', linewidth=0.5, label='Prédictions')

min_val = min(y_test.min(), y_pred_test_final.min())
max_val = max(y_test.max(), y_pred_test_final.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
         linewidth=2, label='y = x (Prédiction parfaite)')

plt.xlabel('Valeurs Réelles (kWh/an)', fontsize=14, fontweight='bold')
plt.ylabel('Valeurs Prédites (kWh/an)', fontsize=14, fontweight='bold')
plt.title(f'Prédictions vs Réalité - Modèle: {best_model_name}', fontsize=16, fontweight='bold')

plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)

r2_test = r2_score(y_test, y_pred_test_final)
mae_test = mean_absolute_error(y_test, y_pred_test_final)
mape_test = np.mean(np.abs(y_test - y_pred_test_final) / (y_test + 1e-9)) * 100

plt.text(0.05, 0.95, 
         f' Modèle: {best_model_name}\n'
         f'R² = {r2_test:.3f}\n'
         f'MAE = {mae_test:.0f} kWh\n'
         f'MAPE = {mape_test:.2f}%',
         transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()


print("\n" + "="*70)
print(f" EXEMPLES DE PRÉDICTIONS - Modèle: {best_model_name}")
print("="*70)
print(f"{'Index':<10} {'Réel (kWh)':<15} {'Prédit (kWh)':<15} {'Erreur abs':<15} {'Erreur %':<12}")
print("-"*70)

echantillons = y_test.sample(10, random_state=42)
for idx in echantillons.index:
    reel = y_test.loc[idx]
    predit = y_pred_test_final[y_test.index.get_loc(idx)]
    erreur_abs = abs(reel - predit)
    erreur_pct = (erreur_abs / max(reel, 1e-9)) * 100
    
    print(f"{idx:<10} {reel:<15.0f} {predit:<15.0f} {erreur_abs:<15.0f} {erreur_pct:<11.2f}%")

print("="*70)


from pathlib import Path
import joblib

model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / f"dpe_model_{best_model_name}.pkl"

joblib.dump(best_pipeline, model_path)

print("\n" + "="*60)
print("MEILLEUR MODÈLE SAUVEGARDÉ")
print("="*60)
print(f"Modèle : {best_model_name}")
print(f"Chemin : {model_path.resolve()}")
print("="*60)