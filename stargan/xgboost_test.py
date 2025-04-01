import numpy as np
import pandas as pd
import xgboost as xgb
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight


# Chargement
l1_matrix = np.load("l1_distances.npy")     # (N_samples, C_DIM)
true_labels = np.load("true_labels.npy")    # (N_samples,)
print("✅ l1_distances shape :", l1_matrix.shape)
print("✅ true_labels shape  :", true_labels.shape)
print("\n▶️ Exemple de vecteur L1 :", l1_matrix[0])
print("🧷 Label associé         :", true_labels[0])

# ==== SPLIT ====
X_train, X_val, y_train, y_val = train_test_split(
    l1_matrix, true_labels, test_size=0.2, stratify=true_labels, random_state=42
)
print(f"\n🧪 Split: {X_train.shape[0]} train - {X_val.shape[0]} val")

# ==== POIDS DE CLASSES ====
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_dict = {i: w for i, w in enumerate(class_weights)}

# ==== XGBOOST ====
clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=l1_matrix.shape[1],
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    verbosity=1,
    eval_metric='mlogloss'
)

sample_weights = np.array([weights_dict[label] for label in y_train])
clf.fit(X_train, y_train, sample_weight=sample_weights)

# ==== ÉVALUATION ====
y_val_pred = clf.predict(X_val)
print("\n📊 Classification Report (validation) :\n", classification_report(y_val, y_val_pred))

cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion - XGBoost (Validation)")
plt.xlabel("Prédictions")
plt.ylabel("Vérité")
plt.tight_layout()
plt.savefig("xgboost_val_confusion_68.png")
plt.show()

# ==== PRÉDICTION SUR DATA DE TEST FINAL ====
l1_test = np.load("l1_distances_test.npy")
test_dir = "/home/miashs2/données/datatest"
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])
test_ids = [os.path.splitext(f)[0] for f in test_files]

assert len(test_ids) == l1_test.shape[0], "🚨 Mismatch entre fichiers et features !"

y_test_pred = clf.predict(l1_test)

df_submit = pd.DataFrame({
    "idx": test_ids,
    "gt": y_test_pred
})

df_submit.to_csv("submission_xgboost_20.csv", index=False)
print("📤 Fichier de soumission généré : submission_xgboost.csv")