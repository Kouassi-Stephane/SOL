import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Charger les données
data = pd.read_csv('sol.csv', sep=';', encoding='latin-1')

# Traitement des colonnes numériques
numeric_columns = [
    'pH (eau)', 'pH (eau)_Min', 'Carbone organique', 'Carbone organique_Min', 'Carbone organique_Max',
    'ratio C/N', 'ratio C/N_Min', 'ratio C/N_Max', 'Azote total', 'Azote total_Min', 'Azote total_Max',
    'Bore (eau chaude)', 'Bore (eau chaude)_Min', 'Bore (eau chaude)_Max', "Capacité d'échange cationique"
]

for col in numeric_columns:
    if data[col].dtype == 'object':
        data[col] = data[col].str.replace(',', '.').astype(float)

# Aperçu des données
print("Information sur le dataset:")
print(data.info())
print("\nAperçu des données:")
print(data.head())
print("\nStatistiques descriptives:")
print(data.describe())
print("\nValeurs manquantes:")
print(data.isnull().sum())

# Gestion des valeurs manquantes : on peut imputer ou supprimer les lignes
# Par exemple, ici on supprime les lignes avec des valeurs manquantes
data.dropna(inplace=True)

# Suppression des doublons
data.drop_duplicates(inplace=True)

# Fonction de création de la cible
def create_target(row):
    # Plages recommandées pour chaque paramètre
    conditions = [
        6.0 <= row['pH (eau)'] <= 7.0,
        row['Carbone organique'] >= 2.0,
        10 <= row['ratio C/N'] <= 12,
        row['Azote total'] >= 0.2,
        row['Phosphore total'] >= 200,
        row['Bore (eau chaude)'] >= 0.5,
        row["Capacité d'échange cationique"] >= 150
    ]
    
    # Compter le nombre de conditions remplies
    conditions_remplies = sum(conditions)
    
    # Afficher l'analyse agronomique pour la première ligne
    if row.name == 0:
        print("\nAnalyse agronomique de la première ligne:")
        print(f"pH: {row['pH (eau)']} (optimal: 6.0-7.0)")
        print(f"Carbone: {row['Carbone organique']}% (min: 2.0%)")
        print(f"C/N: {row['ratio C/N']} (optimal: 10-12)")
        print(f"Azote: {row['Azote total']}% (min: 0.2%)")
        print(f"Phosphore: {row['Phosphore total']} mg/kg (min: 200)")
        print(f"Bore: {row['Bore (eau chaude)']} mg/kg (min: 0.5)")
        print(f"CEC: {row['Capacité d\'échange cationique']} mmol/kg (min: 150)")
        print(f"Critères remplis: {conditions_remplies}/7")
    
    # Si 3 conditions ou plus sont remplies, le sol est compatible
    return int(conditions_remplies >= 3)

# Appliquer la fonction pour créer la colonne 'target'
data['target'] = data.apply(create_target, axis=1)

# Distribution et proportion des classes
print("\nDistribution des classes:")
print(data['target'].value_counts())
print("\nProportion des classes:")
print(data['target'].value_counts(normalize=True))

# Sélection des caractéristiques et de la cible
features = ['pH (eau)', 'Carbone organique', 'ratio C/N', 'Azote total', 
            'Phosphore total', 'Bore (eau chaude)', "Capacité d'échange cationique"]
X = data[features]
y = data['target']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline avec scaler et modèle RandomForest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# Paramètres à tester pour la recherche par grille
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 20, None],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]
}

# Recherche par grille avec validation croisée
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Affichage des meilleurs paramètres
print("\nMeilleurs paramètres:", grid_search.best_params_)

# Prédiction sur l'ensemble de test
y_pred = grid_search.predict(X_test)

# Rapport de classification
print("\nRapport de classification:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
plt.title('Matrice de confusion')
plt.xlabel('Prédictions')
plt.ylabel('Vérités terrain')
plt.show()

# Importance des caractéristiques
importances = grid_search.best_estimator_.named_steps['rf'].feature_importances_
feature_importance = pd.DataFrame({'feature': features, 'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Visualisation de l'importance des caractéristiques
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Importance des features')
plt.show()

# Sauvegarder le modèle
joblib.dump(grid_search.best_estimator_, 'soil_model.pkl')

# Sauvegarder les résultats de la recherche par grille
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv('grid_search_results.csv', index=False)

# Sauvegarder le rapport de classification
with open('classification_report.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))
