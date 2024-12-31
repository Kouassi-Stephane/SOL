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
import streamlit as st
import time

# Phase 1: Collecte et exploration de données
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

# Fonction de création de la cible
def create_target(row):
    conditions = [
        6.0 <= row['pH (eau)'] <= 7.0,
        row['Carbone organique'] >= 2.0,
        10 <= row['ratio C/N'] <= 12,
        row['Azote total'] >= 0.2,
        row['Phosphore total'] >= 200,
        row['Bore (eau chaude)'] >= 0.5,
        row["Capacité d'échange cationique"] >= 150
    ]
    
    conditions_remplies = sum(conditions)
    return int(conditions_remplies >= 3)

# Fonction pour créer la colonne 'target'
data['target'] = data.apply(create_target, axis=1)

# Phase 2: Visualisation des données et sélection du modèle
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

# Phase 3: Formation et test de modèles
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

# Sauvegarder le modèle
joblib.dump(grid_search.best_estimator_, 'soil_model.pkl')

# Phase 4: Déploiement et surveillance du modèle
def load_model():
    return joblib.load('soil_model.pkl')

def predict(model, features):
    prediction = model.predict([features])
    proba = model.predict_proba([features])
    return prediction[0], proba[0]

def main():
    st.set_page_config(page_title="Prédiction de Compatibilité des Sols", page_icon="🌱", layout="wide")
    st.markdown('<h1 style="color:blue;">Prédiction de Compatibilité des Sols pour le Cacao 🌿</h1>', unsafe_allow_html=True)

    st.markdown("""
    Cette application utilise des données sur les pramètres physico-chimiques des sols pour prédire leur compatibilité avec la culture du cacao. 
    Entrez les informations de votre sol pour savoir s'il est adapté à la culture du cacao.
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        ph = round(st.number_input('pH (eau)', min_value=0.0, max_value=14.0, value=6.0), 2)
        carbone = round(st.number_input('Carbone organique (%)', min_value=0.0, max_value=10.0, value=1.5), 2)
        ratio_cn = round(st.number_input('Ratio C/N', min_value=0.0, max_value=20.0, value=12.0), 2)
        azote = round(st.number_input('Azote total (%)', min_value=0.0, max_value=1.0, value=0.2), 2)

    with col2:
        phosphore = round(st.number_input('Phosphore total (ppm)', min_value=0.0, max_value=1200.0, value=300.0), 2)
        bore = round(st.number_input('Bore (eau chaude) (ppm)', min_value=0.0, max_value=1.0, value=0.3), 2)
        cec = round(st.number_input('Capacité d\'échange cationique (méq/100g)', min_value=0.0, max_value=200.0, value=75.0), 2)

    features = [ph, carbone, ratio_cn, azote, phosphore, bore, cec]

    if st.button('Prédire la compatibilité'):
        model = load_model()  # Charger le modèle
        prediction, proba = predict(model, features)

        st.header('Résultats')

        # Animation des ballons flottants si la compatibilité est positive
        if prediction == 1:
            st.success('🌟 Sol compatible pour la culture du cacao 🌟')
            for _ in range(3):
                time.sleep(1)
                st.balloons()  # Animation des ballons flottants
        else:
            st.error('🚫 Sol non compatible pour la culture du cacao 🚫')

        st.write(f'Probabilité de compatibilité : {proba[1]:.2%}')

        st.header('Analyse des paramètres')
        seuils = {
            'pH (eau)': [5.1, 7.0],
            'Carbone organique': [1.5, 4.5],
            'Ratio C/N': [9.5, 15.5],
            'Azote total': [0.2, 0.4],
            'Phosphore total': [200, 600],
            'Bore (eau chaude)': [0.16, 0.9],
            'Capacité d\'échange cationique': [75, 200]
        }

        for param, valeur in zip(['pH (eau)', 'Carbone organique', 'Ratio C/N', 'Azote total', 
                                'Phosphore total', 'Bore (eau chaude)', 
                                'Capacité d\'échange cationique'], features):
            min_val, max_val = seuils[param]
            if min_val <= valeur <= max_val:
                st.success(f'{param}: {valeur} (Dans la plage recommandée: {min_val}-{max_val})')
            else:
                st.error(f'{param}: {valeur} (Hors plage recommandée: {min_val}-{max_val})')

if __name__ == '__main__':
    main()
