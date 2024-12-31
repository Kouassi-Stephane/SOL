import streamlit as st
import pandas as pd
import numpy as np
import joblib

def load_model():
    # Charger le modèle enregistré
    return joblib.load('soil_model.pkl')

def predict(model, features):
    # Prédire la classe et les probabilités
    prediction = model.predict([features])
    proba = model.predict_proba([features])
    return prediction[0], proba[0]

def main():
    st.title('Prédiction de Compatibilité des Sols pour le Cacao')

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
        if prediction == 1:
            st.success('Sol compatible pour la culture du cacao')
        else:
            st.error('Sol non compatible pour la culture du cacao')

        st.write(f'Probabilité de compatibilité : {proba[1]:.2%}')

        # Affichage des seuils
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

        # Vérification de la compatibilité des paramètres avec les seuils
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
