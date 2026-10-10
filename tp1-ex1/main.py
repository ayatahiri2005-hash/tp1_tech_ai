import pandas as pd
import numpy as np
data = pd.read_csv('uber.csv') 
print(data.head()) 
print(data.info()) 
print(data.isnull().sum()) 

data = data.dropna() 
data = data[data['fare_amount'] > 0] 
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime']) 
data['hour'] = data['pickup_datetime'].dt.hour 
data['day_of_week'] = data['pickup_datetime'].dt.dayofweek

def calculer_distance(lat1, lon1, lat2, lon2):
    rayon_terre = 6371  # rayon de la Terre en kilomètres

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return rayon_terre * c

data['distance'] = calculer_distance(
    data['pickup_latitude'],
    data['pickup_longitude'],
    data['dropoff_latitude'],
    data['dropoff_longitude']
)
print(data[['distance', 'fare_amount']].head())
#calcule la distance géographique entre deux points sur la Terre à partir de leurs latitudes et longitudes


from sklearn.model_selection import train_test_split #Si Pandas sert à ranger et nettoyer les données et NumPy à faire des calculs mathématiques rapides, Scikit-Learn sert à faire réfléchir l'ordinateur pour qu'il puisse apprendre à partir de ces données et faire des prédictions
# X = caractéristiques utilisées pour prédire
X = data[['distance', 'hour', 'day_of_week']]
# y = tarif réel à prédire
y = data['fare_amount']
# Séparer les données : 80 % pour apprendre, 20 % pour tester
#avec train_test_split: la première chose que fait l'ordinateur, c'est de mélanger les lignes du tableau pour éviter que les données ne soient toutes triées de la même manière (par exemple, toutes les données du client A au début, et du client B à la fin).
X_train, X_test, y_train, y_test = train_test_split(  #X test et Y test :Caractéristiques des trajets de test et tarifs réels des trajets de test
    X,
    y,
    test_size=0.2, #réserver 20 % des données pour le test.
    random_state=42 #graine ou seed: avec random_state=42, on dit à l'ordinateur:"Mélange les données de façon aléatoire, mais utilise toujours le point de départ numéro 42"=>Le mélange a l'air aléatoire, mais il sera strictement identique à chaque fois qu'on exécute le code.
)
print("Nombre de lignes pour l'entraînement :", len(X_train))
print("Nombre de lignes pour le test :", len(X_test))


from sklearn.linear_model import LinearRegression
# Créer le modèle de regression linéaire
lr = LinearRegression()
# Apprendre à partir des données d'entraînement
lr.fit(X_train, y_train)
# Prédire les tarifs des trajets de test
y_pred = lr.predict(X_test)  #pas forcément identiques aux prix réels de y_test
comparaison = pd.DataFrame({
    'Tarif réel': y_test,
    'Tarif prédit': y_pred
})
print(comparaison.head(10))


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)
r2 = r2_score(y_test, y_pred)  #R2 — Coefficient de détermination indique dans quelle mesure le modèle explique la variation des tarifs observés: =1 : prédictions parfaites sur les données évaluées.=0 : le modèle ne fait pas mieux.<0 : le modèle fait moins bien que cette référence.R2 proche de 1 est généralement souhaitable
print("Régression linéaire")
print("MAE =", mae)
print("RMSE =", rmse)
print("R² =", r2)


from sklearn.ensemble import RandomForestRegressor
# Créer le modèle avec 100 arbres
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1  #ordinateur: utilise tous les cœurs disponibles du processeur en même temps pour faire ce calcul le plus vite possible
)
# Entraîner le modèle
rf.fit(X_train, y_train)
# Prédire les tarifs des trajets de test
y_pred_rf = rf.predict(X_test)
print("Premières prédictions Random Forest :")
print(y_pred_rf[:10])


mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(
    mean_squared_error(y_test, y_pred_rf)
)
r2_rf = r2_score(y_test, y_pred_rf)
print("Random Forest")
print("MAE =", mae_rf)
print("RMSE =", rmse_rf)
print("R² =", r2_rf)
resultats = pd.DataFrame({
    'Modèle': ['Régression linéaire', 'Random Forest'],
    'MAE': [mae, mae_rf],
    'RMSE': [rmse, rmse_rf],
    'R²': [r2, r2_rf]
})
print(resultats.round(3))


import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(
    y_test,
    y_pred_rf,
    alpha=0.5
)
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.title('Prédiction du tarif Uber avec Random Forest')
plt.show()

# Au lieu de demander l'avis d'une seule personne, on demande l'avis de 100 experts différents (n_estimators=100, qui crée 100 arbres de décision).
# Chaque expert examine avec un angle légèrement différent et un échantillon de données un peu différent (c'est le côté Random).
# À la fin, chaque expert donne sa prédiction, et on fait la moyenne de toutes leurs prédictions.
# capable de capturer des relations complexes, non-linéaires et tordues des données