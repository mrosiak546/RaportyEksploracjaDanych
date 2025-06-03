import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import OneHotEncoder

#Wczytanie danych
dane = pd.read_csv('insurance.csv', sep=',', skipinitialspace=True)

# =============================================================================
# Zmienne numeryczne:
# age - wiek osoby 
# bmi - wskaźnik masy ciała
# children - liczba dzieci
# charges - koszty leczenia
# =============================================================================

# =============================================================================
# Zmienne kategoryczne:
# sex - płeć
# smoker - czy jest osobą palącą
# region - region zamieszkania
# =============================================================================

kolumny_kategoryczne = ['region']

#Zamiana sex oraz smoker na wartosci 0 i 1
dane['sex'] = dane['sex'].map({'female': 0, 'male': 1})
dane['smoker'] = dane['smoker'].map({'no': 0, 'yes': 1})

#Informacje o danych
print(dane.info())

#Dane wczytane poprawnie jak można zauważyć po zwróconym wyniku:
# =============================================================================
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1338 entries, 0 to 1337
# Data columns (total 7 columns):
#  #   Column    Non-Null Count  Dtype  
# ---  ------    --------------  -----  
#  0   age       1338 non-null   int64  
#  1   sex       1338 non-null   object 
#  2   bmi       1338 non-null   float64
#  3   children  1338 non-null   int64  
#  4   smoker    1338 non-null   object 
#  5   region    1338 non-null   object 
#  6   charges   1338 non-null   float64
# dtypes: float64(2), int64(2), object(3)
# memory usage: 73.3+ KB
# None
# Wszystkie dane wczytane poprawnie nie wzbudzają podejrzeń 
# =============================================================================

#Podstawowe statystyki
print(dane.describe())
# =============================================================================
#                age          bmi     children       charges
# count  1338.000000  1338.000000  1338.000000   1338.000000
# mean     39.207025    30.663397     1.094918  13270.422265
# std      14.049960     6.098187     1.205493  12110.011237
# min      18.000000    15.960000     0.000000   1121.873900
# 25%      27.000000    26.296250     0.000000   4740.287150
# 50%      39.000000    30.400000     1.000000   9382.033000
# 75%      51.000000    34.693750     2.000000  16639.912515
# max      64.000000    53.130000     5.000000  63770.428010
# =============================================================================

#Sprawdzenie braków danych
# =============================================================================
# print(dane.isnull().any())
# age         False
# sex         False
# bmi         False
# children    False
# smoker      False
# region      False
# charges     False
# dtype: bool
# =============================================================================
print(dane.isna().any())
# =============================================================================
# age         False
# sex         False
# bmi         False
# children    False
# smoker      False
# region      False
# charges     False
# dtype: bool
# =============================================================================

#Sprawdzenie unikalnych wartosci w komórkach
for kolumna in dane.columns:
    print(kolumna,':',dane[kolumna].unique())
    print('\n')
    
# =============================================================================
# Zmienne kategoryczne które potem będziemy musieli rozbić więc zobaczmy ich wyniki:
#sex : ['female' 'male']
#smoker : ['yes' 'no']
#region : ['southwest' 'southeast' 'northwest' 'northeast']
# =============================================================================

#Zakodowanie na postać liczbową zmiennej kategorycznej
dane_enc = OneHotEncoder(sparse_output = False).set_output(transform = 'pandas')

dane_encoded = pd.concat([dane.drop(columns = kolumny_kategoryczne), 
                     dane_enc.fit_transform(dane[kolumny_kategoryczne])
                    ], axis = 1)

#Predyktory:
predyktory = dane_encoded.drop(columns = ['charges'])


#Przejdźmy do budowania Modeli jednakże na początku ustawmy nasze ziarno:
ziarno = 313104
X = predyktory
Y = dane_encoded['charges']

#Podzielenie danych na zbiór uczący i testowy
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=ziarno)

len(X_train)/(len(X_train)+len(X_test)), len(X_test)/(len(X_train)+len(X_test))
# Proporcje są zbliżone do statndardowych 70%-30%.



# LAS REGRESYJNY - budowanie modelu i dostrajanie parametrów
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from math import sqrt

def ocen_model_regresji(y_true, y_pred, digits = 3): 
    print('RMSE - Pierwiastek błędu średniokwadratowego', round(sqrt(mean_squared_error(y_true, y_pred)), digits))
    print('MAE - Średni błąd bezwzględny', round(mean_absolute_error(y_true, y_pred), digits))
    print('MAPE - Średni bezwzględny błąd procentowy', round(100*mean_absolute_percentage_error(y_true, y_pred), digits),'%')

hyperparameters = {'max_depth' : range(1,10), 'min_samples_split' : [20, 30,  50]}
best_forest_reg = GridSearchCV(RandomForestRegressor(random_state = ziarno, max_features = 1.0), hyperparameters, n_jobs = -1)
best_forest_reg.fit(X_train, Y_train)
best_forest_reg.best_params_

Y_train_pred = best_forest_reg.predict(X_train)
Y_test_pred = best_forest_reg.predict(X_test)
print('Zbiór uczący')
ocen_model_regresji(Y_train, Y_train_pred)
print('\n')
print('Zbiór testowy')
ocen_model_regresji(Y_test, Y_test_pred)

def waznosc_predyktorow(drzewo):
    waznosci = pd.Series(drzewo.feature_importances_, index=X_train.columns)
    waznosci.sort_values(inplace=True)
    waznosci.iloc[-10:].plot(kind='barh', figsize=(6,4))
    print(waznosci)

waznosc_predyktorow(best_forest_reg.best_estimator_)

def wykres_przewidywanych(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2) # Linia idealnego dopasowania
    plt.xlabel('Obserwowane wartości')
    plt.ylabel('Przewidywane wartości')
    plt.title(title)
    plt.show()
wykres_przewidywanych(Y_test, Y_test_pred, "LAS REGRESYJNY")


# DRZEWA WYJĄTKOWO LOSOWE - budowanie modelu i dostrajanie parametrów
from sklearn.ensemble import ExtraTreesRegressor
hyperparameters = {'max_depth' : range(1,10), 'min_samples_split' : [10, 20, 50]}
best_xtree_reg = GridSearchCV(ExtraTreesRegressor(random_state = ziarno), hyperparameters, n_jobs = -1)
best_xtree_reg.fit(X_train, Y_train)
best_xtree_reg.best_params_

Y_train_pred = best_xtree_reg.predict(X_train)
Y_test_pred = best_xtree_reg.predict(X_test)
print('Zbiór uczący')
ocen_model_regresji(Y_train, Y_train_pred)
print('\n')
print('Zbiór testowy')
ocen_model_regresji(Y_test, Y_test_pred)

waznosc_predyktorow(best_xtree_reg.best_estimator_)
wykres_przewidywanych(Y_test, Y_test_pred, "DRZEWA WYJĄTKOWO LOSOWE")


