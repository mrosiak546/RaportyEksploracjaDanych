import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt




# Wczytanie danych
dane = pd.read_csv('gym_members_exercise_tracking.csv', sep=';', skipinitialspace=True)

# Informacje o danych
print(dane.info())

# Podstawowe statystyki
print(dane.describe())
# Sprawdzenie braków danych
print(dane.isnull().any())
print(dane.isna().any())
# Brak rekordow z brakujacymi danymi

#Sprawdzmy poprawnosc danych

#Sprawdzenie unikalnych wartosci wypelniajacych komorki
for kolumna in dane.columns:
    print(kolumna, ' : ', dane[kolumna].unique())
    print('\n')
    
#Kolumna Age nie posiada nieprawidłowych danych

dane['Gender'] = dane['Gender'].map({'Female': 0, 'Male': 1})
#Kolumna Gender nie posiada nieprawidłowych danych

# Zamiana przecinków na kropki i konwersja na float w kolumnie 'Weightkg'
dane['Weightkg'] = dane['Weightkg'].str.replace(',', '.').astype(float)
#Kolumna Weightkg nie posiada nieprawidłowych danych

dane['Heightm'] = dane['Heightm'].str.replace(',', '.').astype(float)
#Kolumna Heightm nie posiada nieprawidłowych danych

#Kolumny Max_BPM, Avg_BPM, Resting_BPM nie zawieraja błędnych danych

dane['Session_Durationhours'] = dane['Session_Durationhours'].str.replace(',', '.').astype(float)
#Kolumna Session_Durationhours nie posiada nieprawidłowych danych

#Kolumna Calories_Burned nie posiada nieprawidłowych danych

dane['Fat_Percentage'] = dane['Fat_Percentage'].str.replace(',', '.').astype(float)
#Kolumna Fat_Percentage nie posiada nieprawidłowych danych

#Kolumna Workout_Type nie posiada nieprawidłowych danych

dane['Water_Intakeliters'] = dane['Water_Intakeliters'].str.replace(',', '.').astype(float)
#Kolumna Water_Intakeliters nie posiada nieprawidłowych danych

#Kolumna Workout_Frequencydaysweek nie posiada nieprawidłowych danych

dane['BMI'] = dane['BMI'].str.replace(',', '.').astype(float)
#Kolumna BMI nie posiada nieprawidłowych danych

#Sprawdzenie unikalnych wartosci wypelniajacych komorki po zamianie typow danych
for kolumna in dane.columns:
    print(kolumna, ' : ', dane[kolumna].unique())
    print('\n')

#Sprawdzamy czy dane nie są zduplikowane
len(dane[dane.duplicated()])
#brak duplikatów






#ANALIZA PREDYKTOROW
#Sprawdzmy wplyw wartosci Workout_Type na zmienna celu Experience_level
tab = pd.crosstab(dane['Workout_Type'], dane['Experience_Level'])
tab_normalized = tab.div(tab.sum(axis=1), axis=0) * 100
print(tab_normalized)
#Jak widac liczba osob zaawansowanych i niezaawansowanych rozklada sie podobnie niezaleznie od typu wykonywanego treningu, a więc nie ma on bezposredniego
#wplywu na zmienna celu i mozemy go wykluczyc z predyktorow


#W celu analizy pozostalych predyktorow wykonamy macierz korelacji oraz heatmape
dane2 = dane.drop(columns=['Workout_Type'])
correlation_matrix = dane2.corr()

# Wizualizacja macierzy korelacji
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title('Macierz korelacji')
plt.show()

abs(correlation_matrix['Experience_Level']).sort_values(ascending = False)

#Z wykladu wiemy, ze sila korelacji ma sie nastepujaco:
#|r| = O — brak korelacji,
#0,0 < |r| 0,1 — korelacja nikła,
#0,1 < |r| 0,3 — korelacja słaba,
#0,3 < |r| 0,5 — korelacja umiarkowana,
#0,5 < |r| 0,7 — korelacja wysoka,
#0,7 < |r| 0,9 — korelacja bardzo wysoka,
#0,9 < |r| < 1 — korelacja prawie pełna,
#|r| = 1 — korelacja pełna.

#Jak widac z macierzy korelacji tylko 3 kolumny mają korelacje więcej niż nikłą:
#Workout_Frequencydaysweek - 0.73
#Session_Durationhours - 0.47
#Calories_Burned - 0.42

#Wybieram te 3 kolumny jako predyktory do budowanych modeli
dane_predyktory = dane[['Workout_Frequencydaysweek', 'Session_Durationhours', 'Calories_Burned', 'Experience_Level']]


#Przejdzmy do budowania modelu KNN
ziarno = 313104

X = dane_predyktory.iloc[:,:-1]  # wszystkie wiersze, wszystkie kolumny poza ostatnią
y = dane_predyktory['Experience_Level']
#Podzielenie danych na zbior uczacy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=ziarno, stratify = y)
len(X_train), len(X_test)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

X_train.columns

kolumny_num = ['Workout_Frequencydaysweek', 'Session_Durationhours', 'Calories_Burned']

# Zajmijmy się teraz normalizacja zmiennych numerycznych w zbiorze uczacym.
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(X_train[kolumny_num]) 
scaler.set_output(transform = 'pandas')
X_train = scaler.transform(X_train[kolumny_num])
X_train

# Zajmijmy się teraz normalizacja zmiennych numerycznych w zbiorze testowym.
X_test = scaler.transform(X_test[kolumny_num])
X_test







# Model KNN
# Sprawdzmy modele o ponizszych hiperparametrach
hyperparameters = {'n_neighbors' : [3, 5, 7, 9, 11], 'weights': ['uniform','distance']}
knn_best = GridSearchCV(KNeighborsClassifier(), hyperparameters, n_jobs = -1, error_score = 'raise')
knn_best.fit(X_train,y_train)
# Sprawdzamy hiperparametry optymalnego klasyfikatora.
knn_best.best_params_
# Najlepsze hiperparametry to n=11 i waga 'distance'

# Wyznaczmy przewidywane wartości zmiennej celu, tym razem za pomocą optymalnego modelu
y_pred_train = knn_best.predict(X_train)
y_pred_test = knn_best.predict(X_test)

def ocen_model_klasyfikacji_binarnej(y_true, y_pred, digits = 2):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tn+tp)/(tn+fp+fn+tp)
    overall_error_rate = 1 - accuracy
    sensitivity = tp/(fn+tp)
    fnr = fn/(fn+tp)
    specificity = tn/(tn+fp)
    fpr = fp/(tn+fp)
    precision = tp/(fp+tp)
    f1 = (2 * sensitivity * precision) / (sensitivity + precision)
    print('Trafność: ', round(accuracy, digits))
    print('Całkowity współczynnik błędu', round(overall_error_rate, digits))
    print('Czułość: ', round(sensitivity, digits))
    print('Wskaźnik fałszywie negatywnych: ', round(fnr, digits))
    print('Specyficzność: ', round(specificity, digits))
    print('Wskaźnik fałszywie pozytywnych: ', round(fpr, digits))
    print('Precyzja: ', round(precision, digits))
    print('Wynik F1: ', round(f1, digits))
print('Zbior uczacy')
ocen_model_klasyfikacji_binarnej(y_train, y_pred_train)
print('\n Zbior testowy')
ocen_model_klasyfikacji_binarnej(y_test, y_pred_test)

# Model ma praktycznie doskonałą wydajność na zbiorze testowym. Różnica w wynikach między zbiorem uczącym a testowym sugeruje, 
# że model jest przeuczony i jego zdolność do generalizacji jest ograniczona.

# Sprawdzmy pozostale testowane modele czy ktorys z nich nie jest lepszym wyborem knn_best.cv_results_
results_df = pd.DataFrame(knn_best.cv_results_)

# Zaokrąglenie wyników do 4 miejsc po przecinku
results_df['mean_test_score'] = results_df['mean_test_score'].round(4)
results_df['std_test_score'] = results_df['std_test_score'].round(4)  # Jeśli chcesz też zaokrąglić odchylenie standardowe

# Wyświetlenie tabeli z zaokrąglonymi wynikami
print(results_df[['params', 'mean_test_score', 'rank_test_score']])

# Na drugim miejscu znajduje sie model o parametrach n=3 i wadze = uniform
# Przeprowadzmy analize tego modelu
knn3 = KNeighborsClassifier(n_neighbors=3, weights='uniform')
knn3.fit(X_train, y_train)
y_pred_train_3 = knn3.predict(X_train)
y_pred_test_3 = knn3.predict(X_test)
print('Zbior uczacy dla modelu alternatywnego')
ocen_model_klasyfikacji_binarnej(y_train, y_pred_train_3)
print('\nZbior testowy dla modelu alternatywnego')
ocen_model_klasyfikacji_binarnej(y_test, y_pred_test_3)
#Porownujac modele miedzy soba, widac ze w modelu o parametrach n=3 i wadze = uniform nie zachodzi przetrenowanie, a różnica miedzy trafnoscia
#na modelu uczacym, a testowym jest mniejsza niz w modelu o parametrach n=11 i wadze = distance
#Zarowno model knn3 i knn_best maja tą samą specyficznoc (81%) i wskaznik fałszywie pozytywnych (19%), a więc są lepsze
# w wykrywaniu zaawansowanych osob: Czułość (0.89) jest wyższa niż specyficzność (0.81), co oznacza, że model lepiej radzi sobie z klasyfikacją przypadków pozytywnych (1).
#Problemy z fałszywymi pozytywnymi (klasa 0): Wyższy wskaźnik fałszywie pozytywnych (FPR = 0.19) wskazuje, że obydwa modele 
#częściej błędnie klasyfikuje przypadki klasy niezaawansowanej jako zaawansowane.
#Model knn3 ma wieksza mozliwosc generalizacji, wiec mimo mniejszej o 1 punkt procentowy (85%) trafnosci niz knn_best (86%), moze okazac sie lepszym w przewidywaniu,
#biorac pod uwage mala ilosc danych do analizy - 782 rekordy


def ROC(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - specyficzność')
    plt.ylabel('Czułość')
    plt.title('Krzywa ROC')
    plt.legend(loc="lower right")

y_train_score = knn3.predict_proba(X_train)[:,1]
y_test_score = knn3.predict_proba(X_test)[:,1]

ROC(y_train,y_train_score)
ROC(y_test,y_test_score)
#Z wykladu wiemy, że
#AUC należy (0, 6; O, 7] - słaba,
#AUC należy (0, 7; 0, 8] — akceptowalna,
#AUC należy  (0, 8; O, 9] - dobra,
#AUC należy (0, 9; 1] — bardzo dobra.
#AUC na zbiorze uczacym AUC wynosi 98%, a na zbiorze testowym 91% - miara sukcesu klasyfikatora jest bardzo dobra

from xgboost import XGBClassifier

# Model XGBoost - Hiperparametry do tuningu
xgb_hyperparameters = {
    'n_estimators': [10,25,50, 100, 150,300],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'colsample_bytree': [0.01, 0.1, 0.2, 0.3]
}


xgb_model = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss'),
    param_grid=xgb_hyperparameters,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# Wyświetlenie najlepszych hiperparametrów
print("Najlepsze hiperparametry dla XGBoost:")
print(xgb_model.best_params_)

# Predykcja na zbiorze uczącym i testowym
y_pred_train_xgb = xgb_model.predict(X_train)
y_pred_test_xgb = xgb_model.predict(X_test)

# Ocena modelu na zbiorze uczącym
print("Ocena modelu XGBoost - Zbiór uczący")
ocen_model_klasyfikacji_binarnej(y_train, y_pred_train_xgb)

# Ocena modelu na zbiorze testowym
print("\nOcena modelu XGBoost - Zbiór testowy")
ocen_model_klasyfikacji_binarnej(y_test, y_pred_test_xgb)

# Wyznaczenie krzywej ROC dla modelu XGBoost
y_train_score_xgb = xgb_model.predict_proba(X_train)[:, 1]
y_test_score_xgb = xgb_model.predict_proba(X_test)[:, 1]

#Krzywa ROC - Zbiór uczący
ROC(y_train, y_train_score_xgb)
#Krzywa ROC - Zbiór testowy
ROC(y_test, y_test_score_xgb)