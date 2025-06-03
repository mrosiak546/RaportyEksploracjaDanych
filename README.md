\# Projekty analizy danych w Pythonie

Repozytorium zawiera dwa niezależne projekty analityczne wykonane w języku Python, obejmujące:
- regresję kosztów ubezpieczenia zdrowotnego,
- klasyfikację poziomu doświadczenia użytkowników siłowni.

## 📁 Zawartość

### 1. `insurance.csv` + `program.py`
**Cel:** przewidywanie kosztów ubezpieczenia zdrowotnego (`charges`) na podstawie cech demograficznych i stylu życia.

**Opis danych (`insurance.csv`):**
- `age`: wiek osoby ubezpieczonej,
- `sex`: płeć (`male`, `female`),
- `bmi`: wskaźnik masy ciała,
- `children`: liczba dzieci na utrzymaniu,
- `smoker`: status palacza (`yes`, `no`),
- `region`: region zamieszkania w USA,
- `charges`: koszt leczenia.

**Podejście:**
- Wstępne czyszczenie i kodowanie danych,
- Użycie regresji liniowej, lasu losowego i drzew wyjątkowo losowych,
- Ocena modeli za pomocą RMSE, MAE, MAPE, R²,
- Analiza istotności predyktorów i wizualizacja wyników.

---

### 2. `gym_members_exercise_tracking.csv` + `program.py`
**Cel:** klasyfikacja poziomu doświadczenia (`Experience_Level`: 0 - początkujący, 1 - zaawansowany) członków siłowni na podstawie ich danych treningowych i zdrowotnych.

**Opis danych (`gym_members_exercise_tracking.csv`):**
- Cechy m.in.:
  - `Age`, `Gender`, `Weightkg`, `Heightm`, `BMI`, `Fat_Percentage`
  - `Workout_Frequencydaysweek`, `Session_Durationhours`, `Calories_Burned`
  - `Max_BPM`, `Avg_BPM`, `Resting_BPM`, `Water_Intakeliters`, `Workout_Type`

**Podejście:**
- Wstępne czyszczenie danych i konwersja typów,
- Analiza korelacji cech ze zmienną celu,
- Selekcja najistotniejszych predyktorów (Workout_Frequency, Duration, Calories),
- Budowa modeli:
  - K-Nearest Neighbors (KNN) z GridSearchCV,
  - XGBoost,
- Ocena za pomocą trafności, F1, czułości, specyficzności i krzywej ROC/AUC,
- Porównanie skuteczności modeli i wykrycie przeuczenia.

---
