
# Credit Scoring: Default Probability Prediction

This project is a machine learning competition solution for a fintech company. The goal is to build a classifier that predicts the probability of a client entering serious delinquency (default) within 90 days after loan issuance, based on application data, credit history, and transactions.

## Evaluation Metric
The primary evaluation metric for the model is **ROC-AUC**.

## Tech Stack
* **Programming Language:** Python
* **Data Analysis & Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, CatBoost, LightGBM

## Data
Five datasets were provided for this task:
1. `train.csv` / `test.csv` — Main application information (age, income, education, region, loan parameters).
2. `bureau.csv` — External credit history from Credit Bureaus (limits, current debts, delinquencies).
3. `previous_loans.csv` — History of previous loans within the company itself.
4. `transactions.csv` — Account transaction history prior to the application date.

## Solution Pipeline

### 1. Exploratory Data Analysis (EDA)
* Analyzed the distribution of the target variable. Identified a moderate class imbalance (66% to 34%).
* Built correlation matrices and distribution histograms to detect anomalies and duplicate features.

### 2. Data Leakage Prevention
During the analysis, features containing "future" information (generated after the loan was issued) were identified. Using them led to overfitting (ROC-AUC ~0.94 on cross-validation) and a significant drop in performance on the actual test set. The following columns were removed:
* `post_loan_collection_score`
* `days_until_first_overdue`
* `internal_decision_code`

### 3. Feature Engineering
* Auxiliary datasets were grouped by clients (`groupby`) to extract aggregated statistics (sums, means, maximums, operation counts).
* Custom financial burden features were generated:
  * `debt_to_income` — ratio of total debts to monthly income.
  * `loan_to_income` — ratio of the requested loan amount to income.

### 4. Modeling and Validation
A 5-fold cross-validation (StratifiedKFold) was used for robust evaluation. Three models were trained and compared:

| Model | ROC-AUC (CV) |
| :--- | :--- |
| Logistic Regression (Baseline) | 0.7496 |
| LightGBM | 0.8104 |
| **CatBoost (Final Model)** | **0.8247** |

CatBoost was chosen as the final model because it showed the best performance and natively handles categorical features and missing values out of the box.

## Usage
1. Install the required libraries: `pip install -r requirements.txt`
2. Download the datasets and place them in the root directory (or mount Google Drive).
3. Run the `competition.ipynb` notebook from top to bottom. Upon completion, the `submission.csv` file with the predictions will be generated.

# Кредитный скоринг: предсказание вероятности дефолта

Проект представляет собой решение соревнования по машинному обучению для финтех-компании. Цель — построить классификатор, который на основе данных анкеты, кредитной истории и транзакций клиента предсказывает вероятность его выхода в серьёзную просрочку (дефолт) в течение 90 дней после выдачи займа.

## Метрика качества
Основная метрика оценки модели — ROC-AUC.

## Стек
* Язык программирования: Python
* Анализ и обработка данных: Pandas, NumPy
* Визуализация: Matplotlib, Seaborn
* Машинное обучение: Scikit-learn, CatBoost, LightGBM

## Данные
В распоряжении было 5 датасетов:
1. `train.csv` / `test.csv` — основная информация по заявкам (возраст, доход, образование, регион, параметры кредита).
2. `bureau.csv` — внешняя кредитная история клиента из БКИ (лимиты, текущие долги, просрочки).
3. `previous_loans.csv` — история предыдущих займов внутри самой компании.
4. `transactions.csv` — история транзакций по счету до момента подачи заявки.

## Ход решения

### 1. Разведочный анализ данных (EDA)
* Проведён анализ распределения целевой переменной. Выявлен умеренный дисбаланс классов (66% к 34%).
* Построены матрицы корреляций и гистограммы распределений для поиска аномалий и дублирующихся признаков.

### 2. Предотвращение утечки данных (Data Leakage)
В процессе анализа были выявлены признаки, содержащие информацию из "будущего" (формируются уже после выдачи займа). Их использование привело к переобучению (ROC-AUC ~0.94 на кросс-валидации) и падению качества на реальном тесте. Эти колонки были удалены:
* `post_loan_collection_score`
* `days_until_first_overdue`
* `internal_decision_code`

### 3. Feature Engineering
* Дополнительные датасеты были сгруппированы по клиентам (`groupby`) с извлечением агрегированных статистик (суммы, средние, максимумы, количество операций).
* Сгенерированы кастомные признаки финансовой нагрузки:
  * `debt_to_income` — отношение общих долгов к ежемесячному доходу.
  * `loan_to_income` — отношение запрашиваемой суммы к доходу.

### 4. Моделирование и валидация
Для устойчивой оценки использовалась 5-фолдовая кросс-валидация (StratifiedKFold). Были обучены и сравнены 3 модели:

| Модель | ROC-AUC (CV) |
| :--- | :--- |
| Logistic Regression (Baseline) | 0.7496 |
| LightGBM | 0.8104 |
| CatBoost (Финальная модель) | **0.8247** |

В качестве финальной модели был выбран CatBoost, так как он показал наилучший результат и позволяет из коробки обрабатывать категориальные признаки и пропущенные значения.

## Использование
1. Установить необходимые библиотеки: `pip install -r requirements.txt`
2. Скачать датасеты и поместить их в корневую директорию (или подключить Google Drive).
3. Запустить блокнот `competition.ipynb` сверху вниз. В конце выполнения будет сгенерирован файл с ответом `submission.csv`.

