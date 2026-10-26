import pandas as pd
from src.data_loader import load_data
from src.features import build_features
from src.models import train_model

def main():
    print("Загрузка данных...")
    data = load_data()
    
    print("Генерация признаков...")
    X_train, y_train, X_test, test_ids = build_features(data)
    
    print("Обучение модели...")
    model, cat_features = train_model(X_train, y_train)
    
    print("Подготовка предсказаний...")
    for col in cat_features:
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna("Missing").astype(str)
            
    predictions = model.predict_proba(X_test)[:, 1]
    

    submission = test_ids.copy()
    submission["score"] = predictions
    submission.to_csv("submission.csv", index=False)
    print("Файл submission.csv сохранен.")

if __name__ == "__main__":
    main()