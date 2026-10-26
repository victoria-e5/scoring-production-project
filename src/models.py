from catboost import CatBoostClassifier

def train_model(X_train, y_train):
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
    

    for col in cat_features:
        X_train[col] = X_train[col].fillna("Missing").astype(str)
    
    model = CatBoostClassifier(
        iterations=500, 
        learning_rate=0.05, 
        eval_metric="AUC", 
        random_seed=42
    )
    
    model.fit(X_train, y_train, cat_features=cat_features, verbose=100)
    
    return model, cat_features