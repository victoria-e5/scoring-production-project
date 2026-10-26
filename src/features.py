import pandas as pd

def build_features(data_dict):
    train = data_dict["train"]
    test = data_dict["test"]
    bureau = data_dict["bureau"]
    
    bureau_agg = bureau.groupby("client_id")["credit_limit"].mean().reset_index()
    bureau_agg.rename(columns={"credit_limit": "bureau_credit_limit_mean"}, inplace=True)
    
    train = train.merge(bureau_agg, on="client_id", how="left")
    test = test.merge(bureau_agg, on="client_id", how="left")
    
    y_train = train["target"] if "target" in train.columns else None
    X_train = train.drop(columns=["target", "application_id", "client_id"], errors="ignore")
    X_test = test.drop(columns=["application_id", "client_id"], errors="ignore")
    
   
    X_test = X_test[X_train.columns]
    
    return X_train, y_train, X_test, test[["application_id", "client_id"]]