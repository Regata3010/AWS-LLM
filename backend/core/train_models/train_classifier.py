# # src/train_models/train_classifier.py

# import sys
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from src.logger import logging
# from src.exception import CustomException
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# import streamlit as st
# from core.bias_detector.preprocessor import preprocess_data
# import sys


# def get_model(choice):
#     if choice  == "Random Forest Classifier":
#         return RandomForestClassifier()
#     elif choice == "Logistic Regression":
#         return LogisticRegression()
#     elif choice == "XGBoost":
#         return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#     else:
#         raise CustomException(e,sys)
    
    
# def train_classification_model(X_train, X_test, y_train, y_test, model_choice):
#     try:
#         model = get_model(model_choice)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         acc = accuracy_score(y_test, y_pred)
#         report = classification_report(y_test, y_pred)

#         logging.info(f"✅ Accuracy: {acc}")
#         logging.info(f"📋 Report: \n{report}")

#         return model, y_pred

#     except Exception as e:
#         raise CustomException(e)










