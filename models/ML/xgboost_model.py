# import os
# import joblib
# import optuna
# import numpy as np
# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.multioutput import MultiOutputRegressor
# from xgboost import XGBRegressor
# from DS.models.ML.base.base_model import BaseModel

# class XGBoostModel(BaseModel):
#     def __init__(self,
#                  weights_dir='paras/ML/paras_save/weights/xgboost', 
#                  models_dir='paras/ML/paras_save/models/xgboost', 
#                  n_outputs=5,
#                  random_state=42):
#         """
#         Khởi tạo mô hình XGBoost đa đầu ra.
#         """
#         super().__init__(weights_dir=weights_dir, models_dir=models_dir)
#         self.n_outputs = n_outputs
#         self.random_state = random_state
#         self.model = None
#         self.weights = None
#         print("XGBoostModel đã được khởi tạo.")

#     def _objective(self, trial, X, y):
#         """
#         Hàm objective cho optuna study để tối ưu hyperparameters XGBoost.
#         """
#         # Một số hyperparameters phổ biến
#         n_estimators = trial.suggest_int('n_estimators', 50, 1000)
#         learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.3)
#         max_depth = trial.suggest_int('max_depth', 3, 10)
#         subsample = trial.suggest_uniform('subsample', 0.5, 1.0)
#         colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.5, 1.0)

#         base_model = XGBRegressor(
#             n_estimators=n_estimators,
#             learning_rate=learning_rate,
#             max_depth=max_depth,
#             subsample=subsample,
#             colsample_bytree=colsample_bytree,
#             random_state=self.random_state,
#             n_jobs=-1
#         )

#         multi_model = MultiOutputRegressor(base_model)
#         # Sử dụng K-Fold để đánh giá mô hình
#         kf = KFold(n_splits=2, shuffle=True, random_state=self.random_state)
#         scores = cross_val_score(multi_model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
#         return scores.mean()

#     def train(self, X, y, n_trials=1, direction='maximize', study_name='xgboost_study'):
#         """
#         Huấn luyện mô hình XGBoost đa đầu ra với tối ưu bằng optuna.
#         """
#         study = optuna.create_study(direction=direction, study_name=study_name, load_if_exists=True)
#         study.optimize(lambda trial: self._objective(trial, X, y), n_trials=n_trials)

#         print("Hyperparameters tốt nhất:", study.best_params)
#         print("Best score:", study.best_value)

#         # Huấn luyện mô hình với hyperparameters tối ưu
#         best_params = study.best_params
#         best_base_model = XGBRegressor(
#             n_estimators=best_params['n_estimators'],
#             learning_rate=best_params['learning_rate'],
#             max_depth=best_params['max_depth'],
#             subsample=best_params['subsample'],
#             colsample_bytree=best_params['colsample_bytree'],
#             random_state=self.random_state,
#             n_jobs=-1
#         )

#         self.model = MultiOutputRegressor(best_base_model)
#         self.model.fit(X, y)
#         self.weights = best_params

#         print("Mô hình XGBoost đa đầu ra đã được huấn luyện với hyperparameters tối ưu.")

#         # Lưu trọng số và mô hình
#         self.save_weights(prefix='xgboost_weights')
#         self.save_model(prefix='xgboost_model')

#     def predict(self, X):
#         """
#         Dự đoán giá trị mục tiêu từ dữ liệu đầu vào X.
#         """
#         if self.model:
#             return self.model.predict(X)
#         else:
#             raise Exception("Mô hình chưa được huấn luyện.")

#     def summary_model(self):
#         """
#         In ra thông tin tổng quan về mô hình.
#         """
#         if self.model:
#             print(self.model)
#         else:
#             print("Không có mô hình nào được cung cấp.")
