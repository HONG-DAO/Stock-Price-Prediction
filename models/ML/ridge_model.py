# import os
# import json
# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import cross_val_score
# from DS.utils.DataLoader.data_pross import shuffle_data
# import optuna
# from optuna.samplers import TPESampler

# def training(
#     data_dir="DS/data",
#     model_path="DS/paras/ML/ridge_multi_model.pkl",
#     scaler_path="DS/paras/ML/scaler.pkl",
#     sequence_length=21,
#     training_size=0.8,
#     n_trials=100,
#     cv=5,
#     random_state=42
# ):
#     """
#     Huấn luyện mô hình Ridge sử dụng MultiOutputRegressor với tối ưu hóa Bayesian bằng Optuna.

#     Parameters:
#     - data_dir (str): Đường dẫn tới thư mục dữ liệu.
#     - model_path (str): Đường dẫn để lưu mô hình đã huấn luyện.
#     - scaler_path (str): Đường dẫn để lưu scaler.
#     - sequence_length (int): Độ dài chuỗi dữ liệu.
#     - training_size (float): Tỷ lệ dữ liệu dùng để huấn luyện.
#     - n_trials (int): Số lượng thử nghiệm tối ưu hóa Optuna.
#     - cv (int): Số fold cho cross-validation.
#     - random_state (int): Hạt giống cho tái lập ngẫu nhiên.

#     Returns:
#     - dict: Kết quả bao gồm các chỉ số đánh giá mô hình và tham số tốt nhất.
#     """
#     # Load và xử lý dữ liệu
#     X_train, y_train, X_test, y_test, scaler = shuffle_data(
#         data_dir=data_dir, sequence_length=sequence_length, training_size=training_size
#     )

#     # Reshape dữ liệu cho MultiOutputRegressor
#     n_samples, seq_len, n_features = X_train.shape
#     X_train_flat = X_train.reshape(n_samples, seq_len * n_features)
#     X_test_flat = X_test.reshape(X_test.shape[0], seq_len * n_features)

#     # Định nghĩa hàm mục tiêu cho Optuna
#     def objective(trial):
#         alpha = trial.suggest_float('alpha', 0.001, 70)
#         base_model = Ridge(alpha=alpha, random_state=random_state)
#         multi_model = MultiOutputRegressor(base_model)
#         scores = cross_val_score(
#             multi_model, X_train_flat, y_train, cv=cv, scoring='r2', n_jobs=-1
#         )
#         return scores.mean()

#     # Tạo study Optuna
#     sampler = TPESampler(seed=random_state)
#     study = optuna.create_study(direction='maximize', sampler=sampler)
#     study.optimize(objective, n_trials=n_trials)

#     # Kết quả tối ưu hóa
#     best_alpha = study.best_params['alpha']

#     # Huấn luyện mô hình với tham số tốt nhất
#     best_base_model = Ridge(alpha=best_alpha, random_state=random_state)
#     best_model = MultiOutputRegressor(best_base_model)
#     best_model.fit(X_train_flat, y_train)

#     # Dự đoán và tính toán các chỉ số
#     y_pred = best_model.predict(X_test_flat)
#     mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
#     r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

#     # Lưu mô hình và scaler
#     joblib.dump(best_model, model_path)
#     joblib.dump(scaler, scaler_path)

#     # Lưu tham số tốt nhất
#     best_params_path = os.path.join(os.path.dirname(model_path), 'best_params.json')
#     with open(best_params_path, 'w') as json_file:
#         json.dump(study.best_params, json_file, indent=4)

#     print(json.dumps({
#         "Best Params": study.best_params,
#         "Test R^2 Score": r2,
#         "Test MSE": mse,
#         "Test RMSE": rmse,
#         "Test MAE": mae
#     }, indent=4))

#     return {
#         "best_params": study.best_params,
#         "r2_score": r2,
#         "mse": mse,
#         "rmse": rmse,
#         "mae": mae
#     }

# # if __name__ == "__main__":
# #     results = training()
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from DS.utils.DataLoader.data_pross import shuffle_data
import optuna
from optuna.samplers import TPESampler

def training(
    data_dir="DS/data",
    model_path="DS/paras/ML/ridge_multi_model.pkl",
    scaler_path="DS/paras/ML/scaler.pkl",
    sequence_length=21,
    training_size=0.8,
    n_trials=100,
    cv=5,
    random_state=42
):
    """
    Train a Ridge model using MultiOutputRegressor with Bayesian optimization via Optuna.

    Parameters:
    - data_dir (str): Path to the data directory.
    - model_path (str): Path to save the trained model.
    - scaler_path (str): Path to save the scaler.
    - sequence_length (int): Length of the data sequence.
    - training_size (float): Proportion of data used for training.
    - n_trials (int): Number of Optuna optimization trials.
    - cv (int): Number of cross-validation folds.
    - random_state (int): Seed for reproducibility.

    Returns:
    - dict: Contains evaluation metrics and best hyperparameters.
    """
    # Load and preprocess data
    X_train, y_train, X_test, y_test, scaler = shuffle_data(
        data_dir=data_dir, sequence_length=sequence_length, training_size=training_size
    )

    # Reshape data for MultiOutputRegressor
    n_samples, seq_len, n_features = X_train.shape
    X_train_flat = X_train.reshape(n_samples, seq_len * n_features)
    X_test_flat = X_test.reshape(X_test.shape[0], seq_len * n_features)

    # Define the objective function for Optuna
    def objective(trial):
        alpha = trial.suggest_float('alpha', 0.001, 70)
        base_model = Ridge(alpha=alpha, random_state=random_state)
        multi_model = MultiOutputRegressor(base_model)
        scores = cross_val_score(
            multi_model, X_train_flat, y_train, cv=cv, scoring='r2', n_jobs=-1
        )
        return scores.mean()

    # Create and optimize the study
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Retrieve the best hyperparameters
    best_alpha = study.best_params['alpha']

    # Train the model with the best hyperparameters
    best_base_model = Ridge(alpha=best_alpha, random_state=random_state)
    best_model = MultiOutputRegressor(best_base_model)
    best_model.fit(X_train_flat, y_train)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test_flat)

    # Calculate evaluation metrics
    # Chuyển đổi về giá trị gốc
    y_test_original = scaler.inverse_transform(y_test)
    y_pred_original = scaler.inverse_transform(y_pred)

    # Dự đoán và tính toán các chỉ số
    mse = mean_squared_error(y_test_original, y_pred_original, multioutput='uniform_average')
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original, multioutput='uniform_average')
    r2 = r2_score(y_test_original, y_pred_original, multioutput='uniform_average')
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

    # Save the model and scaler
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save the best hyperparameters
    best_params_path = os.path.join(os.path.dirname(model_path), 'best_params.json')
    with open(best_params_path, 'w') as json_file:
        json.dump(study.best_params, json_file, indent=4)

    # Print the results
    print(json.dumps({
        "Best Params": study.best_params,
        "Test R^2 Score": r2,
        "Test MSE": mse,
        "Test RMSE": rmse,
        "Test MAE": mae,
        "Test MAPE ": mape 
    }, indent=4))

    # Visualization: Plot y_test vs y_pred for each target variable
    target_names = ['Volume', 'Open', 'High', 'Low', 'Close']
    num_targets = y_test.shape[1]

    # Ensure the number of target names matches the number of targets
    if len(target_names) != num_targets:
        raise ValueError(f"Number of target names ({len(target_names)}) does not match number of targets ({num_targets}).")

    # Create a directory to save plots
    plots_dir = os.path.join(os.path.dirname(model_path), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Visualization: Scatter plot y_test vs y_pred for each target variable
    for i, target in enumerate(target_names):
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test_original[:, i], y_pred_original[:, i], alpha=0.5, color='blue')
        plt.plot(
            [min(y_test_original[:, i]), max(y_test_original[:, i])],
            [min(y_test_original[:, i]), max(y_test_original[:, i])],
            color='red', linestyle='--', label='Perfect Prediction'
        )
        plt.title(f'Scatter Plot: Actual vs Predicted for {target.capitalize()}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.tight_layout()

        # Save scatter plot
        scatter_plot_path = os.path.join(plots_dir, f'{target}_scatter_plot1.png')
        plt.savefig(scatter_plot_path)
        plt.close()
        print(f'Scatter plot saved: {scatter_plot_path}')



    return {
        "best_params": study.best_params,
        "r2_score": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape" : mape
    }

# Example usage:
# if __name__ == "__main__":
#     results = training()
