import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load JSON mapping files
with open("symbol.json", "r") as f:
    symbol_map = json.load(f)

with open("session.json", "r") as f:
    session_map = json.load(f)

# # Hàm để xử lý từng file của từng phiên và mã cổ phiếu
def process_data(file_path, symbol, session, sequence_length=32, training_size=0.8):
    # Đọc dữ liệu
    df = pd.read_csv(file_path)
    
    # Thêm ID_Company và Session vào dữ liệu
    df['ID_Company'] = symbol_map[symbol]
    df['Session'] = session_map[session]
    
    # Chọn cột cần scale, bao gồm cả nhãn (y)
    data_to_scale = df[['Volume', 'Open', 'High', 'Low', 'Close', "Percentage Change", "Volatility", "Trading Value", "MA_7", "MA_14", "MA_21", "MA_30", "MA_60", "MA_100", "MA_200", "MACD", "Signal Line", "MACD Histogram", "RSI", "Middle Band", "Upper Band", "Lower Band", "%K", "%D", "OBV"]]
    
    # Scale toàn bộ dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_scale)
    
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_y = y_scaler.fit_transform(df[['Close']])

    # Chuyển dữ liệu scaled thành DataFrame để dễ thao tác
    scaled_df = pd.DataFrame(scaled_data, columns=['Volume', 'Open', 'High', 'Low', 'Close', "Percentage Change", "Volatility", "Trading Value", "MA_7", "MA_14", "MA_21", "MA_30", "MA_60", "MA_100", "MA_200", "MACD", "Signal Line", "MACD Histogram", "RSI", "Middle Band", "Upper Band", "Lower Band", "%K", "%D", "OBV"])
    
    # Kết hợp các cột đã scale với các cột giữ nguyên
    final_features = pd.concat([scaled_df, df[['ID_Company', 'Session']].reset_index(drop=True)], axis=1)

    # Chia thành các sequence sample
    X, y = [], []
    for i in range(len(final_features) - sequence_length):
        X.append(final_features.iloc[i:i+sequence_length].values)
        y.append(scaled_df.iloc[i+sequence_length].values)

    X, y = np.array(X), np.array(y)

    # Split thành tập train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_size, shuffle=False)

    return X_train, y_train, X_test, y_test, scaler, y_scaler


# Lấy tất cả các file trong thư mục
def shuffle_data(data_dir="../../data", sequence_length=32, training_size=0.8):
    X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []
    scaler = None
    
    for session_folder in os.listdir(data_dir):
        session_path = os.path.join(data_dir, session_folder)
        if os.path.isdir(session_path):
            for stock_file in os.listdir(session_path):
                stock_file_path = os.path.join(session_path, stock_file)
                if stock_file.endswith('.csv'):
                    symbol = stock_file.split('_')[0]  # Lấy mã cổ phiếu từ tên file
                    session = session_folder  # Lấy phiên từ thư mục
                    
                    # Xử lý từng phiên và từng mã cổ phiếu
                    # X_train, y_train, X_test, y_test, feature_scaler, y_scaler = process_data(stock_file_path, symbol, session, sequence_length, training_size)
                    X_train, y_train, X_test, y_test, _, y_scaler = process_data(stock_file_path, symbol, session, sequence_length, training_size)
                    
                    # Gộp lại các dữ liệu cho tất cả các phiên và mã cổ phiếu
                    X_train_list.append(X_train)
                    y_train_list.append(y_train)
                    X_test_list.append(X_test)
                    y_test_list.append(y_test)
    
    # Gộp lại tất cả các tập dữ liệu từ các phiên và mã cổ phiếu khác nhau
    X_train_final = np.concatenate(X_train_list, axis=0)
    y_train_final = np.concatenate(y_train_list, axis=0)
    X_test_final = np.concatenate(X_test_list, axis=0)
    y_test_final = np.concatenate(y_test_list, axis=0)

    # Shuffle dữ liệu
    shuffle_train_indices = np.random.permutation(X_train_final.shape[0])
    shuffle_test_indices = np.random.permutation(X_test_final.shape[0])

    X_train_shuffle = X_train_final[shuffle_train_indices]
    y_train_shuffle = y_train_final[shuffle_train_indices]
    X_test_shuffle = X_test_final[shuffle_test_indices]
    y_test_shuffle = y_test_final[shuffle_test_indices]

    return X_train_shuffle, y_train_shuffle, X_test_shuffle, y_test_shuffle, y_scaler

