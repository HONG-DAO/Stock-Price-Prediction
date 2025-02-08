import json
import numpy as np
import pandas as pd
import joblib

class RidgePredictor:
    def __init__(self, 
                 model_path="../../paras/ML/ridge_multi_model.pkl", 
                 scaler_path="../../paras/ML/scaler.pkl", 
                 symbol_file="../DL/symbol.json", 
                 session_file="../DL/session.json", 
                 timepoints=1, 
                 sequence_length=21):
        
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.symbol_file = symbol_file
        self.session_file = session_file
        self.timepoints = timepoints
        self.sequence_length = sequence_length

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        with open(self.symbol_file, "r") as f:
            self.symbol_map = json.load(f)
        with open(self.session_file, "r") as f:
            self.session_map = json.load(f)


    def prepare_data(self, data):

        self.data = data
        scaled_features = self.scaler.transform(self.data[['Volume', 'Open', 'High', 'Low', 'Close']])
        self.scaled_data = pd.DataFrame(scaled_features, 
                                        columns=['Volume', 'Open', 'High', 'Low', 'Close'], 
                                        index=self.data.index)


        X_test = []
        for i in range(len(self.scaled_data) - self.sequence_length + 1):
            X_test.append(self.scaled_data.iloc[i:i + self.sequence_length][['Volume', 'Open', 'High', 'Low', 'Close']].values)
        X_test = np.array(X_test)
        n_samples, seq_len, n_features = X_test.shape
        X_test_flat = X_test.reshape(n_samples, seq_len * n_features)

        self.X_test_flat = X_test_flat[-self.timepoints:]
        return self.X_test_flat
    
    def predict(self, x):

        y_pred = self.model.predict(x)
        y_pred_inv = self.scaler.inverse_transform(y_pred)

        return y_pred_inv
    

if __name__ == "__main__":
    pass