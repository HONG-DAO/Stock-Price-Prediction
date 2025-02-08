import torch
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler

from models.ML.ridge import RidgePredictor
from models.Statistic.garch import AutomaticGARCHPredictor

import onnxruntime as ort

# from ML.ridge import RidgePredictor
# from Statistic.garch import AutomaticGARCHPredictor

class StockPredictor:
    def __init__(self, model_path, symbol_file, session_file, timepoints=8):

        self.timepoints = timepoints
        # Create the Mamba model
        self.mamba_predictor = ort.InferenceSession(model_path)

        self.garch_predictor = AutomaticGARCHPredictor(1, 1)
        self.ridge_predictor = RidgePredictor(model_path="paras/ML/ridge_multi_model.pkl",
                                              scaler_path="paras/ML/scaler.pkl",
                                              symbol_file="models/DL/symbol.json",
                                              session_file="models/DL/session.json")

        # Load JSON mapping files
        with open(symbol_file, "r") as f:
            self.symbol_map = json.load(f)
        with open(session_file, "r") as f:
            self.session_map = json.load(f)
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.sequence_length = 32

        self.mamba_weight = 0.35
        self.garch_weight = 0.35
        self.ridge_weight = 0.3
    
    def prepare_data(self, input_features):
        """
        Prepares real-time data for prediction.

        Args:
            input_features (dict): Dictionary containing real-time input features with keys:
                                  ['stockSymbol', 'session', 'volume', 'open', 'high', 'low', 'close',
                                   'MA_7', 'MA_14', 'MA_21', 'RSI']
        Returns:
            torch.Tensor: Tensor of prepared input sequence ready for prediction.
        """
        stockID = input_features.get('stockID')
        stockSymbol = input_features.get('stockSymbol')
        session = input_features.get('session')
        volume = input_features["inputData"].get('volume')
        open_price = input_features["inputData"].get('open')
        high = input_features["inputData"].get('high')
        low = input_features["inputData"].get('low')
        close = input_features["inputData"].get('close')
        ma_7 = input_features["inputData"].get('MA_7')
        ma_14 = input_features["inputData"].get('MA_14')
        ma_21 = input_features["inputData"].get('MA_21')
        rsi = input_features["inputData"].get('RSI')


        # Kiểm tra các giá trị None
        if None in [stockSymbol, session, volume, open_price, high, low, close, ma_7, ma_14, ma_21, rsi]:
            raise ValueError("All features must be provided for real-time prediction.")
        
        # Map stockSymbol and session to numerical IDs
        self.ID_Company = [[self.symbol_map.get(stockSymbol, 0)]*len(volume)]
        self.Session = [[self.session_map.get(session, 0)]*len(volume)]


        # Create a DataFrame for real-time input
        real_time_data = pd.DataFrame({
            'Volume': [volume],
            'Open': [open_price],
            'High': [high],
            'Low': [low],
            'Close': [close],
            'ID_Company': [self.ID_Company[0]],
            'Session': [self.Session[0]],
            'MA_7': [ma_7],
            'MA_14': [ma_14],
            'MA_21': [ma_21],
            'RSI': [rsi],
        }).apply(pd.Series.explode)

        self.data = pd.DataFrame({
            'Volume': [volume],
            'Open': [open_price],
            'High': [high],
            'Low': [low],
            'Close': [close],
            'ID_Company': [self.ID_Company[0]],
            'Session': [self.Session[0]],
            'MA_7': [ma_7],
            'MA_14': [ma_14],
            'MA_21': [ma_21],
            'RSI': [rsi],
        }).apply(pd.Series.explode).tail(33)

        # Normalize the data using the scaler
        scaled_features = self.scaler.fit_transform(real_time_data[['Volume', 'Open', 'High', 'Low', 'Close']])

        # Add normalized data back to real_time_data
        real_time_data[['Volume', 'Open', 'High', 'Low', 'Close']] = scaled_features
        # Create a sequence tensor for real-time prediction
        real_time_sequence = real_time_data.values.astype(float)
        real_time_tensor = torch.tensor(real_time_sequence, dtype=torch.float32).unsqueeze(0).to("cpu")
        # print("Real-time data prepared for inference.")
        return real_time_tensor, stockID

    
    def mamba_predict(self, X_test_tensor):

        # Predict future timepoints
        last_sequence = X_test_tensor[-1].cpu().numpy()
        future_predictions = []
        for _ in range(self.timepoints):
            input_onnx = {self.mamba_predictor.get_inputs()[0].name: X_test_tensor[:, 1:, :].numpy()}
            future_pred = self.mamba_predictor.run(None, input_onnx)[0][:, -1, :]
            future_pred_inv = self.scaler.inverse_transform(future_pred)
            future_predictions.append(future_pred_inv[0])
            # Update the last sequence with the new prediction
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1, 0:5] = future_pred[0]

        return np.array(future_predictions)

    def ridge_predict(self, data):
        X_test = self.ridge_predictor.prepare_data(data)
        # Predict future timepoints
        last_sequence = X_test
        future_predictions = []
        for _ in range(self.timepoints):
            future_pred = self.ridge_predictor.predict(last_sequence)
            future_predictions.append(list(future_pred[0]))

            # Update the last sequence with the new prediction
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1, 0:5] = future_pred[0]
            
        output = list(future_predictions)
        # output[0] = int(output[0][0])
        return np.array(output)

    def garch_predict(self, data, symbol, session, col_list=['Open', 'High', 'Low', 'Close']):
        garch_preds = []
        for col in col_list:
            col_preds = []
            for i in range(len(data) - self.sequence_length):
                preds = self.garch_predictor.predict_next_point(data[col].iloc[i:i+self.sequence_length].values, symbol, session)
                col_preds.append(preds)
            garch_preds.append(col_preds)
        garch_preds = np.array(garch_preds).T.squeeze(0)[-self.timepoints:]  # Shape: [Timepoints, 4]
        return garch_preds

    def post_process(self, mamba_preds, ridge_preds, garch_preds):
        # Combine predictions
        y_test_pred_combined = (
            self.mamba_weight * mamba_preds[-self.timepoints:, 1:5] +  # Combine open, high, low, close
            self.garch_weight * garch_preds[-self.timepoints:] +
            self.ridge_weight * ridge_preds[-self.timepoints, 1:5]
        )

        # Calculate technical indicators
        self.data['MA_7'] = self.data['Close'].rolling(window=7).mean()
        self.data['MA_14'] = self.data['Close'].rolling(window=14).mean()
        self.data['MA_21'] = self.data['Close'].rolling(window=21).mean()

        def rsi(series, period=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        self.data['RSI'] = rsi(self.data['Close'])

        # Apply post-processing logic
        post_processed_prices = {col: [] for col in ['Open', 'High', 'Low', 'Close']}
        signals = []
        for i in range(21, len(self.data)):
            if i - 21 >= len(y_test_pred_combined):
                break  # Prevent out-of-bounds access

            ma_7 = self.data['MA_7'].iloc[i]
            ma_14 = self.data['MA_14'].iloc[i]
            ma_21 = self.data['MA_21'].iloc[i]
            rsi_value = self.data['RSI'].iloc[i]
            predicted_ohlc = y_test_pred_combined[i - 21]

            # Adjust prices based on logic
            for idx, col in enumerate(['Open', 'High', 'Low', 'Close']):
                if ma_7 > ma_14 > ma_21:
                    if rsi_value > 70:
                        post_processed_prices[col].append(predicted_ohlc[idx] * 0.98)
                    elif rsi_value > 50:
                        post_processed_prices[col].append(predicted_ohlc[idx] * 1.02)
                    else:
                        post_processed_prices[col].append(predicted_ohlc[idx])
                elif ma_7 < ma_14 < ma_21:
                    if rsi_value < 30:
                        post_processed_prices[col].append(predicted_ohlc[idx] * 1.02)
                    elif rsi_value < 50:
                        post_processed_prices[col].append(predicted_ohlc[idx] * 0.98)
                    else:
                        post_processed_prices[col].append(predicted_ohlc[idx])
                else:
                    post_processed_prices[col].append(predicted_ohlc[idx])

            if ma_7 > ma_14 > ma_21 and rsi_value > 70:
                signals.append('Caution - Overbought')
            elif ma_7 < ma_14 < ma_21 and rsi_value < 30:
                signals.append('Caution - Oversold')
            elif ma_7 > ma_14 > ma_21:
                signals.append('Bullish')
            elif ma_7 < ma_14 < ma_21:
                signals.append('Bearish')
            else:
                signals.append('Neutral')

        # Add NaN padding for initial rows
        for col in post_processed_prices:
            while len(post_processed_prices[col]) < len(self.data):
                post_processed_prices[col].insert(0, np.nan)

        for col in ['Open', 'High', 'Low', 'Close']:
            self.data[f'PostProcessed_{col}'] = post_processed_prices[col]

        # Show volume predictions (no processing logic applied)

        post_processed_volume = np.array(mamba_preds[-self.timepoints:, 0], dtype=int).tolist()
        # post_processed_volume = np.array(ridge_preds[-self.timepoints:, 0], dtype=int).tolist()
        post_processed_prices["Volume"] = post_processed_volume
        # print("Post-processing completed with updated logic.")



        return {col: post_processed_prices[col][-self.timepoints:] for col in ['Volume', 'Open', 'High', 'Low', 'Close',]}, signals[-self.timepoints:]


if __name__ == "__main__":
    pass
