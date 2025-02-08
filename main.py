from fastapi import FastAPI, HTTPException, Query, Security, Body
from fastapi.security.api_key import APIKeyHeader, APIKey
from starlette.status import HTTP_403_FORBIDDEN
from datetime import datetime
from models.stock_predictor import StockPredictor  
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = os.getenv("API_KEY_NAME")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI()

predictor = StockPredictor(
    model_path="paras/DL/mamba.onnx",
    symbol_file="models/DL/symbol.json",
    session_file="models/DL/session.json",
)

async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate API Key"
    )


@app.post("/predict")
async def predict(
    inputData: object = Body(..., description="List of stockSymbol, session, volume, open, high, low, close, MA_7, MA_14, MA_21, RSI", 
                             example={
        "stockID": "user's uuid",
        "stockSymbol": "NVDA",
        "session": "1d",
        "inputData": {
        "volume": [
            182325600, 198634700, 191903300, 194463300, 250132900,
            221866000, 227834900, 309871700, 400946600, 236406200,
            344941900, 190287700, 226370900, 141863200, 171682800,
            164414000, 231224300, 172621200, 188505600, 189308600,
            210020900, 184905200, 159211400, 231514900, 237951100,
            259410300, 275643200, 180000000, 200000000, 210000000,
            220000000, 230000000, 275643200, 
        ],
        "open": [
            148.679993, 146.779999, 149.070007, 147.639999, 144.869995,
            139.500000, 141.320007, 147.410004, 149.350006, 145.929993,
            141.990005, 137.699997, 135.009995, 136.779999, 138.830002,
            138.259995, 142.000000, 145.110001, 144.600006, 138.970001,
            139.009995, 137.360001, 137.080002, 138.940002, 134.179993,
            129.089996, 133.860001, 132.000000, 134.000000, 136.000000,
            138.000000, 140.000000, 138.000000
        ],
        "high": [
            148.850006, 149.649994, 149.330002, 149.000000, 145.240005,
            141.550003, 147.130005, 147.559998, 152.889999, 147.160004,
            142.050003, 139.300003, 137.220001, 139.350006, 140.449997,
            140.539993, 145.789993, 146.539993, 145.699997, 139.949997,
            141.820007, 140.169998, 138.440002, 139.600006, 134.399994,
            131.589996, 136.699997, 138.000000, 140.000000, 142.000000,
            144.000000, 146.000000, 144.000000
        ],
        "low": [
            143.570007, 146.009995, 145.899994, 145.550003, 140.080002,
            137.149994, 140.990005, 142.729996, 140.699997, 141.100006,
            135.820007, 135.669998, 131.800003, 136.050003, 137.820007,
            137.949997, 140.289993, 143.949997, 141.309998, 137.130005,
            133.789993, 135.210007, 135.800003, 132.539993, 130.419998,
            126.860001, 128.279999, 130.000000, 132.000000, 134.000000,
            136.000000, 138.000000, 132.000000
        ],
        "close": [
            148.850006, 149.649994, 149.330002, 149.000000, 145.240005,
            141.550003, 147.130005, 147.559998, 152.889999, 147.160004,
            142.050003, 139.300003, 137.220001, 139.350006, 140.449997,
            140.539993, 145.789993, 146.539993, 145.699997, 139.949997,
            141.820007, 140.169998, 138.440002, 139.600006, 134.399994,
            131.589996, 136.699997, 138.000000, 140.000000, 142.000000,
            144.000000, 146.000000, 138.000000
        ],
        "MA_7": [
            147.135430, 147.427160, 147.708570, 147.705712, 147.294288,
            147.058575, 147.135430, 146.627861, 147.008574, 147.791426,
            146.830000, 145.831431, 145.044287, 144.387146, 143.860004,
            143.728573, 144.121432, 144.451431, 143.491432, 141.891432,
            141.030002, 140.451431, 139.880002, 139.831432, 139.171432,
            138.621432, 138.832860, 139.500000, 140.000000, 141.000000,
            142.000000, 143.000000, 139.500000
        ],
        "MA_14": [
            145.537144, 145.680003, 145.890003, 145.787144, 145.374288,
            145.091432, 144.537144, 144.680003, 144.890003, 144.787144,
            144.374288, 143.491432, 142.645288, 141.874288, 141.081431,
            140.270002, 139.441432, 138.800001, 138.400000, 138.537142,
            139.010001, 139.318572, 139.432861, 139.178575, 138.725002,
            138.110002, 137.672858, 137.500000, 137.800000, 138.200000,
            138.700000, 139.200000, 144.537144
        ],
        "MA_21": [
            144.172860, 144.318572, 144.432861, 144.178575, 143.725002,
            143.110002, 142.672858, 142.172002, 142.418570, 142.872858,
            143.318572, 143.632857, 143.432857, 143.178571, 142.725000,
            142.110000, 141.672856, 141.172001, 141.318570, 141.872857,
            142.318571, 142.632856, 142.432857, 141.878571, 141.472856,
            141.010000, 140.632857, 140.500000, 140.700000, 141.000000,
            141.400000, 141.800000, 140.500000
        ],
        "RSI": [
            60.38, 62.47, 65.92, 63.18, 59.86, 54.29, 51.93, 53.19, 51.13,
            41.18, 44.92, 42.91, 41.15, 42.93, 39.17, 36.91, 42.02, 45.00,
            46.18, 47.85, 50.28, 48.61, 47.22, 46.85, 44.90, 42.18, 43.87,
            44.50, 45.30, 46.20, 47.00, 48.00, 59.86
        ]
        }

    }

),
    timepoints: int = Query(default=1, description="Number of future data points to predict"),
    api_key: APIKey = Security(get_api_key),
    ):
    """
    Predict stock data based on input symbol, session, timepoint and inputData.
    """
    start = datetime.now()
    predictor.timepoints = timepoints

    X_test_tensor_test, stockId = predictor.prepare_data(inputData)    
    mamba_preds = predictor.mamba_predict(X_test_tensor_test[:, :, :7])
    ridge_preds = predictor.ridge_predict(predictor.data)
    garch_preds = predictor.garch_predict(predictor.data, inputData.get('stockSymbol'), inputData.get('session'))

    y_processed, signals = predictor.post_process(mamba_preds, ridge_preds, garch_preds)

    end = datetime.now()

    return {
        "stockID": stockId,
        "predictedPrice": y_processed,
        "signals": signals,
        "timeStamp": datetime.now().timestamp(),
        "processingTime": (end - start).total_seconds(),
    }


@app.get("/")
async def root(): 
    return {"Slogan": "Ngã chỗ nào, gấp đôi chỗ đó !!!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8386,)
