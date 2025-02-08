import pandas as pd
import os

# Hàm tính toán các đặc trưng kỹ thuật cho dữ liệu chứng khoán
def calculate_features(data, company_id, session):
    """
    Tính toán các chỉ báo kỹ thuật dựa trên dữ liệu giá chứng khoán.
    
    data: DataFrame chứa dữ liệu chứng khoán với các cột 'Date', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'.
    company_id: Mã công ty.
    
    Returns: DataFrame chứa các đặc trưng kỹ thuật chỉ với các cột cần thiết.
    """
    print("Starting feature calculation...")  # Debug
    print("Initial columns:", data.columns)  # Debug

    # Xác định cột thời gian và chuyển đổi sang datetime nếu cần
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data.set_index('Date', inplace=True)
    elif 'Datetime' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
        data.set_index('Datetime', inplace=True)
    else:
        print("Missing 'Date' or 'Datetime' column")  # Debug
        return None

    print("Index type after setting datetime:", data.index.dtype)  # Debug

    data = data.sort_index()

    # Tính toán các chỉ báo kỹ thuật
    data['Percentage Change'] = data['Close'].pct_change()
    data['Volatility'] = (data['High'] - data['Low']) / data['Close']
    data['Trading Value'] = data['Close'] * data['Volume']

    for window in [7, 14, 21, 30, 60, 100, 200]:
        data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()

    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD Histogram'] = data['MACD'] - data['Signal Line']

    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['Middle Band'] = data['Close'].rolling(window=20).mean()
    data['Upper Band'] = data['Middle Band'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower Band'] = data['Middle Band'] - 2 * data['Close'].rolling(window=20).std()

    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['%K'] = (data['Close'] - low_14) * 100 / (high_14 - low_14)
    data['%D'] = data['%K'].rolling(window=3).mean()

    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv

    data['Session'] = session 
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(method='bfill')

    data['ID_Company'] = company_id
    data.reset_index(inplace=True)

    # Lọc lại các cột chỉ để lấy các cột cần thiết
    columns_to_save = ['Date' if 'Date' in data.columns else 'Datetime', 'ID_Company','Session', 'Open', 'High', 'Low', 'Close', 'Adj Close',
                       'Volume', 'Percentage Change', 'Volatility', 'Trading Value', 'MA_7', 'MA_14', 'MA_21', 
                       'MA_30', 'MA_60', 'MA_100', 'MA_200', 'MACD', 'Signal Line', 'MACD Histogram', 'RSI',
                       'Middle Band', 'Upper Band', 'Lower Band', '%K', '%D', 'OBV']
    
    print("Final columns for saving:", columns_to_save)  # Debug
    return data[columns_to_save]

# Hàm xác định session từ tên file
def determine_session_from_filename(filename):
    sessions = ['5m', '30m', '1h', '4h', '1d', '1wk']
    for session in sessions:
        if session in filename:
            return session
    return None

# Hàm xử lý dữ liệu từ file CSV của từng công ty
def process_company_data(company_folder, company_id):
    for item in os.listdir(company_folder):
        path = os.path.join(company_folder, item)
        print(f"Processing file: {item}")  # Debug

        # Đọc và tính toán đặc trưng từ file CSV
        data = pd.read_csv(path)
        
        session = determine_session_from_filename(item)
        if session is None:
            print(f"Không tìm thấy session hợp lệ trong tên file {item}")
            continue
        
        print("Data columns on load:", data.columns)  # Debug
        feature_data = calculate_features(data, company_id,session)

        if feature_data is None:
            print(f"Không tìm thấy cột 'Date' hoặc 'Datetime' trong file {item}")
            continue

        session_folder = os.path.join('temp_feature_dataset', session)
        os.makedirs(session_folder, exist_ok=True)

        output_path = os.path.join(session_folder, f'{item}')
        feature_data.to_csv(output_path, index=False)
        print(f"Đã lưu dữ liệu đặc trưng cho {item} vào {output_path}")

# Hàm xử lý tất cả các công ty theo phiên giao dịch
def process_all_companies(stock_dataset):
    for company in os.listdir(stock_dataset):
        company_folder = os.path.join(stock_dataset, company)
        if os.path.isdir(company_folder):
            print(f"Processing company: {company}")  # Debug
            process_company_data(company_folder, company_id=company)
            
    return 'temp_feature_dataset'

# # Chạy xử lý dữ liệu cho tất cả các công ty
# process_all_companies('stock_dataset')
