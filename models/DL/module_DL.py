import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
class DataLoader_DL:
    @staticmethod
    def load_data(filepath, file_type='csv'):
        """
        Tải dữ liệu từ nhiều định dạng file khác nhau: csv, excel, sql, v.v.
        Tham số:
            - filepath: Đường dẫn đến file dữ liệu.
            - file_type: Loại file ('csv', 'excel', v.v.). Mặc định là 'csv'.
        Trả về:
            - DataFrame chứa dữ liệu từ file.
        """
        if file_type == 'csv':
            data = pd.read_csv(filepath)
        elif file_type == 'excel':
            data = pd.read_excel(filepath)
        else:
            raise ValueError(f"Loại file không hỗ trợ: {file_type}")
        return data

    @staticmethod
    def convert_to_tensors(X_sequences, y_sequences):
        """
        Chuyển đổi dữ liệu thành tensor của PyTorch.
        
        Tham số:
            - X_sequences: Mảng numpy chứa các chuỗi đầu vào.
            - y_sequences: Mảng numpy chứa các nhãn tương ứng.
        
        Trả về:
            - X_tensor: Tensor của PyTorch từ X_sequences.
            - y_tensor: Tensor của PyTorch từ y_sequences.
        """
        X_tensor = torch.tensor(X_sequences, dtype=torch.float32)
        y_tensor = torch.tensor(y_sequences, dtype=torch.float32)
        return X_tensor, y_tensor

    @staticmethod
    def split_data(X_tensor, y_tensor, train_ratio=0.7, val_ratio=0.15):
        """
        Chia dữ liệu thành các tập huấn luyện, kiểm định và kiểm tra.
        
        Tham số:
            - X_tensor: Tensor đầu vào.
            - y_tensor: Tensor nhãn đầu ra.
            - train_ratio: Tỷ lệ dữ liệu huấn luyện (mặc định là 0.7).
            - val_ratio: Tỷ lệ dữ liệu kiểm định (mặc định là 0.15).
        
        Trả về:
            - X_train, y_train, X_val, y_val, X_test, y_test: Các tập dữ liệu được chia.
        """
        train_size = int(train_ratio * len(X_tensor))
        val_size = int(val_ratio * len(X_tensor))
        test_size = len(X_tensor) - train_size - val_size
        
        X_train, y_train = X_tensor[:train_size], y_tensor[:train_size]
        X_val, y_val = X_tensor[train_size:train_size + val_size], y_tensor[train_size:train_size + val_size]
        X_test, y_test = X_tensor[train_size + val_size:], y_tensor[train_size + val_size:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        """
        Tạo DataLoader cho mỗi tập dữ liệu.
        
        Tham số:
            - X_train, y_train: Tập dữ liệu huấn luyện.
            - X_val, y_val: Tập dữ liệu kiểm định.
            - X_test, y_test: Tập dữ liệu kiểm tra.
            - batch_size: Kích thước mỗi batch (mặc định là 32).
        
        Trả về:
            - train_loader, val_loader, test_loader: DataLoader cho các tập dữ liệu.
        """

        train_data = TensorDataset(X_train, y_train)
        val_data = TensorDataset(X_val, y_val)
        test_data = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader


    @staticmethod
    def split_features_labels(data, feature_columns, label_column):
        """
        Chia dữ liệu thành các biến đầu vào (features) và nhãn đầu ra (labels).
        Tham số:
            - data: DataFrame chứa toàn bộ dữ liệu.
            - feature_columns: Danh sách các cột dữ liệu đầu vào.
            - label_column: Cột nhãn đầu ra.
        Trả về:
            - X: Ma trận các biến đầu vào.
            - y: Mảng chứa nhãn đầu ra.
        """
        X = data[feature_columns].values
        y = data[label_column].values
        return X, y
    
    @staticmethod
    def create_sequences(data, sequence_length):
        """
        Tạo các chuỗi con từ dữ liệu và nhãn tương ứng.
        
        Tham số:
            - data: Dữ liệu đầu vào, có thể là mảng hoặc ma trận.
            - sequence_length: Độ dài của mỗi chuỗi con.

        Trả về:
            - sequences: Mảng các chuỗi con được tạo ra từ dữ liệu đầu vào.
            - labels_out: Mảng các nhãn tương ứng với mỗi chuỗi, đại diện cho giá trị cần dự đoán tiếp theo sau mỗi chuỗi.
        """

        sequences = []
        labels = []
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            label = data[i + sequence_length]  # Predict the next timestep
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)


class Evaluation:
    def __init__(self, model, criterion, data_loader, device):
        """
        Khởi tạo lớp Evaluation với mô hình, hàm mất mát, data loader và thiết bị.

        :param model: Mô hình PyTorch đã được huấn luyện
        :param criterion: Hàm mất mát
        :param data_loader: Đối tượng DataLoader cho tập dữ liệu kiểm tra
        :param device: Thiết bị ('cpu' hoặc 'cuda') để chạy việc đánh giá
        """
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.device = device

    def evaluate_model(self):
        """
        Đánh giá mô hình sử dụng tập dữ liệu kiểm tra.

        :return: Giá trị mất mát trung bình cho tập dữ liệu kiểm tra
        """
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for sequences, labels in self.data_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(self.data_loader)
        print(f'Test Loss: {avg_test_loss:.4f}')
        return avg_test_loss

class Prediction:
    def __init__(self, model, device):
        """
        Khởi tạo lớp Prediction với mô hình và thiết bị.

        :param model: Mô hình PyTorch đã được huấn luyện
        :param device: Thiết bị ('cpu' hoặc 'cuda') để chạy việc dự đoán
        """
        self.model = model
        self.device = device

    
    def make_predictions(self, data, timepoints):
        self.model.eval() # Set the model to evaluation mode
        predictions = []
        current_sequence = list(data[-timepoints:]) 

        with torch.no_grad():
            for _ in range(timepoints):
                # Chuyển current_sequence thành tensor và đưa vào mô hình để dự đoán
                seq = torch.tensor(current_sequence, dtype=torch.float32).to(self.device)
                pred = self.model(seq)         
                next_value = pred.cpu().numpy().flatten()[0]  # Chuyển từ tensor sang numpy và lấy giá trị
                predictions.append(next_value)
                
                current_sequence = current_sequence[-(timepoints - 1):]
                current_sequence.append(next_value) 

        return np.array(predictions)

class DL:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @staticmethod
    def load_model(model_class, *model_args, weights_file=None, device=None):
        """ Tải một mô hình từ một lớp và các tham số được cung cấp, có thể tùy chọn tải trọng số từ một tệp.

        :param model_class: Lớp của mô hình cần khởi tạo.
        :param model_args: Các tham số để khởi tạo mô hình.
        :param weights_file: Đường dẫn tới tệp trọng số cần tải.
        :param device: Thiết bị để tải mô hình.
        """
        model = model_class(*model_args)
        if device is not None:
            model.to(device)
        if weights_file and os.path.exists(weights_file):
            state_dict = torch.load(weights_file, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Model weights loaded successfully from {weights_file}")
        else:
            print(f"Model initialized without pre-trained weights.")
        return model

    @staticmethod
    def load_weights(model, weights_file):
        """ Tải trọng số vào mô hình từ một tệp được cung cấp.

        :param model: Thực thể mô hình để nạp trọng số vào.
        :param weights_file: Đường dẫn tới tệp trọng số cần tải.
        """
        if os.path.exists(weights_file):
            state_dict = torch.load(weights_file, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Model weights loaded successfully from {weights_file}")
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_file}")


    def summary_model(model, input_size):
        """ In ra tóm tắt của mô hình bao gồm các lớp và tham số.

        :param model: Mô hình cần được tóm tắt.
        :param input_size: Một tuple biểu diễn kích thước của đầu vào.
        """
        from torchinfo import summary
        summary(model, input_size=input_size)

    def train(self, train_loader, criterion, optimizer, num_epochs):
        """
        Huấn luyện mô hình với tập dữ liệu huấn luyện.

        :param train_loader: DataLoader cho tập dữ liệu huấn luyện
        :param criterion: Hàm mất mát
        :param optimizer: Bộ tối ưu hóa
        :param num_epochs: Số lượng epochs
        """
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for sequences, labels in train_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)

                # Backward pass và tối ưu hóa
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    @staticmethod
    def save_weights(model, weights_file):
        """
        Save weights of the given model to a file.

        :param model: Model instance whose weights are to be saved.
        :param weights_file: Path to the file to save weights.
        """
        torch.save(model.state_dict(), weights_file)
        print(f"Model weights saved successfully to {weights_file}")
