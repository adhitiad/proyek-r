import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from datetime import datetime
from app.core.database import db
from app.core.data_collector import DataCollector
from app.core.technical_analysis import TechnicalAnalysis
from app.core.sentiment_analysis import SentimentAnalyzer

class SignalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(SignalModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class ModelTrainer:
    def __init__(self, symbol_list, start_date, end_date, target_days=5):
        self.symbol_list = symbol_list
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.target_days = target_days
        self.ta = TechnicalAnalysis()
        self.sentiment = SentimentAnalyzer()

    def prepare_data(self):
        all_features = []
        all_labels = []
        for symbol in self.symbol_list:
            df = DataCollector.get_price_data(symbol, period="3mo", interval="1d")
            if df.empty:
                continue
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            if len(df) < self.target_days + 20:
                continue

            # Hitung fitur teknikal
            df['returns'] = df['Close'].pct_change()
            df['rsi'] = self.ta.rsi(df['Close'])
            macd, signal = self.ta.macd(df['Close'])
            df['macd'] = macd
            df['macd_signal'] = signal
            bb_upper, bb_lower = self.ta.bollinger_bands(df['Close'])
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower

            # Sentimen (konstan per simbol)
            sentiment_score = self.sentiment.analyze_news(symbol)
            df['sentiment'] = sentiment_score

            # Label: apakah harga naik setelah target_days
            df['future_price'] = df['Close'].shift(-self.target_days)
            df['label'] = (df['future_price'] > df['Close']).astype(int)

            df.dropna(inplace=True)

            feature_cols = ['returns', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'sentiment']
            # Normalisasi per fitur
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(df[feature_cols])
            labels = df['label'].values

            all_features.extend(features_scaled)
            all_labels.extend(labels)

        # Simpan scaler dan feature_cols untuk digunakan nanti
        self.scaler = scaler
        self.feature_cols = feature_cols
        return np.array(all_features), np.array(all_labels)

    def train(self, epochs=100, batch_size=32, learning_rate=0.001):
        X, y = self.prepare_data()
        if len(X) == 0:
            raise ValueError("No data available for training")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = SignalModel(input_dim=X.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t)
            test_preds = (test_preds > 0.5).float()
            accuracy = (test_preds == y_test_t).float().mean().item()

        # Simpan model dan scaler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        year_month = datetime.now().strftime("%Y-%m")
        model_dir = f"models/{year_month}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{timestamp}.pt")
        scaler_path = os.path.join(model_dir, f"scaler_{timestamp}.pkl")

        torch.save(model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)

        metadata = {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "timestamp": datetime.now(),
            "accuracy": accuracy,
            "input_dim": X.shape[1],
            "feature_cols": self.feature_cols,
            "target_days": self.target_days,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "symbols": self.symbol_list,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat()
        }
        db.model_metadata.insert_one(metadata)

        return metadata, accuracy