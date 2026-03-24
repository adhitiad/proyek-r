import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class AutoencoderAnomalyDetector(nn.Module):
    """Autoencoder untuk anomaly detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SystemHealthMonitor:
    """Self-healing system dengan anomaly detection"""
    
    def __init__(self):
        self.autoencoder = None
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.anomaly_threshold = 0.1
        
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for anomaly detection"""
        features = []
        
        # Price features
        features.append(df['Close'].pct_change().values)
        features.append(df['Volume'].pct_change().values)
        features.append(df['High'] - df['Low'])
        
        # Technical indicators
        features.append(df['Close'].rolling(20).mean() / df['Close'] - 1)
        features.append(df['Close'].rolling(20).std() / df['Close'])
        
        # Clean NaN
        feature_array = np.column_stack([f for f in features if len(f) == len(df)])
        feature_array = feature_array[~np.isnan(feature_array).any(axis=1)]
        
        return feature_array
    
    def train(self, df: pd.DataFrame):
        """Train anomaly detection models"""
        features = self.extract_features(df)
        
        # Normalize
        features_scaled = self.scaler.fit_transform(features)
        
        # Train autoencoder
        self.autoencoder = AutoencoderAnomalyDetector(input_dim=features_scaled.shape[1])
        self._train_autoencoder(features_scaled)
        
        # Train isolation forest
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest.fit(features_scaled)
        
        logger.info("Anomaly detection models trained")
    
    def _train_autoencoder(self, data: np.ndarray, epochs: int = 100):
        """Train autoencoder"""
        data_tensor = torch.FloatTensor(data)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.autoencoder.train()
            optimizer.zero_grad()
            reconstructed = self.autoencoder(data_tensor)
            loss = criterion(reconstructed, data_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"Autoencoder epoch {epoch}: Loss {loss.item():.6f}")
    
    def detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect anomalies in current data"""
        features = self.extract_features(df)
        if len(features) == 0:
            return {'has_anomaly': False, 'score': 0}
        
        features_scaled = self.scaler.transform(features)
        
        # Autoencoder reconstruction error
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed = self.autoencoder(torch.FloatTensor(features_scaled))
            recon_error = torch.mean((reconstructed - torch.FloatTensor(features_scaled)) ** 2, dim=1).numpy()
        
        # Isolation forest score
        iforest_score = self.isolation_forest.score_samples(features_scaled)
        
        # Combined anomaly score
        anomaly_score = (recon_error > self.anomaly_threshold).mean()
        
        return {
            'has_anomaly': anomaly_score > 0.3,
            'anomaly_score': float(anomaly_score),
            'reconstruction_error': float(recon_error.mean()),
            'isolation_score': float(iforest_score.mean()),
            'anomaly_indices': np.where(recon_error > self.anomaly_threshold)[0].tolist()
        }
    
    def heal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Heal anomalies by imputing with synthetic data"""
        anomalies = self.detect_anomalies(df)
        
        if not anomalies['has_anomaly']:
            return df
        
        # Create synthetic data for anomalous points
        healed_df = df.copy()
        
        for idx in anomalies['anomaly_indices']:
            if idx > 0 and idx < len(df) - 1:
                # Interpolate between neighbors
                healed_df.iloc[idx] = (df.iloc[idx-1] + df.iloc[idx+1]) / 2
        
        logger.info(f"Healed {len(anomalies['anomaly_indices'])} anomalies")
        
        return healed_df

class SelfHealingSystem:
    """Main self-healing system"""
    
    def __init__(self):
        self.monitors = {}
        self.health_check_interval = 60  # seconds
        
    def register_monitor(self, name: str, monitor: SystemHealthMonitor):
        """Register a monitor for a data source"""
        self.monitors[name] = monitor
    
    async def check_and_heal(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Check all monitors and heal if needed"""
        healed_data = {}
        
        for name, df in data.items():
            if name in self.monitors:
                monitor = self.monitors[name]
                anomaly_report = monitor.detect_anomalies(df)
                
                if anomaly_report['has_anomaly']:
                    logger.warning(f"Anomaly detected in {name}: {anomaly_report}")
                    healed_data[name] = monitor.heal(df)
                else:
                    healed_data[name] = df
        
        return healed_data
    
    def get_health_status(self) -> Dict:
        """Get overall system health status"""
        return {
            'monitors': list(self.monitors.keys()),
            'status': 'healthy',
            'last_check': pd.Timestamp.now().isoformat()
        }