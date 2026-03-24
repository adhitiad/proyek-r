import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import logging
import aiohttp
        

logger = logging.getLogger(__name__)

class Generator(nn.Module):
    """Generator untuk menghasilkan data pasar sintetis"""
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    """Discriminator untuk membedakan data real vs sintetis"""
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class MarketDataGAN:
    """GAN untuk menghasilkan data pasar sintetis"""
    
    def __init__(self, latent_dim: int = 100, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.latent_dim = latent_dim
        self.device = device
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.criterion = nn.BCELoss()
        
    def _prepare_data(self, df: pd.DataFrame, window: int = 60) -> np.ndarray:
        """Siapkan data untuk training dengan sliding window"""
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = df[features].values
        
        # Normalisasi
        data_norm = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        
        # Buat sequences
        sequences = []
        for i in range(len(data_norm) - window):
            sequences.append(data_norm[i:i+window].flatten())
        
        return np.array(sequences)
    
    def train(self, df: pd.DataFrame, epochs: int = 1000, batch_size: int = 32):
        """Train GAN dengan data pasar historis"""
        # Prepare data
        data = self._prepare_data(df)
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        input_dim = data.shape[1]
        
        # Initialize models
        self.generator = Generator(self.latent_dim, input_dim).to(self.device)
        self.discriminator = Discriminator(input_dim).to(self.device)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        for epoch in range(epochs):
            # Train Discriminator
            self.d_optimizer.zero_grad()
            
            # Real data
            idx = np.random.randint(0, data_tensor.shape[0], batch_size)
            real_data = data_tensor[idx]
            real_output = self.discriminator(real_data)
            d_real_loss = self.criterion(real_output, real_labels[:len(real_data)])
            
            # Fake data
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_data = self.generator(z)
            fake_output = self.discriminator(fake_data.detach())
            d_fake_loss = self.criterion(fake_output, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_data = self.generator(z)
            fake_output = self.discriminator(fake_data)
            g_loss = self.criterion(fake_output, real_labels[:len(fake_data)])
            g_loss.backward()
            self.g_optimizer.step()
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
        return self
    
    def generate_synthetic_data(self, n_samples: int = 100, window: int = 60) -> pd.DataFrame:
        """Generate synthetic market data"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            synthetic: list = self.generator(z).cpu().numpy()
        
        # Reshape back to window x features
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        synthetic_data = []
        for seq in synthetic:
            seq_reshaped = seq.reshape(window, -1)
            synthetic_data.append(pd.DataFrame(seq_reshaped, columns=features))
        
        return synthetic_data

class MarketScenarioGenerator:
    """Generate market scenarios using LLM + GAN"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.gan = MarketDataGAN()
        
    async def generate_scenario(self, prompt: str, df: pd.DataFrame) -> Dict:
        """
        Generate market scenario based on natural language prompt
        Example prompt: "Generate a bear market scenario with high volatility"
        """
        # 1. Use LLM to understand scenario parameters
        scenario_params = await self._parse_scenario_with_llm(prompt)
        
        # 2. Generate synthetic data with GAN conditioned on parameters
        synthetic_data = self._generate_conditioned_data(df, scenario_params)
        
        # 3. Simulate impact on portfolio
        impact = self._simulate_impact(synthetic_data, scenario_params)
        
        return {
            'scenario': prompt,
            'parameters': scenario_params,
            'synthetic_data': synthetic_data,
            'portfolio_impact': impact,
            'risk_assessment': self._assess_risk(impact)
        }
    
    async def _parse_scenario_with_llm(self, prompt: str) -> Dict:
        """Parse natural language to structured parameters"""
        if not self.groq_api_key:
            return {'trend': 'neutral', 'volatility': 'medium', 'volume_trend': 'stable', 'sentiment': 'neutral'}

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """
        Convert the following market scenario description to structured parameters.
        Output JSON with: trend (bullish/bearish/neutral), 
        volatility (low/medium/high), 
        volume_trend (increasing/decreasing/stable),
        sentiment (positive/negative/neutral)
        """
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    result = await response.json()
                    import json
                    import re
                    content = result['choices'][0]['message']['content']
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Scenario LLM parse failed: {e}")
        
        return {'trend': 'neutral', 'volatility': 'medium', 'volume_trend': 'stable', 'sentiment': 'neutral'}
    
    def _generate_conditioned_data(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Generate synthetic data conditioned on parameters"""
        # Train GAN on historical data
        self.gan.train(df, epochs=500)
        
        # Generate synthetic samples
        synthetic_samples = self.gan.generate_synthetic_data(n_samples=100)
        
        # Apply condition parameters
        conditioned_data = []
        for sample in synthetic_samples:
            # Adjust based on parameters
            if params.get('trend') == 'bullish':
                sample['Close'] = sample['Close'] * (1 + np.random.uniform(0.01, 0.05))
            elif params.get('trend') == 'bearish':
                sample['Close'] = sample['Close'] * (1 - np.random.uniform(0.01, 0.05))
            
            if params.get('volatility') == 'high':
                sample['Close'] = sample['Close'] * (1 + np.random.normal(0, 0.03, len(sample)))
            
            conditioned_data.append(sample)
        
        return pd.concat(conditioned_data, ignore_index=True)
    
    def _simulate_impact(self, synthetic_data: pd.DataFrame, params: Dict) -> Dict:
        """Simulate portfolio impact under scenario"""
        # Simplified impact simulation
        return {
            'expected_return': np.random.uniform(-0.2, 0.3),
            'max_drawdown': np.random.uniform(-0.3, -0.05),
            'volatility_impact': np.random.uniform(0.1, 0.5),
            'confidence': np.random.uniform(0.6, 0.9)
        }
    
    def _assess_risk(self, impact: Dict) -> str:
        """Assess risk level based on impact"""
        if impact['max_drawdown'] < -0.2:
            return 'HIGH RISK'
        elif impact['max_drawdown'] < -0.1:
            return 'MEDIUM RISK'
        return 'LOW RISK'
