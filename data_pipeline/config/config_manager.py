#!/usr/bin/env python3
"""
Data Pipeline Configuration Manager
Manages pipeline settings, API keys, and credentials
"""

import logging
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class APICredentials:
    """API credentials storage"""
    finnhub_api_key: Optional[str] = None
    newsapi_key: Optional[str] = None
    alpha_vantage_key: Optional[str] = None
    fred_api_key: Optional[str] = None
    yfinance_cookie: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class DataPipelineConfig:
    """Main pipeline configuration"""
    
    # Data sources
    mag7_tickers: list = None
    sp500_limit: int = 500
    
    # Collection frequencies
    financial_data_interval: int = 3600  # seconds
    stock_movement_interval: int = 3600
    news_interval: int = 3600
    macro_interval: int = 86400  # daily
    policy_interval: int = 604800  # weekly
    
    # Storage
    data_root_path: str = '/data'
    backup_enabled: bool = True
    retention_days: int = 90
    
    # Processing
    batch_size_financial: int = 100
    batch_size_movement: int = 500
    batch_size_news: int = 50
    batch_size_macro: int = 10
    
    # Logging
    log_level: str = 'INFO'
    log_path: str = '/var/log/stock_pipeline'

    def __post_init__(self):
        if self.mag7_tickers is None:
            self.mag7_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']


class ConfigManager:
    """Manages pipeline configuration"""

    CONFIG_FILE = 'data_pipeline/config/pipeline_config.json'
    CREDENTIALS_FILE = 'data_pipeline/config/credentials.json'

    def __init__(self):
        self.config = DataPipelineConfig()
        self.credentials = APICredentials()
        self._load_configuration()

    def _load_configuration(self):
        """Load configuration from files"""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)
                    for key, value in config_data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                logger.info(f"Loaded configuration from {self.CONFIG_FILE}")
        except Exception as e:
            logger.warning(f"Error loading config: {str(e)}")

        try:
            if os.path.exists(self.CREDENTIALS_FILE):
                with open(self.CREDENTIALS_FILE, 'r') as f:
                    cred_data = json.load(f)
                    for key, value in cred_data.items():
                        if hasattr(self.credentials, key):
                            setattr(self.credentials, key, value)
                logger.info(f"Loaded credentials from {self.CREDENTIALS_FILE}")
        except Exception as e:
            logger.warning(f"Error loading credentials: {str(e)}")

    def save_configuration(self):
        """Save configuration to file"""
        try:
            Path(os.path.dirname(self.CONFIG_FILE)).mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(asdict(self.config), f, indent=2, default=str)
            logger.info(f"Saved configuration to {self.CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")

    def save_credentials(self):
        """Save credentials to file"""
        try:
            Path(os.path.dirname(self.CREDENTIALS_FILE)).mkdir(parents=True, exist_ok=True)
            with open(self.CREDENTIALS_FILE, 'w') as f:
                json.dump(self.credentials.to_dict(), f, indent=2)
            logger.info(f"Saved credentials to {self.CREDENTIALS_FILE}")
            os.chmod(self.CREDENTIALS_FILE, 0o600)  # Restrict permissions
        except Exception as e:
            logger.error(f"Error saving credentials: {str(e)}")

    def get_api_keys(self) -> Dict[str, str]:
        """Get all API keys for data sources"""
        return {
            'finnhub': self.credentials.finnhub_api_key,
            'newsapi': self.credentials.newsapi_key,
            'alpha_vantage': self.credentials.alpha_vantage_key,
            'fred': self.credentials.fred_api_key,
        }

    def set_api_key(self, service: str, api_key: str):
        """Set API key for a service"""
        service_attr = f"{service}_api_key"
        if hasattr(self.credentials, service_attr):
            setattr(self.credentials, service_attr, api_key)
            self.save_credentials()
            logger.info(f"API key set for {service}")
        else:
            logger.warning(f"Unknown service: {service}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'config': asdict(self.config),
            'credentials': self.credentials.to_dict(),
        }


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
