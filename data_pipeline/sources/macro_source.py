#!/usr/bin/env python3
"""
Macroeconomic Data Source - Collects macro indicators for Mag 7 stocks
"""

import logging
from datetime import datetime
from typing import Dict, List, Any
from data_pipeline.sources.financial_source import BaseDataSource

logger = logging.getLogger(__name__)


class MacroeconomicDataSource(BaseDataSource):
    """
    Collects macroeconomic data for Mag 7 stocks:
    - Interest rate
    - Unemployment rate
    - GDP growth
    - Inflation rate
    - Fed funds rate
    """

    MAG7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.update_frequency = config.get('update_frequency', 'daily')
        self.indicators = config.get('indicators', [])
        self.api_keys = config.get('api_keys', {})
        
        try:
            import pandas_datareader as pdr
            self.pdr = pdr
        except ImportError:
            self.logger.error("pandas_datareader not installed")
            raise

    def fetch_fred_data(self) -> Dict[str, Any]:
        """Fetch data from Federal Reserve Economic Data (FRED)"""
        data = {}

        # FRED series IDs
        fred_series = {
            'interest_rate': 'DGS10',  # 10-year treasury yield
            'unemployment_rate': 'UNRATE',  # Unemployment rate
            'gdp_growth': 'A191RL1Q225SBEA',  # Real GDP growth rate
            'inflation_rate': 'CPIAUCSL',  # CPI inflation
            'fed_funds_rate': 'FEDFUNDS',  # Fed funds rate
        }

        try:
            if 'fred' not in self.api_keys:
                self.logger.warning("FRED API key not provided")
                return data

            for indicator, series_id in fred_series.items():
                if indicator not in self.indicators:
                    continue

                try:
                    df = self.pdr.get_data_fred(
                        series_id,
                        api_key=self.api_keys['fred'],
                        start='2024-01-01'
                    )
                    
                    if not df.empty:
                        latest = df.iloc[-1]
                        data[indicator] = {
                            'value': float(latest),
                            'date': str(df.index[-1].date()),
                            'unit': self._get_unit(indicator)
                        }
                        self.logger.debug(f"Fetched {indicator}: {latest}")

                except Exception as e:
                    self.logger.warning(f"Error fetching {indicator}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error fetching FRED data: {str(e)}")

        return data

    def fetch_world_bank_data(self) -> Dict[str, Any]:
        """Fetch data from World Bank API"""
        data = {}

        world_bank_indicators = {
            'gdp_growth': 'NY.GDP.MKTP.KD.ZS',
            'inflation_rate': 'FP.CPI.TOTL.ZG',
        }

        try:
            import requests
            
            for indicator, wb_code in world_bank_indicators.items():
                if indicator not in self.indicators:
                    continue

                try:
                    url = f"https://api.worldbank.org/v2/country/USA/indicator/{wb_code}"
                    params = {'format': 'json', 'date': '2023'}
                    
                    response = requests.get(url, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    
                    result = response.json()
                    if len(result) > 1 and result[1]:
                        latest = result[1][0]
                        value = latest.get('value')
                        
                        if value:
                            data[indicator] = {
                                'value': float(value),
                                'date': latest.get('date'),
                                'unit': self._get_unit(indicator),
                                'source': 'worldbank'
                            }
                            self.logger.debug(f"Fetched {indicator} from World Bank: {value}")

                except Exception as e:
                    self.logger.warning(f"Error fetching {indicator} from World Bank: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error with World Bank source: {str(e)}")

        return data

    def fetch_alpha_vantage_macro(self) -> Dict[str, Any]:
        """Fetch macro data from Alpha Vantage"""
        data = {}

        if 'alpha_vantage' not in self.api_keys:
            self.logger.warning("Alpha Vantage API key not provided")
            return data

        try:
            import requests
            api_key = self.api_keys['alpha_vantage']

            if 'interest_rate' in self.indicators:
                try:
                    url = "https://www.alphavantage.co/query"
                    params = {
                        'function': 'TREASURY_YIELD',
                        'interval': 'monthly',
                        'maturity': '10year',
                        'apikey': api_key
                    }
                    
                    response = requests.get(url, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    result = response.json()
                    
                    if 'data' in result and result['data']:
                        latest = result['data'][0]
                        data['interest_rate'] = {
                            'value': float(latest['value']),
                            'date': latest['date'],
                            'unit': '%',
                            'source': 'alpha_vantage'
                        }
                        self.logger.debug(f"Fetched interest rate: {latest['value']}")

                except Exception as e:
                    self.logger.warning(f"Error fetching interest rate: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error with Alpha Vantage source: {str(e)}")

        return data

    def _get_unit(self, indicator: str) -> str:
        """Get unit for indicator"""
        units = {
            'interest_rate': '%',
            'unemployment_rate': '%',
            'gdp_growth': '%',
            'inflation_rate': '%',
            'fed_funds_rate': '%',
        }
        return units.get(indicator, 'unknown')

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch macroeconomic data from multiple sources"""
        data_records = []

        # Combine data from all sources
        macro_data = {}
        macro_data.update(self.fetch_fred_data())
        macro_data.update(self.fetch_world_bank_data())
        macro_data.update(self.fetch_alpha_vantage_macro())

        if macro_data:
            # Create a single record for macro data (applies to all Mag 7 stocks)
            record = {
                'data_type': 'macroeconomic',
                'timestamp': datetime.utcnow().isoformat(),
                'applicable_stocks': self.MAG7_TICKERS,
                'indicators': macro_data,
                'source': 'aggregated'
            }
            data_records.append(record)
            self.logger.info(f"Fetched macroeconomic data with {len(macro_data)} indicators")

        return data_records
