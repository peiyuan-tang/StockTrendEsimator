#!/usr/bin/env python3
"""
Stock Movement Trend Source - Collects stock trend data and technical indicators
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
from data_pipeline.models.financial_source import BaseDataSource

logger = logging.getLogger(__name__)


class StockMovementSource(BaseDataSource):
    """
    Collects stock movement trends for S&P 500 stocks.
    Calculates technical indicators: SMA_20, SMA_50, RSI, MACD
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.interval_minutes = config.get('interval_minutes', 60)
        self.indices = config.get('indices', ['SP500'])
        self.include_indicators = config.get('include_indicators', [])
        try:
            import yfinance as yf
            import pandas_ta as ta
            self.yf = yf
            self.ta = ta
        except ImportError as e:
            self.logger.error(f"Required package not installed: {str(e)}")
            raise

    def fetch_sp500_tickers(self) -> List[str]:
        """Fetch list of S&P 500 tickers"""
        try:
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp500_df = tables[0]
            tickers = sp500_df['Symbol'].str.replace('.', '-').tolist()
            return tickers[:100]  # Limit for demo, use all in production
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500 tickers: {str(e)}")
            return []

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators"""
        indicators = {}

        try:
            if 'SMA_20' in self.include_indicators:
                indicators['SMA_20'] = float(df['Close'].rolling(window=20).mean().iloc[-1])

            if 'SMA_50' in self.include_indicators:
                indicators['SMA_50'] = float(df['Close'].rolling(window=50).mean().iloc[-1])

            if 'RSI' in self.include_indicators:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                indicators['RSI'] = float(rsi.iloc[-1])

            if 'MACD' in self.include_indicators:
                ema_12 = df['Close'].ewm(span=12).mean()
                ema_26 = df['Close'].ewm(span=26).mean()
                macd = ema_12 - ema_26
                indicators['MACD'] = float(macd.iloc[-1])
                indicators['MACD_Signal'] = float(ema_12.ewm(span=9).mean().iloc[-1])

        except Exception as e:
            self.logger.warning(f"Error calculating indicators: {str(e)}")

        return indicators

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch stock movement data and technical indicators"""
        data_records = []
        tickers = self.fetch_sp500_tickers()

        for ticker in tickers[:50]:  # Process subset
            try:
                stock = self.yf.Ticker(ticker)
                hist = stock.history(period='3mo')  # 3 months of history for indicators

                if len(hist) >= 50:  # Ensure enough data for indicators
                    indicators = self.calculate_indicators(hist)
                    latest = hist.iloc[-1]

                    record = {
                        'data_type': 'stock_movement',
                        'ticker': ticker,
                        'timestamp': datetime.utcnow().isoformat(),
                        'price': float(latest['Close']),
                        'price_change': float(latest['Close'] - hist.iloc[-2]['Close']),
                        'price_change_percent': float(
                            ((latest['Close'] - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close']) * 100
                        ),
                        'high_52week': float(hist['High'].max()),
                        'low_52week': float(hist['Low'].min()),
                        'avg_volume_90d': float(hist['Volume'].tail(90).mean()),
                        'indicators': indicators,
                    }
                    data_records.append(record)
                    self.logger.debug(f"Fetched movement data for {ticker}")

            except Exception as e:
                self.logger.warning(f"Error fetching movement data for {ticker}: {str(e)}")

        return data_records
