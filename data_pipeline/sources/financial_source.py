#!/usr/bin/env python3
"""
Financial Data Source - Collects raw financial data from APIs
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from flume.source import AbstractSource
from flume.event import Event

logger = logging.getLogger(__name__)


class BaseDataSource(AbstractSource, ABC):
    """Base class for all data sources"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.batch_size = config.get('batch_size', 100)
        self.timeout = config.get('timeout', 30)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch raw data from source"""
        pass

    def process(self, batch_size: int = None) -> List[Event]:
        """Process data and convert to Flume events"""
        batch_size = batch_size or self.batch_size
        events = []

        try:
            data = self.fetch_data()
            for record in data[:batch_size]:
                event = Event(
                    headers={
                        'timestamp': datetime.utcnow().isoformat(),
                        'source': self.__class__.__name__,
                        'data_type': record.get('data_type', 'unknown')
                    },
                    body=record
                )
                events.append(event)
            self.logger.info(f"Processed {len(events)} events from {self.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")

        return events


class FinancialDataSource(BaseDataSource):
    """
    Collects financial data for Mag 7 stocks:
    AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
    """

    MAG7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ticker_list = config.get('ticker_list', self.MAG7_TICKERS)
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            self.logger.error("yfinance not installed. Install with: pip install yfinance")
            raise

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch financial data from Yahoo Finance"""
        data_records = []

        for ticker in self.ticker_list:
            try:
                stock = self.yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period='1d')

                if not hist.empty:
                    latest = hist.iloc[-1]
                    record = {
                        'data_type': 'financial_data',
                        'ticker': ticker,
                        'timestamp': datetime.utcnow().isoformat(),
                        'price': float(latest['Close']),
                        'open': float(latest['Open']),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'volume': int(latest['Volume']),
                        'market_cap': info.get('marketCap', None),
                        'pe_ratio': info.get('trailingPE', None),
                        'dividend_yield': info.get('dividendYield', None),
                        '52_week_high': info.get('fiftyTwoWeekHigh', None),
                        '52_week_low': info.get('fiftyTwoWeekLow', None),
                    }
                    data_records.append(record)
                    self.logger.debug(f"Fetched financial data for {ticker}")

            except Exception as e:
                self.logger.warning(f"Error fetching data for {ticker}: {str(e)}")

        return data_records
