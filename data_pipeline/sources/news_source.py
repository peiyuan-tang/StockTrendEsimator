#!/usr/bin/env python3
"""
News Data Source - Collects news and sentiment data for stocks
"""

import logging
from datetime import datetime
from typing import Dict, List, Any
from data_pipeline.sources.financial_source import BaseDataSource

logger = logging.getLogger(__name__)


class NewsDataSource(BaseDataSource):
    """
    Collects news data for S&P 500 stocks from multiple sources.
    Includes sentiment analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.news_sources = config.get('news_sources', ['finnhub', 'alpha_vantage'])
        self.sentiment_analysis = config.get('sentiment_analysis', False)
        self.api_keys = config.get('api_keys', {})
        
        if self.sentiment_analysis:
            try:
                from textblob import TextBlob
                self.textblob = TextBlob
            except ImportError:
                self.logger.warning("TextBlob not installed. Sentiment analysis disabled.")
                self.sentiment_analysis = False

    def fetch_sp500_tickers(self) -> List[str]:
        """Fetch list of S&P 500 tickers"""
        try:
            import pandas as pd
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp500_df = tables[0]
            tickers = sp500_df['Symbol'].str.replace('.', '-').tolist()
            return tickers[:100]  # Limit for demo
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500 tickers: {str(e)}")
            return []

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        if not self.sentiment_analysis or not text:
            return {'polarity': 0.0, 'subjectivity': 0.0}

        try:
            blob = self.textblob(text)
            return {
                'polarity': float(blob.sentiment.polarity),
                'subjectivity': float(blob.sentiment.subjectivity)
            }
        except Exception as e:
            self.logger.warning(f"Error analyzing sentiment: {str(e)}")
            return {'polarity': 0.0, 'subjectivity': 0.0}

    def fetch_finnhub_news(self) -> List[Dict[str, Any]]:
        """Fetch news from Finnhub API"""
        news_records = []
        
        if 'finnhub' not in self.api_keys:
            self.logger.warning("Finnhub API key not provided")
            return news_records

        try:
            import finnhub
            client = finnhub.Client(api_key=self.api_keys['finnhub'])
            
            tickers = self.fetch_sp500_tickers()
            for ticker in tickers[:20]:  # Limit to avoid API quota
                try:
                    news = client.company_news(ticker, _from="2024-01-01", to="2024-12-31")
                    
                    for article in news[:5]:  # Top 5 articles per ticker
                        sentiment = self.analyze_sentiment(
                            f"{article.get('headline', '')} {article.get('summary', '')}"
                        )
                        
                        record = {
                            'data_type': 'news',
                            'ticker': ticker,
                            'timestamp': datetime.utcnow().isoformat(),
                            'headline': article.get('headline', ''),
                            'summary': article.get('summary', ''),
                            'source': article.get('source', 'finnhub'),
                            'url': article.get('url', ''),
                            'published_at': article.get('datetime', ''),
                            'sentiment': sentiment,
                        }
                        news_records.append(record)
                        
                except Exception as e:
                    self.logger.warning(f"Error fetching news for {ticker}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error with Finnhub source: {str(e)}")

        return news_records

    def fetch_newsapi_news(self) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI"""
        news_records = []
        
        if 'newsapi' not in self.api_keys:
            self.logger.warning("NewsAPI key not provided")
            return news_records

        try:
            import requests
            api_key = self.api_keys['newsapi']
            
            tickers = self.fetch_sp500_tickers()
            for ticker in tickers[:20]:
                try:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        'q': ticker,
                        'sortBy': 'publishedAt',
                        'language': 'en',
                        'apiKey': api_key,
                        'pageSize': 5
                    }
                    
                    response = requests.get(url, params=params, timeout=self.timeout)
                    response.raise_for_status()
                    articles = response.json().get('articles', [])
                    
                    for article in articles:
                        sentiment = self.analyze_sentiment(
                            f"{article.get('title', '')} {article.get('description', '')}"
                        )
                        
                        record = {
                            'data_type': 'news',
                            'ticker': ticker,
                            'timestamp': datetime.utcnow().isoformat(),
                            'headline': article.get('title', ''),
                            'summary': article.get('description', ''),
                            'source': article.get('source', {}).get('name', 'newsapi'),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'sentiment': sentiment,
                        }
                        news_records.append(record)
                        
                except Exception as e:
                    self.logger.warning(f"Error fetching news for {ticker}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error with NewsAPI source: {str(e)}")

        return news_records

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch news data from configured sources"""
        data_records = []

        for source in self.news_sources:
            if source == 'finnhub':
                data_records.extend(self.fetch_finnhub_news())
            elif source == 'newsapi':
                data_records.extend(self.fetch_newsapi_news())

        self.logger.info(f"Fetched {len(data_records)} news articles")
        return data_records
