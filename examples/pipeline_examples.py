#!/usr/bin/env python3
"""
Example usage of the Stock Trend Estimator Data Pipeline
"""

import logging
from datetime import datetime, timedelta
from data_pipeline.client.pipeline_client import get_data_client
from data_pipeline.utils.config_manager import get_config_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_financial_data():
    """Example: Retrieve and process financial data"""
    logger.info("=" * 60)
    logger.info("Example 1: Financial Data for Mag 7 Stocks")
    logger.info("=" * 60)
    
    client = get_data_client()
    
    try:
        # Get financial data for Mag 7 stocks
        df = client.get_financial_data()
        logger.info(f"Retrieved {len(df)} records")
        if not df.empty:
            logger.info(f"Columns: {', '.join(df.columns.tolist())}")
            logger.info(f"Sample data:\n{df.head()}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


def example_stock_movements():
    """Example: Retrieve stock movement trends"""
    logger.info("=" * 60)
    logger.info("Example 2: Stock Movement Trends (S&P 500)")
    logger.info("=" * 60)
    
    client = get_data_client()
    
    try:
        # Get movements from last 7 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        df = client.get_stock_movements(
            start_date=start_date,
            end_date=end_date,
            indicators=['SMA_20', 'RSI', 'MACD']
        )
        
        logger.info(f"Retrieved {len(df)} movement records")
        if not df.empty:
            logger.info(f"Tickers: {df['ticker'].unique().tolist()[:5]}...")
            logger.info(f"Sample data:\n{df.head()}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


def example_news_data():
    """Example: Retrieve news with sentiment analysis"""
    logger.info("=" * 60)
    logger.info("Example 3: News Data with Sentiment Analysis")
    logger.info("=" * 60)
    
    client = get_data_client()
    
    try:
        # Get positive sentiment news
        df = client.get_news_data(
            sentiment_filter=(0.5, 1.0),  # Positive sentiment
            tickers=['AAPL', 'MSFT', 'GOOGL']
        )
        
        logger.info(f"Retrieved {len(df)} news articles")
        if not df.empty:
            logger.info(f"Sources: {df['source'].unique().tolist()}")
            logger.info(f"Sample data:\n{df.head()}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


def example_macroeconomic_data():
    """Example: Retrieve macroeconomic indicators"""
    logger.info("=" * 60)
    logger.info("Example 4: Macroeconomic Indicators")
    logger.info("=" * 60)
    
    client = get_data_client()
    
    try:
        # Get macro data for Mag 7
        df = client.get_macroeconomic_data(
            indicators=['interest_rate', 'unemployment_rate', 'inflation_rate']
        )
        
        logger.info(f"Retrieved {len(df)} macro records")
        if not df.empty:
            logger.info(f"Sample data:\n{df.head()}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


def example_policy_data():
    """Example: Retrieve policy and economic announcements"""
    logger.info("=" * 60)
    logger.info("Example 5: Policy & Economic Announcements")
    logger.info("=" * 60)
    
    client = get_data_client()
    
    try:
        # Get recent policy announcements
        df = client.get_policy_data(
            data_types=['policy_announcements', 'fomc_meeting']
        )
        
        logger.info(f"Retrieved {len(df)} policy records")
        if not df.empty:
            logger.info(f"Data types: {df['data_type'].unique().tolist()}")
            logger.info(f"Sample data:\n{df.head()}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


def example_data_export():
    """Example: Export data to different formats"""
    logger.info("=" * 60)
    logger.info("Example 6: Export Data to Different Formats")
    logger.info("=" * 60)
    
    client = get_data_client()
    
    try:
        # Export financial data to CSV
        result = client.export_data(
            'financial_data',
            'examples/mag7_financial.csv',
            format='csv'
        )
        logger.info(f"CSV export: {'Success' if result else 'Failed'}")
        
        # Export to Parquet (more efficient)
        result = client.export_data(
            'financial_data',
            'examples/mag7_financial.parquet',
            format='parquet'
        )
        logger.info(f"Parquet export: {'Success' if result else 'Failed'}")
        
        # Export to JSON
        result = client.export_data(
            'financial_data',
            'examples/mag7_financial.json',
            format='json'
        )
        logger.info(f"JSON export: {'Success' if result else 'Failed'}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")


def example_data_summary():
    """Example: Get data collection summary"""
    logger.info("=" * 60)
    logger.info("Example 7: Data Collection Summary")
    logger.info("=" * 60)
    
    client = get_data_client()
    
    try:
        summary = client.get_data_summary()
        logger.info(f"Summary:\n{summary}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


def example_configuration():
    """Example: Manage pipeline configuration"""
    logger.info("=" * 60)
    logger.info("Example 8: Configuration Management")
    logger.info("=" * 60)
    
    config_manager = get_config_manager()
    
    try:
        # Get current configuration
        config_dict = config_manager.to_dict()
        logger.info(f"Current configuration:\n{config_dict}")
        
        # Set API key (example - don't actually set test keys)
        # config_manager.set_api_key('finnhub', 'test_key_123')
        
        # Save configuration
        config_manager.save_configuration()
        logger.info("Configuration saved")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")


def main():
    """Run all examples"""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " Stock Trend Estimator - Pipeline Usage Examples ".center(58) + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("\n")
    
    # Run examples
    example_financial_data()
    example_stock_movements()
    example_news_data()
    example_macroeconomic_data()
    example_policy_data()
    example_data_export()
    example_data_summary()
    example_configuration()
    
    logger.info("\n")
    logger.info("=" * 60)
    logger.info("All examples completed!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
