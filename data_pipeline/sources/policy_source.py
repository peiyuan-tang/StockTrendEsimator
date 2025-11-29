#!/usr/bin/env python3
"""
Policy Data Source - Collects fiscal and monetary policy information
"""

import logging
from datetime import datetime
from typing import Dict, List, Any
from data_pipeline.sources.financial_source import BaseDataSource

logger = logging.getLogger(__name__)


class PolicyDataSource(BaseDataSource):
    """
    Collects fiscal and monetary policy data for Mag 7 stocks:
    - Policy announcements
    - Fed meeting notes
    - Treasury decisions
    - Economic indicators schedule
    """

    MAG7_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.update_frequency = config.get('update_frequency', 'weekly')
        self.data_types = config.get('data_types', [])
        self.api_keys = config.get('api_keys', {})

    def fetch_fed_announcements(self) -> List[Dict[str, Any]]:
        """Fetch Federal Reserve announcements"""
        announcements = []

        try:
            import requests
            from datetime import datetime, timedelta
            
            # Using a public feed for Fed data
            url = "https://www.federalreserve.gov/feeds/press.rss"
            
            import feedparser
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:10]:  # Get last 10 announcements
                announcement = {
                    'data_type': 'fed_announcement',
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'published': entry.get('published', ''),
                    'link': entry.get('link', ''),
                    'timestamp': datetime.utcnow().isoformat(),
                }
                announcements.append(announcement)
                self.logger.debug(f"Fetched Fed announcement: {entry.get('title', '')[:50]}")

        except ImportError:
            self.logger.warning("feedparser not installed. Install with: pip install feedparser")
        except Exception as e:
            self.logger.warning(f"Error fetching Fed announcements: {str(e)}")

        return announcements

    def fetch_fomc_meeting_schedule(self) -> List[Dict[str, Any]]:
        """Fetch FOMC meeting schedule and minutes"""
        meetings = []

        try:
            import requests
            from bs4 import BeautifulSoup
            
            url = "https://www.federalreserve.gov/monetarypolicy/fomccalendar.htm"
            
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse FOMC meeting schedule
            for tr in soup.find_all('tr')[:12]:  # Get upcoming meetings
                tds = tr.find_all('td')
                if len(tds) >= 2:
                    meeting = {
                        'data_type': 'fomc_meeting',
                        'meeting_dates': tds[0].text.strip(),
                        'statement_date': tds[1].text.strip() if len(tds) > 1 else '',
                        'timestamp': datetime.utcnow().isoformat(),
                    }
                    meetings.append(meeting)
                    self.logger.debug(f"Fetched FOMC meeting: {meeting['meeting_dates']}")

        except ImportError:
            self.logger.warning("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
        except Exception as e:
            self.logger.warning(f"Error fetching FOMC meetings: {str(e)}")

        return meetings

    def fetch_treasury_data(self) -> List[Dict[str, Any]]:
        """Fetch Treasury decisions and auction schedule"""
        treasury_data = []

        try:
            import requests
            
            # Treasury auction schedule
            url = "https://www.treasurydirect.gov/NP/BPDLogin?application=TA"
            
            # Using public treasury data endpoint
            treasury_api = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1"
            
            # Get recent treasury operations
            endpoint = f"{treasury_api}/accounting/od/debt_to_penny"
            
            response = requests.get(endpoint, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data and data['data']:
                for record in data['data'][:5]:
                    treasury_record = {
                        'data_type': 'treasury_data',
                        'date': record.get('record_date'),
                        'total_debt': record.get('total_debt_held_by_public'),
                        'timestamp': datetime.utcnow().isoformat(),
                    }
                    treasury_data.append(treasury_record)
                    self.logger.debug(f"Fetched treasury data for {record.get('record_date')}")

        except Exception as e:
            self.logger.warning(f"Error fetching treasury data: {str(e)}")

        return treasury_data

    def fetch_economic_calendar(self) -> List[Dict[str, Any]]:
        """Fetch economic indicators schedule"""
        calendar_events = []

        try:
            import requests
            
            # Using investing.com's economic calendar API (if available)
            # Alternative: Use Alpha Vantage for economic calendar
            
            if 'alpha_vantage' in self.api_keys:
                api_key = self.api_keys['alpha_vantage']
                url = "https://www.alphavantage.co/query"
                
                params = {
                    'function': 'ECONOMIC_INDICATORS',
                    'apikey': api_key
                }
                
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                
                if 'data' in result:
                    for event in result['data'][:20]:
                        event_record = {
                            'data_type': 'economic_event',
                            'event': event.get('event'),
                            'date': event.get('date'),
                            'importance': event.get('importance'),
                            'forecast': event.get('forecast'),
                            'previous': event.get('previous'),
                            'timestamp': datetime.utcnow().isoformat(),
                        }
                        calendar_events.append(event_record)
                        self.logger.debug(f"Fetched economic event: {event.get('event')}")

        except Exception as e:
            self.logger.warning(f"Error fetching economic calendar: {str(e)}")

        return calendar_events

    def fetch_policy_statements(self) -> List[Dict[str, Any]]:
        """Fetch recent policy statements"""
        statements = []

        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Fetch from Federal Reserve press releases
            url = "https://www.federalreserve.gov/news/default.htm"
            
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract recent statements
            for link in soup.find_all('a', class_='description')[:10]:
                statement = {
                    'data_type': 'policy_statement',
                    'title': link.text.strip(),
                    'url': link.get('href'),
                    'timestamp': datetime.utcnow().isoformat(),
                }
                statements.append(statement)
                self.logger.debug(f"Fetched policy statement: {link.text.strip()[:50]}")

        except Exception as e:
            self.logger.warning(f"Error fetching policy statements: {str(e)}")

        return statements

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch all policy-related data"""
        data_records = []

        for data_type in self.data_types:
            if data_type == 'policy_announcements':
                data_records.extend(self.fetch_fed_announcements())
            elif data_type == 'fed_meeting_notes':
                data_records.extend(self.fetch_fomc_meeting_schedule())
            elif data_type == 'treasury_decisions':
                data_records.extend(self.fetch_treasury_data())
            elif data_type == 'economic_indicators_schedule':
                data_records.extend(self.fetch_economic_calendar())
            elif data_type == 'policy_statements':
                data_records.extend(self.fetch_policy_statements())

        # Add applicable stocks metadata
        for record in data_records:
            record['applicable_stocks'] = self.MAG7_TICKERS

        self.logger.info(f"Fetched {len(data_records)} policy records")
        return data_records
