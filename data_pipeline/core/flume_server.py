#!/usr/bin/env python3
"""
Flume Python Server - Offline data collection pipeline
Main orchestrator for data pipeline
"""

import logging
import os
import yaml
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
from flume.agent import Agent
from flume.configuration import FlumeConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """Main data collection pipeline orchestrator"""

    def __init__(self, config_path: str):
        """
        Initialize the data collector
        
        Args:
            config_path: Path to Flume configuration YAML file
        """
        self.config_path = config_path
        self.agents = {}
        self.running = False
        self._load_configuration()

    def _load_configuration(self):
        """Load Flume configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {str(e)}")
            raise

    def initialize_agents(self):
        """Initialize all agents from configuration"""
        agents_config = self.config.get('agents', {})

        for agent_name, agent_config in agents_config.items():
            try:
                agent = Agent(
                    name=agent_name,
                    sources=agent_config.get('sources', []),
                    channels=agent_config.get('channels', []),
                    sinks=agent_config.get('sinks', [])
                )
                self.agents[agent_name] = agent
                logger.info(f"Initialized agent: {agent_name}")
            except Exception as e:
                logger.error(f"Error initializing agent {agent_name}: {str(e)}")

    def _create_directories(self):
        """Create necessary directories for data storage"""
        base_paths = [
            '/data/raw/financial_data',
            '/data/raw/stock_movements',
            '/data/raw/news',
            '/data/context/macroeconomic',
            '/data/context/policy',
            '/var/data/checkpoint',
            '/var/data/queue',
        ]

        for path in base_paths:
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured directory exists: {path}")
            except Exception as e:
                logger.warning(f"Could not create directory {path}: {str(e)}")

    def start(self):
        """Start the data collection pipeline"""
        logger.info("Starting Stock Data Collection Pipeline...")
        
        try:
            self._create_directories()
            self.initialize_agents()
            
            self.running = True
            
            for agent_name, agent in self.agents.items():
                logger.info(f"Starting agent: {agent_name}")
                # Start agent processing
                # agent.start()
            
            logger.info("Data collection pipeline started successfully")
            
        except Exception as e:
            logger.error(f"Error starting pipeline: {str(e)}")
            self.running = False
            raise

    def stop(self):
        """Stop the data collection pipeline"""
        logger.info("Stopping Stock Data Collection Pipeline...")
        
        try:
            for agent_name, agent in self.agents.items():
                logger.info(f"Stopping agent: {agent_name}")
                # agent.stop()
            
            self.running = False
            logger.info("Data collection pipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {str(e)}")

    def run_pipeline(self):
        """Run the pipeline (blocking call)"""
        try:
            self.start()
            # Keep pipeline running
            while self.running:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            self.stop()
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            self.stop()

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            'running': self.running,
            'agents': list(self.agents.keys()),
            'timestamp': datetime.utcnow().isoformat(),
        }

    def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Get specific agent status"""
        if agent_name not in self.agents:
            return {'error': f'Agent {agent_name} not found'}
        
        agent = self.agents[agent_name]
        return {
            'name': agent_name,
            'sources': agent.sources,
            'channels': agent.channels,
            'sinks': agent.sinks,
            'timestamp': datetime.utcnow().isoformat(),
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Data Collection Pipeline')
    parser.add_argument(
        '--config',
        default='data_pipeline/config/flume_config.yaml',
        help='Path to Flume configuration file'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize and run pipeline
    try:
        collector = StockDataCollector(config_path=args.config)
        collector.run_pipeline()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()
