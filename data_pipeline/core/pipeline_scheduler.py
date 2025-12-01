#!/usr/bin/env python3
"""
Pipeline Scheduler - Manages data collection schedules
Uses APScheduler for robust task scheduling
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """Manages collection schedules for different data types"""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.jobs = {}

    def start(self):
        """Start the scheduler"""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Pipeline scheduler started")

    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Pipeline scheduler stopped")

    def add_interval_job(
        self,
        func: Callable,
        job_id: str,
        seconds: int = 3600,
        replace_existing: bool = True
    ) -> str:
        """
        Add a job to run at intervals
        
        Args:
            func: Callable to execute
            job_id: Unique job identifier
            seconds: Interval in seconds
            replace_existing: Replace if job exists
            
        Returns:
            Job ID
        """
        try:
            self.scheduler.add_job(
                func,
                trigger=IntervalTrigger(seconds=seconds),
                id=job_id,
                name=job_id,
                replace_existing=replace_existing,
                max_instances=1
            )
            self.jobs[job_id] = {
                'type': 'interval',
                'seconds': seconds,
                'added_at': datetime.utcnow().isoformat()
            }
            logger.info(f"Added interval job: {job_id} (every {seconds}s)")
            return job_id
        except Exception as e:
            logger.error(f"Error adding interval job: {str(e)}")
            raise

    def add_cron_job(
        self,
        func: Callable,
        job_id: str,
        hour: Optional[int] = None,
        minute: Optional[int] = None,
        day_of_week: Optional[int] = None,
        replace_existing: bool = True
    ) -> str:
        """
        Add a job to run at specific times (cron-style)
        
        Args:
            func: Callable to execute
            job_id: Unique job identifier
            hour: Hour (0-23)
            minute: Minute (0-59)
            day_of_week: Day of week (0-6, Monday-Sunday)
            replace_existing: Replace if job exists
            
        Returns:
            Job ID
        """
        try:
            self.scheduler.add_job(
                func,
                trigger=CronTrigger(hour=hour, minute=minute, day_of_week=day_of_week),
                id=job_id,
                name=job_id,
                replace_existing=replace_existing,
                max_instances=1
            )
            self.jobs[job_id] = {
                'type': 'cron',
                'hour': hour,
                'minute': minute,
                'day_of_week': day_of_week,
                'added_at': datetime.utcnow().isoformat()
            }
            logger.info(f"Added cron job: {job_id}")
            return job_id
        except Exception as e:
            logger.error(f"Error adding cron job: {str(e)}")
            raise

    def remove_job(self, job_id: str) -> bool:
        """Remove a job"""
        try:
            self.scheduler.remove_job(job_id)
            if job_id in self.jobs:
                del self.jobs[job_id]
            logger.info(f"Removed job: {job_id}")
            return True
        except Exception as e:
            logger.warning(f"Error removing job {job_id}: {str(e)}")
            return False

    def get_jobs(self) -> Dict[str, Any]:
        """Get all scheduled jobs"""
        return self.jobs.copy()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of specific job"""
        if job_id not in self.jobs:
            return {'error': f'Job {job_id} not found'}
        
        job = self.scheduler.get_job(job_id)
        status = {
            'job_id': job_id,
            'config': self.jobs[job_id],
        }
        
        if job:
            status.update({
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger),
                'pending': job.pending
            })
        
        return status


class CollectionScheduler(PipelineScheduler):
    """Specialized scheduler for data collection tasks"""

    def __init__(self):
        super().__init__()
        self.collection_configs = {}

    def schedule_financial_collection(self, callback: Callable, day_of_week: int = 0, hour: int = 9, minute: int = 0):
        """Schedule financial data collection (weekly)
        
        Args:
            callback: Function to execute
            day_of_week: Day of week (0=Monday, 6=Sunday, default=0/Monday)
            hour: Hour of day (0-23, default=9 AM)
            minute: Minute of hour (0-59, default=0)
        """
        job_id = 'financial_data_collection'
        self.add_cron_job(
            callback,
            job_id=job_id,
            hour=hour,
            minute=minute,
            day_of_week=day_of_week
        )
        self.collection_configs[job_id] = {
            'data_type': 'financial_data',
            'stocks': 'Mag 7',
            'schedule': 'weekly',
            'day_of_week': day_of_week,
            'time': f'{hour:02d}:{minute:02d}'
        }

    def schedule_movement_collection(self, callback: Callable, day_of_week: int = 1, hour: int = 10, minute: int = 0):
        """Schedule stock movement collection (weekly)
        
        Args:
            callback: Function to execute
            day_of_week: Day of week (0=Monday, 6=Sunday, default=1/Tuesday)
            hour: Hour of day (0-23, default=10 AM)
            minute: Minute of hour (0-59, default=0)
        """
        job_id = 'movement_collection'
        self.add_cron_job(
            callback,
            job_id=job_id,
            hour=hour,
            minute=minute,
            day_of_week=day_of_week
        )
        self.collection_configs[job_id] = {
            'data_type': 'stock_movements',
            'stocks': 'S&P 500',
            'schedule': 'weekly',
            'day_of_week': day_of_week,
            'time': f'{hour:02d}:{minute:02d}'
        }

    def schedule_news_collection(self, callback: Callable, day_of_week: int = 2, hour: int = 11, minute: int = 0):
        """Schedule news data collection (weekly)
        
        Args:
            callback: Function to execute
            day_of_week: Day of week (0=Monday, 6=Sunday, default=2/Wednesday)
            hour: Hour of day (0-23, default=11 AM)
            minute: Minute of hour (0-59, default=0)
        """
        job_id = 'news_collection'
        self.add_cron_job(
            callback,
            job_id=job_id,
            hour=hour,
            minute=minute,
            day_of_week=day_of_week
        )
        self.collection_configs[job_id] = {
            'data_type': 'news',
            'stocks': 'S&P 500',
            'schedule': 'weekly',
            'day_of_week': day_of_week,
            'time': f'{hour:02d}:{minute:02d}'
        }

    def schedule_macro_collection(self, callback: Callable, day_of_week: int = 3, hour: int = 14, minute: int = 0):
        """Schedule macroeconomic data collection (weekly at specific time)
        
        Args:
            callback: Function to execute
            day_of_week: Day of week (0=Monday, 6=Sunday, default=3/Thursday)
            hour: Hour of day (0-23, default=14/2 PM)
            minute: Minute of hour (0-59, default=0)
        """
        job_id = 'macro_collection'
        self.add_cron_job(
            callback,
            job_id=job_id,
            hour=hour,
            minute=minute,
            day_of_week=day_of_week
        )
        self.collection_configs[job_id] = {
            'data_type': 'macroeconomic',
            'stocks': 'Mag 7',
            'schedule': 'weekly',
            'day_of_week': day_of_week,
            'time': f'{hour:02d}:{minute:02d}'
        }

    def schedule_policy_collection(self, callback: Callable, day_of_week: int = 4, hour: int = 15, minute: int = 30):
        """Schedule policy data collection (weekly at specific time)
        
        Args:
            callback: Function to execute
            day_of_week: Day of week (0=Monday, 6=Sunday, default=4/Friday)
            hour: Hour of day (0-23, default=15/3 PM)
            minute: Minute of hour (0-59, default=30)
        """
        job_id = 'policy_collection'
        self.add_cron_job(
            callback,
            job_id=job_id,
            hour=hour,
            minute=minute,
            day_of_week=day_of_week
        )
        self.collection_configs[job_id] = {
            'data_type': 'policy',
            'stocks': 'Mag 7',
            'schedule': 'weekly',
            'day_of_week': day_of_week,
            'time': f'{hour:02d}:{minute:02d}'
        }

    def get_collection_status(self) -> Dict[str, Any]:
        """Get status of all collections"""
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'scheduler_running': self.scheduler.running,
            'collections': {}
        }
        
        for job_id, config in self.collection_configs.items():
            status['collections'][job_id] = {
                'config': config,
                'job_status': self.get_job_status(job_id)
            }
        
        return status


# Global scheduler instance
_scheduler = None


def get_scheduler() -> CollectionScheduler:
    """Get global collection scheduler"""
    global _scheduler
    if _scheduler is None:
        _scheduler = CollectionScheduler()
    return _scheduler
