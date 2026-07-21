"""
RQ worker for processing APVBT inference jobs.

This module provides a worker process that can be run to process jobs
from Redis Queue.
"""

import os
import sys
import logging
from redis import Redis
from rq import Worker, Queue
from rq.connections import RedisConnection as Connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def listen(queue_names=['default']):
    """
    Start listening for jobs on the specified queues.
    
    Args:
        queue_names: List of queue names to listen on
    """
    # Redis connection
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    redis_db = int(os.environ.get('REDIS_DB', 0))
    redis_password = os.environ.get('REDIS_PASSWORD')
    
    logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
    
    if redis_password:
        redis_conn = Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True
        )
    else:
        redis_conn = Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
    
    # Test connection
    try:
        redis_conn.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        sys.exit(1)
    
    # Create queues
    queues = [Queue(name, connection=redis_conn) for name in queue_names]
    
    logger.info(f"Starting worker listening on queues: {queue_names}")
    
    # Start worker
    with Connection(redis_conn):
        worker = Worker(queues)
        worker.work()


def main():
    """Main entry point for worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="APVBT RQ Worker")
    parser.add_argument(
        '--queues',
        type=str,
        nargs='+',
        default=['default'],
        help='Queue names to listen on (default: default)'
    )
    parser.add_argument(
        '--burst',
        action='store_true',
        help='Run in burst mode (exit when no jobs left)'
    )
    
    args = parser.parse_args()
    
    # Set burst mode environment variable (RQ respects JOB_WORKER_BURST)
    if args.burst:
        os.environ['JOB_WORKER_BURST'] = 'true'
    
    listen(queue_names=args.queues)


if __name__ == '__main__':
    main()