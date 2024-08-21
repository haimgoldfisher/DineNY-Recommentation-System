from confluent_kafka import Consumer, KafkaError
import requests
import json
import logging
import signal
import sys
import os
import time
from collections import deque

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend URL
backend_url = os.getenv('BACKEND_URL', 'http://localhost:5050/events')

# Define batch size and flush interval (in seconds)
BATCH_SIZE = 1000
FLUSH_INTERVAL = 300  # 5 minutes

def create_kafka_consumer():
    conf = {
        'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        'group.id': 'analytics_group',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False  # Disable auto-commit of offsets
    }
    return Consumer(conf)

def consume_events():
    consumer = create_kafka_consumer()
    consumer.subscribe([os.getenv('KAFKA_TOPIC', 'analytics_topic')])  # Get topic name from environment

    def signal_handler(sig, frame):
        logger.info('Shutting down...')
        consumer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    event_buffer = deque()
    last_flush_time = time.time()

    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition, not an error
                continue
            else:
                logger.error(msg.error())
                break

        try:
            # Decode and parse the message value
            event = json.loads(msg.value().decode('utf-8'))
            event_buffer.append(event)

            # Check if the buffer is full or flush interval has passed
            current_time = time.time()
            if len(event_buffer) >= BATCH_SIZE or (current_time - last_flush_time >= FLUSH_INTERVAL):
                # Send events to backend
                response = requests.post(backend_url, json={'events': list(event_buffer)})
                if response.status_code == 200:
                    logger.info(f"Events successfully sent to backend: {list(event_buffer)}")
                    consumer.commit()
                    event_buffer.clear()
                    last_flush_time = current_time
                else:
                    logger.error(f"Failed to send events to backend. Status code: {response.status_code}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
        except requests.RequestException as e:
            logger.error(f"HTTP request failed: {e}")

    consumer.close()

if __name__ == '__main__':
    consume_events()
