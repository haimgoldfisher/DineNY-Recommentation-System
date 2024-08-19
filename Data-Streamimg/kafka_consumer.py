from confluent_kafka import Consumer, KafkaError
import requests
import json
import logging
import signal
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend URL
backend_url = os.getenv('BACKEND_URL', 'http://localhost:5050/events')


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

            # Send the event to the backend
            response = requests.post(backend_url, json=event)
            if response.status_code == 200:
                logger.info(f"Event successfully sent to backend: {event}")
                # Manually commit the offset after processing the message
                consumer.commit()
            else:
                logger.error(f"Failed to send event to backend. Status code: {response.status_code}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
        except requests.RequestException as e:
            logger.error(f"HTTP request failed: {e}")

    consumer.close()


if __name__ == '__main__':
    consume_events()
