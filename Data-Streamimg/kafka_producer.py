from kafka import KafkaProducer
import json

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Send a test message
producer.send('test_topic', {'key': 'value'})
producer.flush()

print("Message sent successfully!")
