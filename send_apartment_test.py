import pika
import json

credentials = pika.PlainCredentials("rabbit_user", "ChangeMeRabbit!")
params = pika.ConnectionParameters(
    host="localhost",
    port=5672,
    credentials=credentials
)

conn = pika.BlockingConnection(params)
ch = conn.channel()

ch.queue_declare(queue="poi_results", durable=True)

msg = {"apartment_id": 64}

ch.basic_publish(
    exchange="",
    routing_key="poi_results",
    body=json.dumps(msg),
    properties=pika.BasicProperties(delivery_mode=2)
)

print("✓ Wysłano:", msg)
conn.close()
