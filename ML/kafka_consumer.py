# kafka_consumer.py - Updated version with threaded consumption

import json
import logging
import threading
import time
from queue import Queue, Empty
from kafka import KafkaConsumer
import pandas as pd
import os

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TrafficDataConsumer:
    def __init__(self, bootstrap_servers='localhost:9092', topic='traffic-data'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_frames = {}
        self.max_records = 50

        # Shared data structure updated by the consumer thread
        self.latest_data = {}

        # Flag to control the consumer thread
        self.running = False
        self.consumer_thread = None

        logger.info(f"Kafka consumer initialized for topic: {topic}")

        # Start the consumer thread
        self.start_consumer_thread()

    def start_consumer_thread(self):
        """Start the consumer thread that runs in the background"""
        if not self.running:
            self.running = True
            self.consumer_thread = threading.Thread(target=self._consume_messages)
            self.consumer_thread.daemon = True
            self.consumer_thread.start()
            logger.info("Kafka consumer thread started")

    def _consume_messages(self):
        """Background thread that continuously consumes messages from Kafka"""
        try:
            consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id='tariq-ml-consumer',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )

            logger.info(f"Consumer thread connected to Kafka topic: {self.topic}")

            for message in consumer:
                if not self.running:
                    break

                try:
                    data = message.value
                    road_id = data['roadId']

                    # Create the new record
                    record = {
                        'timestamp': data['timestamp'],
                        'vehicleCount': data['vehicleCount'],
                        'congestionLevel': data['congestionLevel'],
                        'averageSpeed': data['averageSpeed'],
                        'hour': data['hour'],
                        'dayOfWeek': data['dayOfWeek'],
                        'isWeekend': data['isWeekend']
                    }

                    # Update latest data (shared resource)
                    self.latest_data[road_id] = record

                    # Add to in-memory DataFrame
                    if road_id not in self.data_frames:
                        self.data_frames[road_id] = pd.DataFrame([record])
                    else:
                        self.data_frames[road_id] = pd.concat(
                            [self.data_frames[road_id], pd.DataFrame([record])],
                            ignore_index=True
                        )

                    # Handle CSV storage
                    csv_path = os.path.join(self.data_dir, f"{road_id}_traffic_data.csv")

                    # Check how many records are already saved
                    if os.path.exists(csv_path):
                        existing_df = pd.read_csv(csv_path)
                        if len(existing_df) >= self.max_records:
                            logger.debug(f"Already {self.max_records} records for road {road_id}, skipping save...")
                            continue
                    else:
                        existing_df = pd.DataFrame()

                    # Combine and truncate if necessary
                    updated_df = pd.concat([existing_df, pd.DataFrame([record])], ignore_index=True)

                    if len(updated_df) <= self.max_records:
                        updated_df.to_csv(csv_path, index=False)
                        logger.info(f"Saved {len(updated_df)} records for road {road_id}")
                    else:
                        logger.debug(f"Would exceed {self.max_records} records for road {road_id}, skipping write")

                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except Exception as e:
            logger.error(f"Error in consumer thread: {e}")
            self.running = False

    def consume_and_store(self):
        """Non-blocking method to get the latest data"""
        # Simply return the current state of the data frames
        # This method no longer waits for Kafka messages
        return self.data_frames

    def get_latest_data(self):
        """Get the latest data for all roads"""
        return self.latest_data

    def get_data_for_road(self, road_id):
        """Get the stored data for a specific road"""
        csv_path = os.path.join(self.data_dir, f"{road_id}_traffic_data.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return None

    def close(self):
        """Close the Kafka consumer thread"""
        if self.running:
            self.running = False
            if self.consumer_thread:
                self.consumer_thread.join(timeout=5.0)
            logger.info("Kafka consumer thread stopped")