import os
import logging
import threading
import time
from kafka_consumer import TrafficDataConsumer
from ml_trainer import TrafficPredictionModel
from api_service import start_api_server, initialize_services, data_collection_thread

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Print welcome message
    print("=" * 70)
    print("Tariq - AI-Powered Traffic Management System")
    print("ML Prediction Service")
    print("=" * 70)

    # Initialize services
    initialize_services()

    # Start data collection thread
    logger.info("Starting data collection thread")
    collection_thread = threading.Thread(target=data_collection_thread)
    collection_thread.daemon = True
    collection_thread.start()

    # Start API server
    logger.info("Starting API server on port 5000")
    start_api_server()