import os
import time
import logging
import threading
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from kafka_consumer import TrafficDataConsumer
from ml_trainer import TrafficPredictionModel
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global objects
consumer = None
model = None
latest_predictions = {}
training_status = {}


def initialize_services():
    global consumer, model

    # Initialize Kafka consumer
    consumer = TrafficDataConsumer()

    # Initialize prediction model
    model = TrafficPredictionModel()
    model.load_all_models()


    logger.info("All services initialized")


def data_collection_thread():
    """Thread to continuously process data and update models"""
    global consumer, model, latest_predictions, training_status

    logger.info("Starting data collection thread")

    while True:
        try:
            # Get the latest data from the consumer
            # (This is now a non-blocking operation since the consumer has its own thread)
            road_data = consumer.data_frames

            # Get the latest data from the consumer's shared structure
            print("here1")
            latest_data = consumer.get_latest_data()
            print("here2")
            # Process each road's data
            for road_id, latest_row in latest_data.items():
                # Check if we should train model based on time elapsed
                should_train = (road_id not in training_status or
                                (datetime.now() - training_status.get(road_id, datetime.min)).total_seconds() > 600)

                if should_train:
                    # Load data from CSV instead of using in-memory dataframe
                    csv_path = os.path.join(consumer.data_dir, f"{road_id}_traffic_data.csv")

                    if os.path.exists(csv_path):
                        logger.info(f"Training model for road {road_id} using CSV data")
                        training_status[road_id] = datetime.now()

                        # Load data from CSV
                        csv_df = pd.read_csv(csv_path)

                        if len(csv_df) >= 1:  # Make sure we have at least some data
                            logger.info(f"Starting training for road {road_id} with {len(csv_df)} records")
                            training_start_time = time.time()

                            if model.train_model(road_id, csv_df):
                                training_duration = time.time() - training_start_time
                                logger.info(
                                    f"Model for road {road_id} trained successfully in {training_duration:.2f} seconds")
                            else:
                                logger.warning(f"Failed to train model for road {road_id}")
                        else:
                            logger.warning(f"Not enough data in CSV for road {road_id}")
                    else:
                        logger.warning(f"No CSV file found for road {road_id}")

                # Generate prediction using the latest data
                prediction = model.predict_congestion(
                    road_id,
                    latest_row['hour'],
                    latest_row['dayOfWeek'],
                    latest_row['isWeekend'],
                    latest_row['vehicleCount'],
                    latest_row['averageSpeed']
                )

                if prediction:
                    latest_predictions[road_id] = prediction

            # Sleep for a short time to avoid consuming too much CPU
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error in data collection thread: {e}")
            time.sleep(5)  # Wait a bit before retrying


@app.route('/api/traffic/current', methods=['GET'])
def get_current_traffic():
    """API endpoint to get current traffic data for all roads"""
    global consumer
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'roads': consumer.get_latest_data()
    })


@app.route('/api/traffic/current/<road_id>', methods=['GET'])
def get_road_traffic(road_id):
    """API endpoint to get current traffic data for a specific road"""
    global consumer
    latest_data = consumer.get_latest_data()

    if road_id in latest_data:
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'road': latest_data[road_id]
        })
    else:
        return jsonify({'error': f'No data available for road {road_id}'}), 404


@app.route('/api/traffic/predictions', methods=['GET'])
def get_predictions():
    """API endpoint to get traffic predictions for all roads"""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'predictions': latest_predictions
    })


@app.route('/api/traffic/predictions/<road_id>', methods=['GET'])
def get_road_prediction(road_id):
    """API endpoint to get traffic prediction for a specific road"""
    if road_id in latest_predictions:
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'prediction': latest_predictions[road_id]
        })
    else:
        return jsonify({'error': f'No prediction available for road {road_id}'}), 404


@app.route('/api/traffic/history/<road_id>', methods=['GET'])
def get_road_history(road_id):
    """API endpoint to get historical traffic data for a specific road"""
    global consumer

    # Get optional query parameters
    limit = request.args.get('limit', default=100, type=int)

    # Get data from CSV file
    csv_path = os.path.join(consumer.data_dir, f"{road_id}_traffic_data.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Convert to list of dictionaries (limit to the most recent records)
        history = df.tail(limit).to_dict('records')
        return jsonify({
            'road_id': road_id,
            'history': history
        })
    else:
        return jsonify({'error': f'No historical data available for road {road_id}'}), 404


@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    """API endpoint to get the training status of all models"""
    global model, training_status, consumer

    status_info = {}

    # Get list of all roads from training status and model objects
    all_roads = set(list(training_status.keys()) + list(model.models.keys()) + list(consumer.get_latest_data().keys()))

    for road_id in all_roads:
        # Check if model exists for this road
        model_exists = road_id in model.models

        # Get last training time if available
        last_trained = None
        if road_id in training_status:
            last_trained = training_status[road_id].isoformat()

        # Check if model file exists
        model_path = os.path.join(model.model_dir, f"{road_id}_model.joblib")
        model_file_exists = os.path.exists(model_path)

        # Check data file
        csv_path = os.path.join(consumer.data_dir, f"{road_id}_traffic_data.csv")
        data_exists = os.path.exists(csv_path)
        data_count = 0

        if data_exists:
            try:
                df = pd.read_csv(csv_path)
                data_count = len(df)
            except Exception as e:
                logger.error(f"Error reading CSV for road {road_id}: {e}")

        # Compile status for this road
        status_info[road_id] = {
            "model_loaded": model_exists,
            "model_file_exists": model_file_exists,
            "last_trained": last_trained,
            "data_file_exists": data_exists,
            "record_count": data_count
        }

    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "road_models": status_info
    })


@app.route('/api/traffic/retrain/<road_id>', methods=['POST'])
def force_retrain_model(road_id):
    """API endpoint to force retraining of a model for a specific road"""
    global consumer, model, training_status

    csv_path = os.path.join(consumer.data_dir, f"{road_id}_traffic_data.csv")

    if os.path.exists(csv_path):
        csv_df = pd.read_csv(csv_path)

        if len(csv_df) >= 1:
            training_status[road_id] = datetime.now()

            if model.train_model(road_id, csv_df):
                return jsonify({
                    'success': True,
                    'message': f"Model for road {road_id} retrained successfully with {len(csv_df)} records"
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f"Failed to train model for road {road_id}"
                }), 500
        else:
            return jsonify({
                'success': False,
                'message': f"Not enough data for road {road_id}"
            }), 400
    else:
        return jsonify({
            'success': False,
            'message': f"No data available for road {road_id}"
        }), 404


def start_api_server():
    """Start the Flask API server"""
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    # Initialize services
    initialize_services()

    # Start data collection thread
    collection_thread = threading.Thread(target=data_collection_thread)
    collection_thread.daemon = True
    collection_thread.start()

    # Start API server
    logger.info("Starting API server on port 5000")
    start_api_server()