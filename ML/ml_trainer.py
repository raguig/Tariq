import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
import random  # For synthetic data generation

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Changed to DEBUG for more detailed logs
)
logger = logging.getLogger(__name__)


class TrafficPredictionModel:
    def __init__(self):
        self.models = {}  # Road ID -> model
        self.scalers = {}  # Road ID -> scaler
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info("Traffic prediction model initialized")

        # Define congestion level thresholds based on vehicle count and speed
        # These will help generate better synthetic data if needed
        self.congestion_thresholds = [
            # (max_vehicles, min_speed, congestion_level)
            (10, 80, 0),  # Free flow: few vehicles, high speed
            (25, 60, 1),  # Light: more vehicles, good speed
            (50, 45, 2),  # Moderate: medium vehicles, medium speed
            (100, 30, 3),  # Heavy: many vehicles, slower speed
            (200, 15, 4),  # Very heavy: lots of vehicles, very slow
            (float('inf'), 0, 5)  # Gridlock: any number of vehicles, very slow or stopped
        ]

    def prepare_features(self, df):
        """Prepare features for model training with feature engineering"""
        # Basic features
        features = ['hour', 'dayOfWeek', 'isWeekend', 'vehicleCount', 'averageSpeed']

        # Add engineered features if they don't exist

        target = 'congestionLevel'

        # Drop rows with NaN values
        df = df.dropna(subset=features + [target])

        X = df[features]
        y = df[target].astype(int)  # Ensure target is integer type

        return X, y


    def train_model(self, road_id, df):
        """Train a prediction model for a specific road"""
        logger.info(f"Training model for road {road_id} with {len(df)} records")

        # Check if we have enough real data

        X, y = self.prepare_features(df)

        # Print target distribution
        unique_values, counts = np.unique(y, return_counts=True)
        logger.info(f"Target distribution: {dict(zip(unique_values, counts))}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Try different models - Gradient Boosting often performs better for this type of problem
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # optional, helps with imbalanced datasets
        )

        # Simple parameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )

        logger.info("Starting model training with GridSearchCV...")
        grid_search.fit(X_train_scaled, y_train)

        # Get best model
        model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")

        # Evaluate model
        y_pred = model.predict(X_test_scaled)

        # Verify prediction distribution
        pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
        logger.info(f"Prediction distribution on test set: {dict(zip(pred_unique, pred_counts))}")

        # Detailed classification report
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        logger.info(f"Model accuracy for road {road_id}: {accuracy:.4f}")
        logger.info(f"Classification report:\n{class_report}")

        # Feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        logger.info(f"Feature importance: {feature_importance}")

        # Save model and scaler
        self.models[road_id] = model
        self.scalers[road_id] = scaler

        model_path = os.path.join(self.model_dir, f"{road_id}_model.joblib")
        scaler_path = os.path.join(self.model_dir, f"{road_id}_scaler.joblib")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        logger.info(f"Model for road {road_id} saved to {model_path}")
        return True

    def predict_congestion(self, road_id, hour, day_of_week, is_weekend, vehicle_count, average_speed):
        """Predict congestion level for a road based on current conditions"""
        if road_id not in self.models:
            self.load_model(road_id)

        if road_id not in self.models:
            logger.warning(f"No model available for road {road_id}")
            return None


        # Prepare input data with all features
        X = np.array([[
            hour,
            day_of_week,
            is_weekend,
            vehicle_count,
            average_speed,

        ]])

        # Log input data
        logger.debug(f"Input for road {road_id}: {X}")

        try:
            # Scale input - make sure scaler expects all features
            X_scaled = self.scalers[road_id].transform(X)

            # Make prediction
            predicted_class = self.models[road_id].predict(X_scaled)[0]

            # Get class probabilities
            class_probs = self.models[road_id].predict_proba(X_scaled)[0]
            prob_dict = {i: prob for i, prob in enumerate(class_probs)}
            logger.debug(f"Class probabilities: {prob_dict}")

            # Log the raw prediction
            logger.info(
                f"Raw predicted congestion level for road {road_id}: {predicted_class} (type: {type(predicted_class)})")

            # Ensure prediction is a valid integer between 0-5
            congestion_level = max(0, min(5, int(predicted_class)))

            logger.info(f"Final predicted congestion level for road {road_id}: {congestion_level}")

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")

            # Fallback: Use rule-based prediction instead of model

            logger.info(f"Using rule-based fallback prediction: {congestion_level}")

        return {
            'road_id': road_id,
            'predicted_congestion': congestion_level,
            'current_vehicle_count': vehicle_count,
            'current_average_speed': average_speed,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'raw_prediction': int(congestion_level)  # Include for debugging
        }


    def load_model(self, road_id):
        """Load a saved model for a specific road"""
        model_path = os.path.join(self.model_dir, f"{road_id}_model.joblib")
        scaler_path = os.path.join(self.model_dir, f"{road_id}_scaler.joblib")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.models[road_id] = joblib.load(model_path)
                self.scalers[road_id] = joblib.load(scaler_path)
                logger.info(f"Loaded model for road {road_id}")
                return True
            except Exception as e:
                logger.error(f"Error loading model for road {road_id}: {str(e)}")
                return False
        else:
            logger.warning(f"No saved model found for road {road_id}")
            return False

    def load_all_models(self):
        """Load all saved models"""
        loaded_models = 0
        for filename in os.listdir(self.model_dir):
            if filename.endswith('_model.joblib'):
                road_id = filename.split('_')[0]
                if self.load_model(road_id):
                    loaded_models += 1

        logger.info(f"Loaded {loaded_models} road models")
        return loaded_models