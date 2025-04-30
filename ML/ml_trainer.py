import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from urllib3.util import timeout as time
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
import random  

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  
)
logger = logging.getLogger(__name__)


class TrafficPredictionModel:
    def __init__(self):
        self.models = {}  
        self.scalers = {} 
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info("Traffic prediction model initialized")

    
        self.congestion_thresholds = [
          
            (10, 80, 0),  
            (25, 60, 1), 
            (50, 45, 2),  
            (100, 30, 3),  
            (200, 15, 4),  
            (float('inf'), 0, 5)  #
        ]

    def prepare_features(self, df):
        """Prepare features for model training with feature engineering"""
        features = ['hour', 'dayOfWeek', 'isWeekend', 'vehicleCount', 'averageSpeed']


        target = 'congestionLevel'

        df = df.dropna(subset=features + [target])

        X = df[features]
        y = df[target].astype(int) 

        return X, y

    def train_model(self, road_id, df):
        """Train a prediction model for a specific road using XGBoost"""
        logger.info(f"Training XGBoost model for road {road_id} with {len(df)} records")

        X, y = self.prepare_features(df)

        unique_values, counts = np.unique(y, return_counts=True)
        logger.info(f"Target distribution: {dict(zip(unique_values, counts))}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(
            objective="multi:softprob", 
            num_class=6, 
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,  
            eval_metric='mlogloss'  
        )

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )

        logger.info("Starting XGBoost model training with GridSearchCV...")
        grid_search.fit(X_train_scaled, y_train)

        model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")

        y_pred = model.predict(X_test_scaled)

        pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
        logger.info(f"Prediction distribution on test set: {dict(zip(pred_unique, pred_counts))}")

        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        logger.info(f"Model accuracy for road {road_id}: {accuracy:.4f}")
        logger.info(f"Classification report:\n{class_report}")

        feature_importance = dict(zip(X.columns, model.feature_importances_))
        logger.info(f"Feature importance: {feature_importance}")

        self.models[road_id] = model
        self.scalers[road_id] = scaler

        model_path = os.path.join(self.model_dir, f"{road_id}_xgboost_model.joblib")
        scaler_path = os.path.join(self.model_dir, f"{road_id}_scaler.joblib")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        logger.info(f"XGBoost model for road {road_id} saved to {model_path}")
        return True

    def predict_congestion(self, road_id, hour, day_of_week, is_weekend, vehicle_count, average_speed,
                           forecast_seconds=10):
        """Predict congestion level for a road based on current or future conditions"""
        if road_id not in self.models:
            self.load_model(road_id)

        if road_id not in self.models:
            logger.warning(f"No model available for road {road_id}")
            return None

        current_obs = {
            'hour': hour,
            'dayOfWeek': day_of_week,
            'isWeekend': is_weekend,
            'vehicleCount': vehicle_count,
            'averageSpeed': average_speed,

        }
        self.add_observation(road_id, current_obs)

        if forecast_seconds > 0:
            if road_id in self.recent_observations and len(self.recent_observations[road_id]) >= 2:
                observations = self.recent_observations[road_id]
                if len(observations) >= 2:
                    recent_obs = observations[-2:]
                    time_diff = recent_obs[1]['timestamp'] - recent_obs[0]['timestamp']

                    if time_diff > 0:  
                        
                        vehicle_change_rate = (recent_obs[1]['vehicleCount'] - recent_obs[0][
                            'vehicleCount']) / time_diff
                        speed_change_rate = (recent_obs[1]['averageSpeed'] - recent_obs[0]['averageSpeed']) / time_diff

                        vehicle_count = max(0, vehicle_count + (vehicle_change_rate * forecast_seconds))
                        average_speed = max(0, min(120, average_speed + (speed_change_rate * forecast_seconds)))

                        logger.info(f"Forecasted values for {forecast_seconds}s ahead: "
                                    f"vehicles={vehicle_count:.1f}, speed={average_speed:.1f}")

        X = np.array([[
            hour,
            day_of_week,
            is_weekend,
            vehicle_count,
            average_speed,
        ]])

        logger.debug(f"Input for road {road_id}: {X}")

        try:
            X_scaled = self.scalers[road_id].transform(X)

            predicted_class = self.models[road_id].predict(X_scaled)[0]

            class_probs = self.models[road_id].predict_proba(X_scaled)[0]
            prob_dict = {i: prob for i, prob in enumerate(class_probs)}
            logger.debug(f"Class probabilities: {prob_dict}")

            logger.info(
                f"Raw predicted congestion level for road {road_id}: {predicted_class} (type: {type(predicted_class)})")

            congestion_level = max(0, min(5, int(predicted_class)))

            logger.info(f"Final predicted congestion level for road {road_id}: {congestion_level}")

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")

            congestion_level = self.rule_based_prediction(vehicle_count, average_speed)
            logger.info(f"Using rule-based fallback prediction: {congestion_level}")

        return {
            'road_id': road_id,
            'predicted_congestion': congestion_level,
            'current_vehicle_count': vehicle_count,
            'current_average_speed': average_speed,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'forecast_seconds': forecast_seconds,
            'raw_prediction': int(congestion_level)  
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
