import os
import pandas as pd
import joblib
import pickle
import re
from flask import Flask, request, jsonify
from flask_cors import CORS 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google_play_scraper import reviews, Sort
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Constants
APP_IDS = {
    "traveloka": "com.traveloka.android",
    "tiket": "com.tiket.gits",
    "agoda": "com.agoda.mobile.consumer"
}

VALID_SOURCES = ['agoda', 'traveloka', 'tiket', 'user']
REQUIRED_COLUMNS = ['review', 'sentiment', 'kategori', 'tanggal']
MODEL_FOLDER = "trying"

class ModelManager:
    """Handles loading and managing ML models"""
    
    def __init__(self, model_folder: str):
        self.model_folder = model_folder
        self._models = None
    
    @property
    def models(self) -> Dict[str, Any]:
        """Lazy load models"""
        if self._models is None:
            self._models = self._load_models()
        return self._models
    
    def _load_models(self) -> Dict[str, Any]:
        """Load all ML models and components"""
        base_path = os.path.join(os.getcwd(), self.model_folder)
        
        try:
            # Load sentiment model components
            sentiment_model = load_model(os.path.join(base_path, 'bilstm_sentiment_model.h5'))
            
            with open(os.path.join(base_path, 'tokenizer.pickle'), 'rb') as f:
                tokenizer = pickle.load(f)
            
            with open(os.path.join(base_path, 'model_seq.pickle'), 'rb') as f:
                config = pickle.load(f)
            
            with open(os.path.join(base_path, 'label_encoder_sentimen.pkl'), 'rb') as f:
                label_encoder_sentimen = pickle.load(f)
            
            # Load category model components
            kategori_model = joblib.load(os.path.join(base_path, 'logreg_kategori_model.pkl'))
            tfidf_vectorizer = joblib.load(os.path.join(base_path, 'tfidf_vectorizer.pkl'))
            
            with open(os.path.join(base_path, 'label_encoder_kategori.pkl'), 'rb') as f:
                label_encoder_kategori = pickle.load(f)
            
            return {
                "sentiment_model": sentiment_model,
                "tokenizer": tokenizer,
                "max_len": config['max_sequence_length'],
                "label_encoder_sentimen": label_encoder_sentimen,
                "kategori_model": kategori_model,
                "tfidf_vectorizer": tfidf_vectorizer,
                "label_encoder_kategori": label_encoder_kategori
            }
        except Exception as e:
            app.logger.error(f"Failed to load models: {str(e)}")
            return {}

class DataManager:
    """Handles dataset operations"""
    
    @staticmethod
    def get_dataset_path(source: str) -> str:
        """Get path to dataset file"""
        return os.path.join(os.getcwd(), source, 'kategori.xlsx')
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    @staticmethod
    def load_dataset(source: str) -> pd.DataFrame:
        """Load and standardize dataset"""
        file_path = DataManager.get_dataset_path(source)
        
        if not os.path.exists(file_path):
            if source == 'user':
                # Create empty user dataset if it doesn't exist
                df = pd.DataFrame(columns=REQUIRED_COLUMNS)
                df.to_excel(file_path, index=False)
                return df
            else:
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        df = pd.read_excel(file_path)
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Ensure required columns exist
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        
        return df
    
    @staticmethod
    def save_dataset(df: pd.DataFrame, source: str) -> None:
        """Save dataset to file"""
        file_path = DataManager.get_dataset_path(source)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_excel(file_path, index=False)
    
    @staticmethod
    def filter_by_date_range(df: pd.DataFrame, start_year: int = 2024, end_year: int = 2025) -> pd.DataFrame:
        """Filter dataframe by date range"""
        if 'tanggal' not in df.columns:
            return df
        
        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
        return df[(df['tanggal'].dt.year >= start_year) & (df['tanggal'].dt.year <= end_year)]

class SentimentAnalyzer:
    """Handles sentiment and category analysis"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def predict_sentiment_and_category(self, text: str) -> Tuple[str, str]:
        """Predict sentiment and category for given text"""
        models = self.model_manager.models
        if not models:
            raise RuntimeError("Models not loaded")
        
        cleaned_text = DataManager.preprocess_text(text)
        
        # Predict sentiment
        seq = models["tokenizer"].texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=models["max_len"], padding='post')
        sent_pred = models["sentiment_model"].predict(padded)[0][0]
        sent_label = 1 if sent_pred > 0.5 else 0
        sentiment = models["label_encoder_sentimen"].inverse_transform([sent_label])[0]
        
        # Predict category
        tfidf_input = models["tfidf_vectorizer"].transform([cleaned_text])
        kategori_pred = models["kategori_model"].predict(tfidf_input)
        category = models["label_encoder_kategori"].inverse_transform(kategori_pred)[0]
        
        return sentiment, category

# Initialize managers
model_manager = ModelManager(MODEL_FOLDER)
data_manager = DataManager()
analyzer = SentimentAnalyzer(model_manager)

# Helper functions
def validate_source(source: str) -> bool:
    """Validate if source is allowed"""
    return source in VALID_SOURCES

def create_error_response(message: str, status_code: int = 400) -> Tuple[Dict, int]:
    """Create standardized error response"""
    return jsonify({"error": message}), status_code

def create_success_response(data: Any, message: str = None) -> Dict:
    """Create standardized success response"""
    response = {"data": data}
    if message:
        response["message"] = message
    return jsonify(response)

# API Endpoints
@app.route("/dataset", methods=["GET"])
def get_dataset():
    """Get dataset for specific source"""
    source = request.args.get("source")
    
    if not source or not validate_source(source):
        return create_error_response("Invalid or missing source")
    
    try:
        df = data_manager.load_dataset(source)
        
        # Sort by batch if available, otherwise by index
        if 'batch' in df.columns:
            df = df.sort_values(by='batch', ascending=False)
        
        data = df[REQUIRED_COLUMNS + (['batch'] if 'batch' in df.columns else [])].to_dict(orient="records")
        return create_success_response(data)
        
    except Exception as e:
        app.logger.error(f"Error loading dataset for {source}: {str(e)}")
        return create_error_response(f"Failed to load dataset: {str(e)}", 500)

@app.route("/dataset/stats", methods=["GET"])
def get_dataset_stats():
    """Get dataset statistics"""
    source = request.args.get("source")
    
    if not source or source not in ['agoda', 'traveloka', 'tiket']:
        return create_error_response("Invalid source")
    
    try:
        df = data_manager.load_dataset(source)
        df = data_manager.filter_by_date_range(df)
        
        # Calculate statistics
        sentimen_count = df['sentiment'].str.lower().value_counts().to_dict()
        kategori_count = df['kategori'].str.lower().value_counts().to_dict()
        
        # Category-sentiment distribution
        kategori_sentimen = {}
        for _, row in df.iterrows():
            kat = str(row['kategori']).strip().lower()
            sent = str(row['sentiment']).strip().lower()
            if kat not in kategori_sentimen:
                kategori_sentimen[kat] = {"positif": 0, "negatif": 0}
            if sent in kategori_sentimen[kat]:
                kategori_sentimen[kat][sent] += 1
        
        # Monthly sentiment data
        df['bulan'] = df['tanggal'].dt.to_period("M").astype(str)
        monthly_sentiment = (
            df.groupby(['bulan', 'sentiment'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        
        monthly_sentiment_data = []
        for _, row in monthly_sentiment.iterrows():
            monthly_sentiment_data.append({
                "bulan": row['bulan'],
                "Positif": row.get('positif', 0),
                "Negatif": row.get('negatif', 0)
            })
        
        stats = {
            "sentimen": sentimen_count,
            "kategori": kategori_count,
            "kategori_sentimen": kategori_sentimen,
            "monthly_sentiment": monthly_sentiment_data
        }
        
        return create_success_response(stats)
        
    except Exception as e:
        app.logger.error(f"Error getting stats for {source}: {str(e)}")
        return create_error_response(f"Failed to get statistics: {str(e)}", 500)

@app.route("/dataset/compare-sentiment", methods=["GET"])
def compare_sentiment():
    """Compare sentiment across all sources"""
    sources = ['agoda', 'traveloka', 'tiket']
    comparison = []
    monthly_comparison = {}
    kategori_comparison = {}
    
    try:
        for source in sources:
            try:
                df = data_manager.load_dataset(source)
                df = data_manager.filter_by_date_range(df)
                
                # Sentiment totals
                sentiment_counts = df['sentiment'].str.lower().value_counts().to_dict()
                comparison.append({
                    "source": source.capitalize(),
                    "positif": sentiment_counts.get("positif", 0),
                    "negatif": sentiment_counts.get("negatif", 0),
                })
                
                # Monthly data
                df['bulan'] = df['tanggal'].dt.to_period("M").astype(str)
                monthly = df.groupby(['bulan', 'sentiment']).size().unstack(fill_value=0).reset_index()
                formatted_monthly = []
                for _, row in monthly.iterrows():
                    formatted_monthly.append({
                        "bulan": row['bulan'],
                        "Positif": row.get('positif', 0),
                        "Negatif": row.get('negatif', 0),
                    })
                monthly_comparison[source.capitalize()] = formatted_monthly
                
                # Category data
                if 'kategori' in df.columns:
                    kategori_sentimen_counts = (
                        df.groupby(['kategori', 'sentiment'])
                          .size()
                          .unstack(fill_value=0)
                          .reset_index()
                    )
                    
                    kategori_result = {}
                    for _, row in kategori_sentimen_counts.iterrows():
                        kategori_result[row['kategori']] = {
                            "positif": int(row.get('positif', 0)),
                            "negatif": int(row.get('negatif', 0)),
                        }
                    
                    kategori_comparison[source.capitalize()] = kategori_result
                    
            except Exception as e:
                app.logger.warning(f"Error processing {source}: {str(e)}")
                continue
        
        result = {
            "comparison": comparison,
            "monthly_comparison": monthly_comparison,
            "kategori_comparison": kategori_comparison
        }
        
        return create_success_response(result)
        
    except Exception as e:
        app.logger.error(f"Error in comparison: {str(e)}")
        return create_error_response(f"Failed to compare data: {str(e)}", 500)

@app.route("/dataset/update", methods=["POST"])
def update_dataset():
    """Update dataset by scraping new reviews"""
    data = request.get_json()
    source = data.get("source")
    
    if source not in APP_IDS:
        return create_error_response("Invalid source")
    
    try:
        app_id = APP_IDS[source]
        
        # Scrape reviews
        result, _ = reviews(
            app_id,
            lang='id',
            country='id',
            sort=Sort.MOST_RELEVANT,
            count=100,
        )
        
        # Process scraped data
        df = pd.DataFrame(result)
        df = df[['score', 'content']].rename(columns={'score': 'rating', 'content': 'review'})
        df.drop_duplicates(subset='review', inplace=True)
        
        # Predict sentiment and category
        predictions = []
        for review in df['review']:
            try:
                sentiment, category = analyzer.predict_sentiment_and_category(review)
                predictions.append({'sentiment': sentiment, 'kategori': category})
            except Exception as e:
                app.logger.warning(f"Error predicting for review: {str(e)}")
                predictions.append({'sentiment': 'unknown', 'kategori': 'unknown'})
        
        pred_df = pd.DataFrame(predictions)
        df = pd.concat([df, pred_df], axis=1)
        df['tanggal'] = datetime.now().strftime('%Y-%m-%d')
        
        # Merge with existing dataset
        try:
            existing_df = data_manager.load_dataset(source)
            max_batch = existing_df['batch'].max() if 'batch' in existing_df.columns else 0
            current_batch = max_batch + 1
        except:
            existing_df = pd.DataFrame()
            current_batch = 1
        
        df['batch'] = current_batch
        
        # Filter out duplicates
        if not existing_df.empty:
            existing_reviews = set(existing_df['review'].dropna().astype(str))
            new_data = df[~df['review'].astype(str).isin(existing_reviews)]
            
            if new_data.empty:
                return create_success_response({"new_count": 0}, "No new data to update")
            
            combined = pd.concat([existing_df, new_data], ignore_index=True)
        else:
            combined = df
            new_data = df
        
        data_manager.save_dataset(combined, source)
        
        return create_success_response(
            {"new_count": len(new_data)}, 
            "Dataset updated successfully"
        )
        
    except Exception as e:
        app.logger.error(f"Error updating dataset for {source}: {str(e)}")
        return create_error_response(f"Failed to update dataset: {str(e)}", 500)

@app.route("/analyze", methods=["POST"])
def analyze_review():
    """Analyze single review for sentiment and category"""
    try:
        data = request.get_json()
        if not data:
            return create_error_response("No data provided")
        
        review = data.get("review", "").strip()
        if not review:
            return create_error_response("Review cannot be empty")
        
        sentiment, category = analyzer.predict_sentiment_and_category(review)
        
        result = {
            "review": review,
            "sentimen": sentiment,
            "kategori": category
        }
        
        return create_success_response(result)
        
    except Exception as e:
        app.logger.error(f"Error analyzing review: {str(e)}")
        return create_error_response(f"Failed to analyze review: {str(e)}", 500)

@app.route("/dataset-user", methods=["GET"])
def get_user_dataset():
    """Get user dataset"""
    try:
        df = data_manager.load_dataset('user')
        
        # Sort by batch or date if available
        if 'batch' in df.columns:
            df = df.sort_values(by='batch', ascending=False)
        elif 'tanggal' in df.columns:
            try:
                df['tanggal_sort'] = pd.to_datetime(df['tanggal'], format='%d-%m-%Y', errors='coerce')
                df = df.sort_values(by='tanggal_sort', ascending=False, na_position='last')
                df = df.drop('tanggal_sort', axis=1)
            except:
                pass
        
        # Clean data
        for col in REQUIRED_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna("")
        
        data = df[REQUIRED_COLUMNS].to_dict(orient="records")
        return create_success_response(data)
        
    except Exception as e:
        app.logger.error(f"Error getting user dataset: {str(e)}")
        return create_error_response(f"Failed to get user dataset: {str(e)}", 500)

@app.route("/save-user-sentiment", methods=["POST"])
def save_user_sentiment():
    """Save user sentiment data"""
    try:
        data = request.json
        if not data:
            return create_error_response("No data provided")
        
        review = data.get("review", "").strip()
        sentimen = data.get("sentimen", "").strip()
        kategori = data.get("kategori", "").strip()
        
        if not all([review, sentimen, kategori]):
            return create_error_response("Incomplete data")
        
        df = data_manager.load_dataset('user')
        
        # Check for duplicates
        df['review'] = df['review'].astype(str).str.strip()
        if review in df["review"].values:
            return create_error_response("Review already exists in database")
        
        # Add new row
        new_row = pd.DataFrame([{
            "review": review,
            "sentiment": sentimen,
            "kategori": kategori,
            "tanggal": datetime.now().strftime("%d-%m-%Y")
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        data_manager.save_dataset(df, 'user')
        
        return create_success_response({}, "Data saved successfully")
        
    except Exception as e:
        app.logger.error(f"Error saving user sentiment: {str(e)}")
        return create_error_response(f"Failed to save data: {str(e)}", 500)

@app.route("/update-user", methods=["POST"])
def update_user_sentiment():
    """Update user sentiment data"""
    try:
        data = request.json
        if not data:
            return create_error_response("No data provided")
        
        review_to_update = data.get("review", "").strip()
        new_sentiment = data.get("sentimen", "").strip()
        new_kategori = data.get("kategori", "").strip()
        
        if not all([review_to_update, new_sentiment, new_kategori]):
            return create_error_response("Incomplete data")
        
        df = data_manager.load_dataset('user')
        df['review'] = df['review'].astype(str).str.strip()
        
        matching_rows = df[df['review'] == review_to_update]
        if matching_rows.empty:
            return create_error_response("Review not found")
        
        # Update first matching row
        idx = matching_rows.index[0]
        df.loc[idx, 'sentiment'] = new_sentiment
        df.loc[idx, 'kategori'] = new_kategori
        df.loc[idx, 'tanggal'] = datetime.now().strftime("%d-%m-%Y")
        
        data_manager.save_dataset(df, 'user')
        
        return create_success_response({}, "Data updated successfully")
        
    except Exception as e:
        app.logger.error(f"Error updating user sentiment: {str(e)}")
        return create_error_response(f"Failed to update data: {str(e)}", 500)

@app.route("/delete-user", methods=["POST"])
def delete_user_sentiment():
    """Delete user sentiment data"""
    try:
        data = request.json
        if not data:
            return create_error_response("No data provided")
        
        review_to_delete = data.get("review", "").strip()
        if not review_to_delete:
            return create_error_response("Review cannot be empty")
        
        df = data_manager.load_dataset('user')
        df['review'] = df['review'].astype(str).str.strip()
        
        initial_count = len(df)
        df = df[df['review'] != review_to_delete]
        
        if len(df) == initial_count:
            return create_error_response("Review not found")
        
        data_manager.save_dataset(df, 'user')
        
        return create_success_response({}, "Data deleted successfully")
        
    except Exception as e:
        app.logger.error(f"Error deleting user sentiment: {str(e)}")
        return create_error_response(f"Failed to delete data: {str(e)}", 500)

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)