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


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}) # supaya bisa diakses oleh FE
# Preprocessing teks
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load semua file model
def load_models(source_folder):
    base_path = os.path.join(os.getcwd(), source_folder)
    # Sentimen model
    sentiment_model = load_model(os.path.join(base_path, 'bilstm_sentiment_model.h5'))
    with open(os.path.join(base_path, 'tokenizer.pickle'), 'rb') as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(base_path, 'model_seq.pickle'), 'rb') as f:
        config = pickle.load(f)
    max_len = config['max_sequence_length']
    with open(os.path.join(base_path, 'label_encoder_sentimen.pkl'), 'rb') as f:
        label_encoder_sentimen = pickle.load(f)

    # Kategori model
    kategori_model = joblib.load(os.path.join(base_path, 'logreg_kategori_model.pkl'))
    tfidf_vectorizer = joblib.load(os.path.join(base_path, 'tfidf_vectorizer.pkl'))
    with open(os.path.join(base_path, 'label_encoder_kategori.pkl'), 'rb') as f:
        label_encoder_kategori = pickle.load(f)

    return {
        "sentiment_model": sentiment_model,
        "tokenizer": tokenizer,
        "max_len": max_len,
        "label_encoder_sentimen": label_encoder_sentimen,
        "kategori_model": kategori_model,
        "tfidf_vectorizer": tfidf_vectorizer,
        "label_encoder_kategori": label_encoder_kategori
    }

# Helper Function
def dataset_path(source):
    return os.path.join(os.getcwd(), source, 'kategori.xlsx')

# API tampil dataset
@app.route("/dataset", methods=["GET"])
def get_data():
    source = request.args.get("source")
    if not source or source not in ['agoda', 'traveloka', 'tiket']:
        return jsonify({"error": "Invalid or missing source"}), 400

    try:
        file_path = dataset_path(source)
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404

        df = pd.read_excel(file_path)
        df.columns = [col.lower() for col in df.columns]

        # Urutkan berdasarkan batch DESC jika ada
        if 'batch' in df.columns:
            df = df.sort_values(by='batch', ascending=False)

        expected_columns = {'sentiment', 'kategori', 'review', 'tanggal'}
        missing_columns = expected_columns - set(df.columns)
        if missing_columns:
            return jsonify({
                "error": f"Missing expected columns: {', '.join(missing_columns)}"
            }), 500

        data = df[['review', 'sentiment', 'kategori', 'batch']].to_dict(orient="records")
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Gagal membaca dataset: {str(e)}"}), 500

# API jumlah dataset
@app.route("/dataset/stats", methods=["GET"])
def get_distribution_stats():
    source = request.args.get("source")
    if source not in ['agoda', 'traveloka', 'tiket']:
        return jsonify({"error": "Invalid source"}), 400
    try:
        file_path = dataset_path(source)
        df = pd.read_excel(file_path)

        df.columns = [col.lower() for col in df.columns] # lowercase kolom

        # Pastikan kolom 'tanggal' ada
        if 'tanggal' not in df.columns:
            return jsonify({"error": "Kolom 'tanggal' tidak ditemukan"}), 400
        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')

        # Filter data hanya untuk tahun 2024-2025
        df = df[(df['tanggal'].dt.year >= 2024) & (df['tanggal'].dt.year <= 2025)]
        sentimen_count = df['sentiment'].str.lower().value_counts().to_dict()
        kategori_count = df['kategori'].str.lower().value_counts().to_dict()

        # Hitung distribusi kategori per sentimen
        kategori_sentimen = {}
        for _, row in df.iterrows():
            kat = str(row['kategori']).strip().lower()
            sent = str(row['sentiment']).strip().lower()
            if kat not in kategori_sentimen:
                kategori_sentimen[kat] = {"positif": 0, "negatif": 0}
            if sent in kategori_sentimen[kat]:
                kategori_sentimen[kat][sent] += 1

        df['bulan'] = df['tanggal'].dt.to_period("M").astype(str)  # e.g. '2024-01'

        # Hitung jumlah sentimen per bulan
        monthly_sentiment = (
            df.groupby(['bulan', 'sentiment'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        # Format agar cocok dengan line chart frontend
        monthly_sentiment_data = []
        for _, row in monthly_sentiment.iterrows():
            monthly_sentiment_data.append({
                "bulan": row['bulan'],
                "Positif": row.get('positif', 0),
                "Negatif": row.get('negatif', 0)
            })

        return jsonify({
            "sentimen": sentimen_count,
            "kategori": kategori_count,
            "kategori_sentimen": kategori_sentimen,
            "monthly_sentiment": monthly_sentiment_data 
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API Compare
@app.route("/dataset/compare-sentiment", methods=["GET"])
def compare_sentiment_across_sources():
    sources = ['agoda', 'traveloka', 'tiket']
    comparison = []
    monthly_comparison = {}
    kategori_comparison = {}

    try:
        for source in sources:
            file_path = dataset_path(source)
            df = pd.read_excel(file_path)
            df.columns = [col.lower() for col in df.columns]

            if 'tanggal' not in df.columns or 'sentiment' not in df.columns:
                continue

            df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
            df = df[(df['tanggal'].dt.year >= 2024) & (df['tanggal'].dt.year <= 2025)]

            # === Pie Chart Sentimen Total ===
            sentiment_counts = df['sentiment'].str.lower().value_counts().to_dict()
            comparison.append({
                "source": source.capitalize(),
                "positif": sentiment_counts.get("positif", 0),
                "negatif": sentiment_counts.get("negatif", 0),
            })

            # === Line Chart Sentimen Bulanan ===
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

            # === Bar Chart Kategori per Aplikasi dengan Sentimen ===
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

        return jsonify({
            "comparison": comparison,
            "monthly_comparison": monthly_comparison,
            "kategori_comparison": kategori_comparison
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Update Dataset
@app.route("/dataset/update", methods=["POST"])
def update_dataset():
    data = request.get_json()
    source = data.get("source")

    app_ids = {
        "traveloka": "com.traveloka.android",
        "tiket": "com.tiket.gits",
        "agoda": "com.agoda.mobile.consumer"
    }

    if source not in app_ids:
        return jsonify({"error": "Invalid source"}), 400

    try:
        app_id = app_ids[source]

        # Scrape reviews
        result, _ = reviews(
            app_id,
            lang='id',
            country='id',
            sort=Sort.MOST_RELEVANT,
            count=100,
        )

        df = pd.DataFrame(result)
        df = df[['score', 'content']]
        df = df.rename(columns={'score': 'rating', 'content': 'review'})
        df.drop_duplicates(subset='review', inplace=True)

        # Preprocessing
        df['review_clean'] = df['review'].apply(preprocess_text)

        # Load models
        models = load_models("trying")

        # Sentiment prediction
        seq = models["tokenizer"].texts_to_sequences(df['review_clean'].tolist())
        padded = pad_sequences(seq, maxlen=models["max_len"], padding='post')
        pred_sentiments = models["sentiment_model"].predict(padded)
        labels_sent = (pred_sentiments > 0.5).astype(int).flatten()
        df['sentiment'] = models["label_encoder_sentimen"].inverse_transform(labels_sent)

        # Kategori prediction
        tfidf_input = models["tfidf_vectorizer"].transform(df['review_clean'].tolist())
        pred_kategori = models["kategori_model"].predict(tfidf_input)
        df['kategori'] = models["label_encoder_kategori"].inverse_transform(pred_kategori)

        # Tambahkan tanggal scraping
        df['tanggal'] = datetime.now().strftime('%Y-%m-%d')

        # Simpan ke file kategori.xlsx
        save_path = dataset_path(source)
        # Hitung batch keberapa
        if os.path.exists(save_path):
            existing_df = pd.read_excel(save_path)

            # Pastikan kolom 'tanggal' dan 'batch' ada
            if 'tanggal' not in existing_df.columns:
                existing_df['tanggal'] = pd.NaT
            if 'batch' not in existing_df.columns:
                existing_df['batch'] = 1  # asumsikan data lama masuk ke batch 1

            max_batch = existing_df['batch'].max()
            current_batch = max_batch + 1
        else:
            existing_df = pd.DataFrame()
            current_batch = 1

        # Tambahkan kolom batch ke data baru
        df['batch'] = current_batch

        # Gabungkan dengan dataset lama
        new_data = df[['sentiment', 'kategori', 'review', 'tanggal', 'batch']]
        if not existing_df.empty:
            existing_reviews_set = set(existing_df['review'].dropna().astype(str))
            new_data_filtered = new_data[~new_data['review'].astype(str).isin(existing_reviews_set)]

            if new_data_filtered.empty:
                return jsonify({"message": "Tidak ada pembaruan dataset terbaru", "new_count": 0})

            new_data_filtered['batch'] = current_batch
            combined = pd.concat([existing_df, new_data_filtered], ignore_index=True)
            combined.to_excel(save_path, index=False)

            return jsonify({"message": "Dataset updated successfully", "new_count": len(new_data_filtered)})

        else:
            combined = new_data

        combined.to_excel(save_path, index=False)

        return jsonify({"message": "Dataset updated successfully", "new_count": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API analisis - Fixed version
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        review = data.get("review", "").strip()
        
        if not review:
            return jsonify({"error": "Review tidak boleh kosong"}), 400

        # Load models
        models = load_models("trying")
        if not models:
            return jsonify({"error": "Model tidak dapat dimuat"}), 500

        # Preprocess and analyze
        cleaned_review = preprocess_text(review)

        # === Prediksi Sentimen ===
        seq = models["tokenizer"].texts_to_sequences([cleaned_review])
        padded = pad_sequences(seq, maxlen=models["max_len"], padding='post')
        sent_pred = models["sentiment_model"].predict(padded)[0][0]
        sent_label = 1 if sent_pred > 0.5 else 0
        sent_result = models["label_encoder_sentimen"].inverse_transform([sent_label])[0]

        # === Prediksi Kategori ===
        tfidf_input = models["tfidf_vectorizer"].transform([cleaned_review])
        kategori_pred = models["kategori_model"].predict(tfidf_input)
        kategori_label = models["label_encoder_kategori"].inverse_transform(kategori_pred)[0]

        return jsonify({
            "review": review,
            "sentimen": sent_result,
            "kategori": kategori_label
        })

    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        return jsonify({"error": f"Gagal menganalisis review: {str(e)}"}), 500

# API get data user - Fixed version
@app.route("/dataset-user", methods=["GET"])
def get_user_dataset():
    try:
        file_path = dataset_path('user')
        if not os.path.exists(file_path):
            # Create empty file if doesn't exist
            df = pd.DataFrame(columns=["review", "sentiment", "kategori", "tanggal"])
            df.to_excel(file_path, index=False)
            return jsonify([])

        df = pd.read_excel(file_path)
        
        # Normalize column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Ensure required columns exist
        required_columns = ['review', 'sentiment', 'kategori', 'tanggal']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

        # Sort by date if possible, otherwise by index (newest first)
        try:
            if 'batch' in df.columns:
                df = df.sort_values(by='batch', ascending=False)
            elif 'tanggal' in df.columns:
                # Try to sort by date
                df['tanggal_sort'] = pd.to_datetime(df['tanggal'], format='%d-%m-%Y', errors='coerce')
                df = df.sort_values(by='tanggal_sort', ascending=False, na_position='last')
                df = df.drop('tanggal_sort', axis=1)
            else:
                # Sort by index (newest first)
                df = df.iloc[::-1]
        except Exception as sort_error:
            print(f"Sorting error: {sort_error}")
            # Continue without sorting

        # Select only required columns and convert to records
        data_columns = ['review', 'sentiment', 'kategori', 'tanggal']
        available_columns = [col for col in data_columns if col in df.columns]
        
        # Fill missing values
        for col in available_columns:
            df[col] = df[col].fillna("")
        
        data = df[available_columns].to_dict(orient="records")
        
        return jsonify(data)
        
    except Exception as e:
        print(f"Error in get_user_dataset: {str(e)}")
        return jsonify({"error": f"Gagal membaca dataset user: {str(e)}"}), 500

# API update dataset user - Fixed version
@app.route("/update-user", methods=["POST"])
def update_user_sentiment():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        review_to_update = data.get("review", "").strip()
        new_sentiment = data.get("sentimen", "").strip()
        new_kategori = data.get("kategori", "").strip()
        
        if not all([review_to_update, new_sentiment, new_kategori]):
            return jsonify({"error": "Data tidak lengkap"}), 400

        new_tanggal = datetime.now().strftime("%d-%m-%Y")

        file_path = dataset_path('user')
        if not os.path.exists(file_path):
            return jsonify({"error": "File dataset tidak ditemukan"}), 404

        df = pd.read_excel(file_path)
        
        # Normalize column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Ensure review column exists and convert to string
        if 'review' not in df.columns:
            return jsonify({"error": "Kolom review tidak ditemukan"}), 404
            
        df['review'] = df['review'].astype(str).str.strip()

        # Find the review to update
        matching_rows = df[df['review'] == review_to_update]
        
        if matching_rows.empty:
            return jsonify({"error": "Review tidak ditemukan untuk diupdate"}), 404

        # Update the first matching row
        idx = matching_rows.index[0]
        df.loc[idx, 'sentiment'] = new_sentiment
        df.loc[idx, 'kategori'] = new_kategori
        df.loc[idx, 'tanggal'] = new_tanggal
        
        # Save the updated dataframe
        df.to_excel(file_path, index=False)

        return jsonify({"success": True, "message": "Data berhasil diperbarui."})
        
    except Exception as e:
        print(f"Error in update_user_sentiment: {str(e)}")
        return jsonify({"error": f"Gagal memperbarui data: {str(e)}"}), 500

# API save dataset user - Fixed version
@app.route("/save-user-sentiment", methods=["POST"])
def save_user_sentiment():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        review = data.get("review", "").strip()
        sentimen = data.get("sentimen", "").strip()
        kategori = data.get("kategori", "").strip()
        
        if not all([review, sentimen, kategori]):
            return jsonify({"error": "Data tidak lengkap"}), 400

        tanggal = datetime.now().strftime("%d-%m-%Y")

        file_path = dataset_path('user')
        
        # Load existing data or create new dataframe
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            # Normalize column names
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Ensure required columns exist
            required_columns = ["review", "sentiment", "kategori", "tanggal"]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ""
        else:
            df = pd.DataFrame(columns=["review", "sentiment", "kategori", "tanggal"])

        # Convert review column to string for comparison
        df['review'] = df['review'].astype(str).str.strip()

        # Check if review already exists
        if review in df["review"].values:
            return jsonify({"error": "Review sudah ada dalam database"}), 400

        # Create new row
        new_row = pd.DataFrame([{
            "review": review,
            "sentiment": sentimen,
            "kategori": kategori,
            "tanggal": tanggal
        }])
        
        # Append new row
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save to file
        df.to_excel(file_path, index=False)

        return jsonify({"success": True, "message": "Data berhasil disimpan."})
        
    except Exception as e:
        print(f"Error in save_user_sentiment: {str(e)}")
        return jsonify({"error": f"Gagal menyimpan data: {str(e)}"}), 500

# API delete dataset user - New addition
@app.route("/delete-user", methods=["POST"])
def delete_user_sentiment():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        review_to_delete = data.get("review", "").strip()
        
        if not review_to_delete:
            return jsonify({"error": "Review tidak boleh kosong"}), 400

        file_path = dataset_path('user')
        if not os.path.exists(file_path):
            return jsonify({"error": "File dataset tidak ditemukan"}), 404

        df = pd.read_excel(file_path)
        
        # Normalize column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Ensure review column exists and convert to string
        if 'review' not in df.columns:
            return jsonify({"error": "Kolom review tidak ditemukan"}), 404
            
        df['review'] = df['review'].astype(str).str.strip()

        # Find and remove the review
        initial_count = len(df)
        df = df[df['review'] != review_to_delete]
        
        if len(df) == initial_count:
            return jsonify({"error": "Review tidak ditemukan"}), 404
        
        # Save the updated dataframe
        df.to_excel(file_path, index=False)

        return jsonify({"success": True, "message": "Data berhasil dihapus."})
        
    except Exception as e:
        print(f"Error in delete_user_sentiment: {str(e)}")
        return jsonify({"error": f"Gagal menghapus data: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)