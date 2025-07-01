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

# Helper Function join path
def dataset_path(source):
    return os.path.join(os.getcwd(), source, 'kategori.xlsx')

# Helper function untuk generate ID unik
def generate_unique_id(df):
    if df.empty or 'id' not in df.columns:
        return 1
    return df['id'].max() + 1

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
            "sentimen": str(sent_result),
            "kategori": str(kategori_label)
        })

    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        return jsonify({"error": f"Gagal menganalisis review: {str(e)}"}), 500

# API get data user - Fixed version dengan ID
@app.route("/dataset-user", methods=["GET"])
def get_user_dataset():
    try:
        file_path = dataset_path('user')
        if not os.path.exists(file_path):
            # Create empty file if doesn't exist
            df = pd.DataFrame(columns=["id", "review", "sentiment", "kategori", "tanggal"])
            df.to_excel(file_path, index=False)
            return jsonify([])

        df = pd.read_excel(file_path)
        
        # Normalize column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Ensure required columns exist
        required_columns = ['id', 'review', 'sentiment', 'kategori', 'tanggal']
        for col in required_columns:
            if col not in df.columns:
                if col == 'id':
                    df[col] = range(1, len(df) + 1)
                else:
                    df[col] = ""

        # Convert ID to integer
        df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)
        
        # Sort by ID descending (newest first)
        df = df.sort_values(by='id', ascending=False)
        
        # Select only required columns and convert to records
        data_columns = ['id', 'review', 'sentiment', 'kategori', 'tanggal']
        available_columns = [col for col in data_columns if col in df.columns]
        
        # Fill missing values
        for col in available_columns:
            if col != 'id':
                df[col] = df[col].fillna("")
        
        data = df[available_columns].to_dict(orient="records")
        
        return jsonify(data)
        
    except Exception as e:
        print(f"Error in get_user_dataset: {str(e)}")
        return jsonify({"error": f"Gagal membaca dataset user: {str(e)}"}), 500

# API update dataset user - Fixed version with better column handling
@app.route("/update-user", methods=["POST"])
def update_user_sentiment():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        record_id = data.get("id")
        new_review = data.get("review", "").strip()
        new_sentiment = data.get("sentimen", "").strip()
        new_kategori = data.get("kategori", "").strip()
        
        print(f"Received update request: ID={record_id}, Review={new_review[:50]}..., Sentiment={new_sentiment}, Category={new_kategori}")
        
        if not all([record_id, new_sentiment, new_kategori]):
            return jsonify({"error": "Data tidak lengkap (ID, sentimen, dan kategori diperlukan)"}), 400

        new_tanggal = datetime.now().strftime("%d-%m-%Y")

        file_path = dataset_path('user')
        if not os.path.exists(file_path):
            return jsonify({"error": "File dataset tidak ditemukan"}), 404

        df = pd.read_excel(file_path)
        
        # Debug: Print original column names
        print(f"Original columns: {list(df.columns)}")
        
        # Create a mapping of original columns to standardized names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'id' in col_lower:
                column_mapping[col] = 'id'
            elif 'review' in col_lower or 'ulasan' in col_lower:
                column_mapping[col] = 'review'
            elif 'sentiment' in col_lower or 'sentimen' in col_lower:
                column_mapping[col] = 'sentiment'
            elif 'kategori' in col_lower or 'category' in col_lower:
                column_mapping[col] = 'kategori'
            elif 'tanggal' in col_lower or 'date' in col_lower:
                column_mapping[col] = 'tanggal'
            else:
                column_mapping[col] = col_lower.strip()
        
        # Rename columns using the mapping
        df = df.rename(columns=column_mapping)
        
        print(f"Mapped columns: {list(df.columns)}")
        
        # Check if required columns exist
        required_columns = ['id', 'review', 'sentiment', 'kategori', 'tanggal']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({"error": f"Kolom yang diperlukan tidak ditemukan: {missing_columns}. Kolom yang tersedia: {list(df.columns)}"}), 400
        
        # Ensure ID column is numeric and handle any conversion issues
        try:
            df['id'] = pd.to_numeric(df['id'], errors='coerce')
            # Remove rows with invalid IDs (NaN)
            df = df.dropna(subset=['id'])
            df['id'] = df['id'].astype(int)
        except Exception as e:
            return jsonify({"error": f"Error processing ID column: {str(e)}"}), 500

        # Find the record to update by ID
        record_id_int = int(record_id)
        matching_rows = df[df['id'] == record_id_int]
        
        print(f"Looking for ID {record_id_int}, found {len(matching_rows)} matching rows")
        
        if matching_rows.empty:
            available_ids = df['id'].tolist()
            return jsonify({"error": f"Data dengan ID {record_id} tidak ditemukan. ID yang tersedia: {available_ids}"}), 404

        # Check if new review text already exists in other records (exclude current record)
        if new_review:
            df['review'] = df['review'].astype(str).str.strip()
            existing_review = df[(df['review'] == new_review) & (df['id'] != record_id_int)]
            if not existing_review.empty:
                existing_id = existing_review.iloc[0]['id']
                return jsonify({"error": f"Review text sudah ada dalam database dengan ID {existing_id}"}), 400

        # Update the record
        idx = matching_rows.index[0]
        
        print(f"Updating record at index {idx}")
        
        # Update review text if provided
        if new_review:
            df.loc[idx, 'review'] = new_review
            
        df.loc[idx, 'sentiment'] = new_sentiment
        df.loc[idx, 'kategori'] = new_kategori
        df.loc[idx, 'tanggal'] = new_tanggal
        
        # Save the updated dataframe
        try:
            df.to_excel(file_path, index=False)
            print(f"Successfully updated record ID {record_id}")
        except Exception as e:
            return jsonify({"error": f"Gagal menyimpan file: {str(e)}"}), 500

        update_message = f"Data dengan ID {record_id} berhasil diperbarui."
        if new_review:
            update_message += " Review text juga telah diperbarui."

        return jsonify({
            "success": True, 
            "message": update_message,
            "id": record_id,
            "updated_data": {
                "review": new_review if new_review else df.loc[idx, 'review'],
                "sentiment": new_sentiment,
                "kategori": new_kategori,
                "tanggal": new_tanggal
            }
        })
        
    except ValueError as e:
        return jsonify({"error": f"Invalid ID format: {str(e)}"}), 400
    except Exception as e:
        print(f"Error in update_user_sentiment: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Gagal memperbarui data: {str(e)}"}), 500

# API save dataset user - Fixed version dengan ID
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
            required_columns = ["id", "review", "sentiment", "kategori", "tanggal"]
            for col in required_columns:
                if col not in df.columns:
                    if col == 'id':
                        df[col] = range(1, len(df) + 1) if not df.empty else []
                    else:
                        df[col] = ""
        else:
            df = pd.DataFrame(columns=["id", "review", "sentiment", "kategori", "tanggal"])

        # Convert ID to integer if exists
        if not df.empty and 'id' in df.columns:
            df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)

        # Convert review column to string for comparison
        df['review'] = df['review'].astype(str).str.strip()

        # Check if review already exists
        if review in df["review"].values:
            return jsonify({"error": "Review sudah ada dalam database"}), 400

        # Generate new ID
        new_id = generate_unique_id(df)

        # Create new row
        new_row = pd.DataFrame([{
            "id": new_id,
            "review": review,
            "sentiment": sentimen,
            "kategori": kategori,
            "tanggal": tanggal
        }])
        
        # Append new row
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save to file
        df.to_excel(file_path, index=False)

        return jsonify({
            "success": True, 
            "message": "Data berhasil disimpan.",
            "id": int(new_id)  # konversi ke int standar Python
        })

        
    except Exception as e:
        print(f"Error in save_user_sentiment: {str(e)}")
        return jsonify({"error": f"Gagal menyimpan data: {str(e)}"}), 500

# API delete user data - Implementasi yang hilang
@app.route("/delete-user", methods=["POST"])
def delete_user_sentiment():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        record_id = data.get("id")
        
        if not record_id:
            return jsonify({"error": "ID diperlukan untuk menghapus data"}), 400

        file_path = dataset_path('user')
        if not os.path.exists(file_path):
            return jsonify({"error": "File dataset tidak ditemukan"}), 404

        df = pd.read_excel(file_path)
        
        # Normalize column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Ensure ID column exists
        if 'id' not in df.columns:
            return jsonify({"error": "Kolom ID tidak ditemukan"}), 404
            
        df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)

        # Find the record to delete
        matching_rows = df[df['id'] == int(record_id)]
        
        if matching_rows.empty:
            return jsonify({"error": f"Data dengan ID {record_id} tidak ditemukan"}), 404

        # Remove the record
        df = df[df['id'] != int(record_id)]
        
        # Save the updated dataframe
        df.to_excel(file_path, index=False)

        return jsonify({
            "success": True, 
            "message": f"Data dengan ID {record_id} berhasil dihapus."
        })
        
    except Exception as e:
        print(f"Error in delete_user_sentiment: {str(e)}")
        return jsonify({"error": f"Gagal menghapus data: {str(e)}"}), 500
    
if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)