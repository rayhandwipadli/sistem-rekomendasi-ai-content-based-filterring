from flask import Flask, jsonify, request 
import mysql.connector 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database=""
    )

def get_all_products():
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM produk")
    products = cursor.fetchall()
    db.close()
    return products

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = text.replace('-', ' ')
    return text

def get_recommendations(produk_id):
    products = get_all_products()
    produk_dict = {p['produk_id']: p for p in products}

    if produk_id not in produk_dict:
        return []

    selected_product = produk_dict[produk_id]
    selected_category = selected_product["produk_kategori"]
    selected_price = selected_product["produk_harga"]

    descriptions = [
        clean_text((p["produk_nama"] or "") + " " + (p["produk_keterangan"] or ""))
        for p in products
    ]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    idx = next(i for i, p in enumerate(products) if p["produk_id"] == produk_id)
    similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    recommendations = []
    for i, score in enumerate(similarity_scores):
        if products[i]["produk_id"] == produk_id:
            continue
        if score >= 0.2:
            product = products[i]
            price_range = selected_price * 0.8 <= product["produk_harga"] <= selected_price * 1.2
            if product["produk_kategori"] == selected_category and price_range:
                product_with_score = product.copy()
                product_with_score["similarity_score"] = round(float(score), 4)
                recommendations.append(product_with_score)

    return sorted(recommendations, key=lambda x: x["similarity_score"], reverse=True)

@app.route('/rekomendasi', methods=['GET'])
def rekomendasi_produk():
    produk_id = request.args.get('produk_id', type=int)
    
    if not produk_id:
        return jsonify({"error": "produk_id diperlukan"}), 400
    
    rekomendasi = get_recommendations(produk_id)
    
    if not rekomendasi:
        return jsonify({"error": "Tidak ada rekomendasi yang ditemukan"}), 404

    return jsonify({"rekomendasi": rekomendasi})

if __name__ == '__main__':
    app.run(debug=True)
