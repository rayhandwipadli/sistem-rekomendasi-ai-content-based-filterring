import mysql.connector
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def connect_db():
    return mysql.connector.connect(
        host="",
        user="",
        password="",
        database=""
    )

def get_all_products():
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT produk_id, produk_nama, produk_keterangan FROM produk")
    products = cursor.fetchall()
    db.close()
    return products

def clean_text(text):
    """ Membersihkan teks: lowercase dan hapus tanda baca sederhana """
    if not text:
        return ""
    text = text.lower()
    text = text.replace('-', ' ')
    return text

def compute_cosine_similarity(products):
    """ Menghitung Cosine Similarity antara produk """
    descriptions = [
        clean_text((p["produk_nama"] or "") + " " + (p["produk_keterangan"] or ""))
        for p in products
    ]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(produk_id, cosine_sim, products):
    idx = next((i for i, p in enumerate(products) if p["produk_id"] == produk_id), None)
    if idx is None:
        return []

    rekomendasi = []
    for i in range(len(products)):
        if i != idx:
            sim_score = cosine_sim[idx][i]
            rekomendasi.append({
                "produk_id": products[i]["produk_id"],
                "produk_nama": products[i]["produk_nama"],
                "produk_keterangan": products[i]["produk_keterangan"],
                "cosine_similarity": round(sim_score, 4)
            })

    rekomendasi = sorted(rekomendasi, key=lambda x: x["cosine_similarity"], reverse=True)
    return rekomendasi

@app.route('/cek_cosine', methods=['GET'])
def cek_cosine_similarity():
    produk_id = request.args.get('produk_id', type=int)
    if not produk_id:
        return jsonify({"error": "produk_id diperlukan"}), 400

    products = get_all_products()
    cosine_sim = compute_cosine_similarity(products)
    hasil = get_recommendations(produk_id, cosine_sim, products)

    return jsonify({"rekomendasi": hasil})

if __name__ == '__main__':
    app.run(debug=True)
