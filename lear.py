import os, re, logging, numpy as np
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pyarabic.araby as araby

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
engine = create_engine("postgresql://major:s8NI3g6zUNdUJzJ@major.flycast:5432/major")

class SimilarityDetector:
    def __init__(self):
        self.model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        self.vectorizer = TfidfVectorizer()

    def preprocess(self, text):
        text = text.lower().strip()
        return re.sub(r"[^\u0600-\u06FFa-z\s]", "", araby.strip_tatweel(text))

    def get_embedding(self, text):
        return self.model.encode(self.preprocess(text), normalize_embeddings=True)

    def similarity(self, t1, t2):
        e1, e2 = self.get_embedding(t1), self.get_embedding(t2)
        return cosine_similarity([e1], [e2])[0][0]

    def lexical_similarity(self, t1, t2):
        try:
            tfidf_matrix = self.vectorizer.fit_transform([self.preprocess(t1), self.preprocess(t2)])
            return cosine_similarity(tfidf_matrix)[0][1]
        except:
            return 0

    def detect_similarities(self, projects, name, desc, threshold=0.6):
        results, desc = [], self.preprocess(desc)
        for p in projects:
            sim = np.sqrt(self.similarity(name, p["name"]) * self.similarity(desc, self.preprocess(p["description"])))
            if sim >= threshold:
                results.append({**p, "similarity_score": round(sim * 100, 2)})
        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:3]

detector = SimilarityDetector()

def fetch_projects():
    with engine.connect() as conn:
        books = conn.execute(text("SELECT id, name, description FROM book")).fetchall()
        pre_projects = conn.execute(text("SELECT id, name, description FROM pre_project")).fetchall()
    return [{"id": str(r.id), "name": r.name, "description": r.description, "source": t} 
            for t, rows in [("book", books), ("pre_project", pre_projects)] for r in rows]

@app.route("/detect_similarities", methods=["POST"])
def check_similarity():
    data = request.get_json()
    projects = fetch_projects()
    similar = detector.detect_similarities(projects, data.get("project_name", ""), data.get("project_description", ""), float(data.get("similarity_threshold", 50)) / 100)
    return jsonify({"error": "Similar projects found", "similar_projects": similar, "status": "failure"} if similar else {"message": "No similar projects found", "status": "success"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
