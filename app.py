import os
import re
import numpy as np
import logging
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import pyarabic.araby as araby

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARABIC_STOPWORDS = set([
    'في', 'من', 'إلى', 'على', 'عن', 'أن', 'إن', 'إذا', 'ذلك', 'هذا',
    'هذه', 'كان', 'يكون', 'ما', 'مع', 'هنا', 'هناك', 'قد', 'لا', 'لم',
    'لكن', 'و', 'أو', 'بعد', 'قبل', 'حتى', 'حيث', 'هو', 'هي', 'هم',
    'أنت', 'أنا', 'نحن', 'إذ', 'إن', 'إلا', 'ليس', 'ليست', 'لن', 'لك',
    'له', 'لها', 'بين', 'بعض', 'كل', 'كذلك', 'كما', 'عند', 'عندما',
    'عليه', 'فقط', 'قد', 'كيف', 'لا', 'لما', 'لماذا', 'ما', 'متى',
    'ماذا', 'مع', 'مما', 'منذ', 'نحن', 'هو', 'هي', 'هل', 'و', 'أيضا',
    'إلى', 'أمام', 'أي', 'أين', 'أن', 'إن', 'أول', 'أيها', 'إنه', 'إيه',
    'أكثر', 'أقل', 'أكبر', 'أصغر', 'أجل', 'أبدا', 'إذن', 'الآن', 'التي',
    'الذي', 'الذين', 'اللاتي', 'اللائي', 'اللواتي'
])

class AdvancedMultilingualPreprocessor:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        self.english_stopwords = set(stopwords.words('english'))
        self.arabic_stopwords = ARABIC_STOPWORDS

    def detect_language(self, text):
        return 'ar' if re.search(r'[\u0600-\u06FF]', text) else 'en'

    def preprocess_arabic(self, text):
        text = araby.strip_tatweel(text)
        text = araby.normalize_hamza(text)
        text = araby.normalize_ligature(text)
        text = araby.normalize_alef(text)
        return re.sub(r'[^\u0600-\u06FF\s]', '', text)

    def preprocess_text(self, text):
        lang = self.detect_language(text)
        text = text.lower().strip()
        
        if lang == 'ar':
            text = self.preprocess_arabic(text)
            tokens = araby.tokenize(text)
            tokens = [t for t in tokens if t not in self.arabic_stopwords and len(t) > 1]
        else:
            text = re.sub(r'[^a-z\s]', '', text)
            tokens = text.split()
            tokens = [t for t in tokens if t not in self.english_stopwords and len(t) > 2]
        
        return ' '.join(tokens), lang

class HighAccuracySimilarityDetector:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.tfidf_vectorizer = TfidfVectorizer()
        self.preprocessor = AdvancedMultilingualPreprocessor()

    def get_embedding(self, text):
        preprocessed_text, _ = self.preprocessor.preprocess_text(text)
        return self.model.encode(preprocessed_text, normalize_embeddings=True)

    def calculate_similarity(self, text1, text2):
        # Semantic similarity
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        semantic_sim = cosine_similarity([emb1], [emb2])[0][0]
        
        # Lexical similarity
        preprocessed1, _ = self.preprocessor.preprocess_text(text1)
        preprocessed2, _ = self.preprocessor.preprocess_text(text2)
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([preprocessed1, preprocessed2])
            tfidf_sim = cosine_similarity(tfidf_matrix)[0][1]
        except:
            tfidf_sim = 0

        # Combined score
        return (semantic_sim * 0.7) + (tfidf_sim * 0.3)

    def detect_similarities(self, projects, new_name, new_desc, threshold=0.6):
        results = []
        
        new_desc_processed = preprocess_description(new_desc)

        for project in projects:
            existing_desc_processed = preprocess_description(project['description'])

            name_sim = self.calculate_similarity(new_name, project['name'])
            desc_sim = self.calculate_similarity(new_desc_processed, existing_desc_processed)
            combined_sim = np.sqrt(name_sim * desc_sim)
            
            if combined_sim >= threshold:
                results.append({
                    'project_id': project['id'],
                    'name': project['name'],
                    'description': project['description'],
                    'similarity_score': round(combined_sim * 100, 2),
                    'source_table': project['source_table']
                })
        
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)

app = Flask(__name__)
detector = HighAccuracySimilarityDetector()

# Database connection URL
DATABASE_URL = "postgresql://postgres:091093Aa@localhost:5432/major?sslmode=disable"

engine = create_engine(DATABASE_URL)
def preprocess_description(description):
    # Split the description into sections
    sections = description.split("النظام المقترح:")
    if len(sections) > 1:
        return sections[1].strip()  
    return description.strip() 

def fetch_projects_from_db():
    with engine.connect() as connection:
        books = connection.execute(text("SELECT id, name, description FROM book")).fetchall()
        pre_projects = connection.execute(text("SELECT id, name, description FROM pre_project")).fetchall()

    projects = []
    for row in books:
        projects.append({
            "id": str(row.id),
            "name": row.name,
            "description": row.description,
            "source_table": "book"
        })

    for row in pre_projects:
        projects.append({
            "id": str(row.id),
            "name": row.name,
            "description": row.description,
            "source_table": "pre_project"
        })

    return projects

@app.route("/detect_similarities", methods=["POST"])
def handle_similarity_check():
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")  
        project_name = data.get("project_name", "").strip()
        project_description = data.get("project_description", "").strip()
        threshold = float(data.get("similarity_threshold", 50)) / 100

        if not project_name or not project_description:
            return jsonify({"error": "Project name and description are required"}), 400

        projects = fetch_projects_from_db()
        similar_projects = detector.detect_similarities(
            projects, project_name, project_description, threshold
        )
        logger.info(f"Similar projects found: {similar_projects}")

        # Process similar projects to only show "نظام المقترح" if it exists
        for project in similar_projects:
            project['description'] = preprocess_description(project['description'])

        if similar_projects:
            return jsonify({
                "error": "Similar projects found",
                "similar_projects": similar_projects[:3],
                "status": "failure"
            }), 409

        return jsonify({
            "message": "No similar projects found",
            "status": "success"
        }), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
# Test endpoint
@app.route("/test", methods=["GET"])
def test_endpoint():
    test_cases = [
        ("مشروع تعليمي", "مشروع حول تحسين التعليم باستخدام الذكاء الاصطناعي"),
        ("AI Education Project", "A project about improving education using AI")
    ]
    
    results = []
    for name, desc in test_cases:
        similar = detector.detect_similarities(fetch_projects_from_db(), name, desc)
        results.append({
            "input_name": name,
            "input_description": desc,
            "matches": similar[:2]
        })
    
    return jsonify({"tests": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Change port to 8080