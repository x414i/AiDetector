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

# Comprehensive Arabic stopwords list
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
        
        for project in projects:
            name_sim = self.calculate_similarity(new_name, project['name'])
            desc_sim = self.calculate_similarity(new_desc, project['description'])
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

def evaluate_similarity_detection(detector, project_database, test_cases):
    """
    Comprehensive evaluation of similarity detection
    """
    all_true_labels = []
    all_predicted_labels = []
    detailed_results = []

    for case in test_cases:
        # Detect similarities
        similar_projects = detector.detect_similarities(
            project_database,
            case['new_project_name'], 
            case['new_project_description'], 
            threshold=case.get('threshold', 0.30)
        )

        # Prepare true and predicted labels
        true_label = case['expected_similar']
        predicted_label = 1 if similar_projects else 0

        all_true_labels.append(true_label)
        all_predicted_labels.append(predicted_label)

        # Store detailed result
        detailed_results.append({
            'new_project_name': case['new_project_name'],
            'new_project_description': case['new_project_description'],
            'similar_projects': similar_projects,
            'true_label': true_label,
            'predicted_label': predicted_label
        })

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, 
        all_predicted_labels, 
        average='binary',
        zero_division=0 )

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'detailed_results': detailed_results
    }

def generate_comprehensive_test_cases():
    """
    Generate a comprehensive set of test cases
    """
    return [
        {
            'new_project_name': 'الذكاء الاصطناعي في الطب',
            'new_project_description': 'استغلال تقنيات الذكاء الاصطناعي لتحسين الممارسات الطبية.',
            'expected_similar': 1,  # Should find similar projects
            'threshold': 0.30
        },
        {
            'new_project_name': 'Machine Learning in Finance',
            'new_project_description': 'Advanced predictive models for financial analysis',
            'expected_similar': 0,  # Should not find similar projects
            'threshold': 0.30
        },
        {
            'new_project_name': 'معالجة اللغات المتقدمة',
            'new_project_description': 'تطوير نظام متقدم لفهم وتحليل اللغات المختلفة',
            'expected_similar': 1,  # Should find similar NLP projects
            'threshold': 0.30
        }
    ]
def main():
    # Initialize project database and detector
    project_database = ProjectDatabase()
    detector = AdvancedSimilarityDetector()
    
    # Adding diverse projects in Arabic and English
    project_database.add_project(Project(
        "الذكاء الاصطناعي في الرعاية الصحية", 
        "استخدام الذكاء الاصطناعي لتحسين نتائج الرعاية الصحية والتشخيص الطبي", 
        ["تحليل البيانات", "التعلم الآلي"], 
        domain="Healthcare"
    ))
    
    project_database.add_project(Project(
        "AI in Medical Diagnostics", 
        "Leveraging artificial intelligence to enhance medical diagnosis accuracy and patient outcomes", 
        ["Data Analysis", "Machine Learning"], 
        domain="Healthcare"
    ))
    
    project_database.add_project(Project(
        "معالجة اللغة الطبيعية", 
        "تقنيات متقدمة لفهم وتحليل اللغات البشرية باستخدام الذكاء الاصطناعي", 
        ["تحليل النصوص", "التعلم العميق"], 
        domain="NLP"
    ))
    
    project_database.add_project(Project(
        "Natural Language Processing Advanced", 
        "Cutting-edge techniques for understanding and analyzing human languages using AI", 
        ["Text Analysis", "Deep Learning"], 
        domain="NLP"
    ))

    # Test cases for cross-language similarity
    cross_language_test_cases = [
        {
            'new_project_name': 'الذكاء الاصطناعي في التشخيص الطبي',
            'new_project_description': 'تطبيق الذكاء الاصطناعي المتقدم لتحسين دقة التشخيص الطبي',
            'expected_similar': 1,
            'threshold': 0.30
        },
        {
            'new_project_name': 'Advanced NLP Techniques',
            'new_project_description': 'Innovative approaches to understanding complex language structures',
            'expected_similar': 1,
            'threshold': 0.30
        }
    ]

    # Perform evaluation
    evaluation_results = evaluate_similarity_detection(
        detector, 
        project_database, 
        cross_language_test_cases
    )
    
    # Print detailed results
    print("\n--- Cross-Language Similarity Detection Evaluation ---")
    print(f"Precision: {evaluation_results['precision']:.2f}")
    print(f"Recall: {evaluation_results['recall']:.2f}")
    print(f"F1 Score: {evaluation_results['f1_score']:.2f}")
    
    print("\nDetailed Results:")
    for result in evaluation_results['detailed_results']:
        print("\nTest Case:")
        print(f"Project Name: {result['new_project_name']}")
        print(f"Project Description: {result['new_project_description']}")
        print(f"True Label: {result['true_label']}")
        print(f"Predicted Label: {result['predicted_label']}")
        print("Similar Projects:")
        for proj in result['similar_projects']:
            print(f"- {proj['name']} (Similarity: {proj['similarity_score']:.2f}%)")

if __name__ == '__main__':
    main()