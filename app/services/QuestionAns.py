import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import logging
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from app.services.article_service import ArticleService
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize NLP tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Initialize lightweight embedding model (90MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

logger = logging.getLogger(__name__)

def preprocess_text(text):
    """Comprehensive text cleaning pipeline"""
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Text cleaning
    cleaned = []
    for token in tokens:
        # Remove special characters and numbers
        token = re.sub(r'[^a-zA-Z]', '', token)
        if len(token) < 2:
            continue
            
        # Stopword removal
        if token in stop_words:
            continue
            
        # Stemming
        cleaned.append(stemmer.stem(token))
    
    return cleaned

def clean_html(html_text):
    """Remove HTML tags and clean the text"""
    if not html_text:
        return ""
        
    # Check if the text contains HTML tags
    if '<' in html_text and '>' in html_text:
        # Use BeautifulSoup to parse and extract text
        soup = BeautifulSoup(html_text, 'html.parser')
        # Get text and preserve some formatting with spaces
        clean_text = soup.get_text(separator=' ', strip=True)
    else:
        clean_text = html_text
        
    # Additional cleaning
    clean_text = re.sub(r'\s+', ' ', clean_text)  # Replace multiple spaces with single space
    clean_text = clean_text.strip()
    
    return clean_text

def process_article(article_text):
    """Enhanced article processing with NLP pipeline"""
    # Clean HTML tags if present
    clean_text = clean_html(article_text)
    
    chunks = []
    current_section = []
    
    # Structural processing
    for line in clean_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Section detection (titles, headers)
        if re.match(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)*(?: - \d+\.\d+)?$', line):
            if current_section:
                chunks.append(' '.join(current_section))
                current_section = []
            continue
                
        # Content chunking with sentences
        sentences = sent_tokenize(line)
        for sent in sentences:
            if len(current_section) < 3:
                current_section.append(sent)
            else:
                chunks.append(' '.join(current_section))
                current_section = [sent]
    
    # Add the last chunk if it exists
    if current_section:
        chunks.append(' '.join(current_section))
    
    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk.split()) > 3]
    
    # Create search corpus with full preprocessing
    return {
        'original': chunks,
        'processed': [preprocess_text(chunk) for chunk in chunks]
    }

def find_answer(question, article_text):
    """Enhanced QA with full text normalization and BM25"""
    # Process article fresh for each query
    article_data = process_article(article_text)
    
    if not article_data['original']:
        return "No relevant information found in the article."
    
    # Process question
    query_terms = preprocess_text(question)
    
    # BM25 search
    bm25 = BM25Okapi(article_data['processed'])
    bm25_scores = bm25.get_scores(query_terms)
    
    # Semantic search with embeddings
    question_embed = model.encode([question])
    chunk_embeds = model.encode(article_data['original'])
    semantic_scores = np.dot(chunk_embeds, question_embed.T).flatten()
    
    # Combine scores with normalization
    combined = []
    # Check if scores are not empty before finding max
    max_bm25 = max(bm25_scores) if bm25_scores.size > 0 and max(bm25_scores) > 0 else 1
    max_semantic = max(semantic_scores) if semantic_scores.size > 0 and max(semantic_scores) > 0 else 1
    
    for b, s in zip(bm25_scores, semantic_scores):
        combined.append(
            (0.6 * (b/max_bm25)) + (0.4 * (s/max_semantic))
        )
    
    # Get best match
    if not combined:
        return "Could not find a relevant answer."
    
    best_idx = np.argmax(combined)
    return article_data['original'][best_idx]

class QuestionAnswerService:
    def __init__(self):
        self.article_service = ArticleService()
        self._article_content = None
        self._current_article_id = None
        
    def set_article_content_by_id(self, article_id):
        """Set the article content using article_id"""
        try:
            article_content = self.article_service.get_article_content(article_id)
            if article_content:
                # Clean the article content from HTML if present
                clean_content = clean_html(article_content)
                self._article_content = clean_content
                self._current_article_id = article_id
                
                # Console log the article content for debugging
                print(f"Article content loaded for ID {article_id}:")
                print(f"Length: {len(clean_content)} characters")
                print(f"Preview: {clean_content[:200]}...")
                
                # Test chunking on the content
                article_data = process_article(clean_content)
                print(f"Created {len(article_data['original'])} chunks from article content")
                print(f"Preprocessed text with stemming and stopword removal")
                
                return True
            else:
                logger.error(f"No content found for article ID: {article_id}")
                return False
        except Exception as e:
            logger.error(f"Error setting article content: {str(e)}")
            return False
            
    def get_article_content(self):
        """Get the current article content"""
        return self._article_content
        
    def get_current_article_id(self):
        """Get the current article ID"""
        return self._current_article_id

# Sample test code to demonstrate functionality
def test_question_answer(article_id, question):
    qa_service = QuestionAnswerService()
    if qa_service.set_article_content_by_id(article_id):
        article_content = qa_service.get_article_content()
        answer = find_answer(question, article_content)
        return {
            "question": question,
            "answer": answer,
            "article_id": article_id
        }
    else:
        return {"error": f"Could not load article content for ID: {article_id}"}

# Example usage (commented out for production use)
"""
if __name__ == "__main__":
    # Example article ID - replace with actual ID from your database
    test_article_id = "92e7a89e-dd6c-49b9-ae86-83d6ded04f4e"
    
    # Test with a sample question
    test_question = "what is a stack?"
    
    result = test_question_answer(test_article_id, test_question)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
"""