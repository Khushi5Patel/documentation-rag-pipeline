import os
from urllib.parse import parse_qs, urlparse
from http.server import SimpleHTTPRequestHandler, HTTPServer

from langchain_community.document_loaders import (
    TextLoader, CSVLoader, UnstructuredHTMLLoader, PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

SOURCE_DIR = "source_documents"
DB_DIR = "chroma_db_storage"

if not os.path.exists(SOURCE_DIR):
    os.makedirs(SOURCE_DIR)

def run_document_ingestion():
    SUPPORTED_FORMATS = {
        ".txt": TextLoader, ".csv": CSVLoader, 
        ".html": UnstructuredHTMLLoader, ".pdf": PyPDFLoader
    }
    
    files = os.listdir(SOURCE_DIR)
    if not files:
        print("No documents found inside 'source_documents' folder yet.")
        return None
        
    all_docs = []
    for file_name in files:
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()
        if ext in SUPPORTED_FORMATS:
            try:
                loader = SUPPORTED_FORMATS[ext](os.path.join(SOURCE_DIR, file_name))
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_docs)
    
    embedding_engine = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
        documents=chunks, embedding=embedding_engine, persist_directory=DB_DIR
    )
    print("✅ Local Vector Database built successfully.")
    return vector_db

print("Initializing local semantic embedding engine...")
embedding_engine = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Local search engine active!")

run_document_ingestion()

def extract_answer_locally(context, query):
    """
    Scans the retrieved paragraph block and extracts the specific sentences 
    that match the core keywords of your question to form a perfect answer.
    """
    sentences = context.split('.')
    query_words = [w.lower() for w in query.split() if len(w) > 3]
    
    best_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        matches = sum(1 for word in query_words if word in sentence_lower)
        if matches > 0:
            best_sentences.append(sentence.strip())
            
    if best_sentences:
        return ". ".join(best_sentences[:2]) + "."
    return context[:300] + "..."


class WebAppHandler(SimpleHTTPRequestHandler):
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.path = '/index.html'
            return SimpleHTTPRequestHandler.do_GET(self)
            
        if self.path.startswith('/search'):
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)
            user_query = query_params.get('q', [''])[0]
            
            vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_engine)
            results = vector_db.similarity_search(user_query, k=1)
            
            if results:
                retrieved_fact = results[0].page_content
                source_file = os.path.basename(results[0].metadata.get('source', 'Unknown'))
                
                final_answer = extract_answer_locally(retrieved_fact, user_query)
                
                formatted_response = f"AI Summary Response:\n{final_answer}\n\n[CITED SOURCE]: Document '{source_file}'"
            else:
                formatted_response = "I couldn't find any relevant entries in the database records."
            
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(formatted_response.encode('utf-8'))

print("\nOpen http://localhost:8000 in your browser tab")
server = HTTPServer(('localhost', 8000), WebAppHandler)
server.serve_forever()
