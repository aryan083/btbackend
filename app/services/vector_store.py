"""
Vector store service using ChromaDB for document storage and retrieval
"""
import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store service for managing document embeddings and retrieval
    """
    def __init__(self, persist_dir: str):
        """
        Initialize the vector store
        @param persist_dir: str - Directory to persist ChromaDB data
        @returns: None
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    def create_collection(self, collection_name: str) -> Any:
        """
        Create or get a ChromaDB collection
        @param collection_name: str - Name of the collection
        @returns: Any - ChromaDB collection instance
        """
        try:
            return self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except ValueError:
            # Collection exists, get it
            return self.client.get_collection(collection_name)

    def add_documents(self, collection_name: str, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector store
        @param collection_name: str - Name of the collection
        @param documents: List[Dict[str, Any]] - List of documents to add
        @returns: None
        """
        try:
            collection = self.create_collection(collection_name)
            
            # Prepare documents for insertion
            ids = [str(i) for i in range(len(documents))]
            texts = [doc['text'] for doc in documents]
            metadatas = [
                {
                    'page_num': doc.get('page_num', 0),
                    'type': doc.get('type', 'text'),
                    'source': doc.get('source', ''),
                    'title': doc.get('title', '')
                }
                for doc in documents
            ]
            
            # Add documents to collection
            collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            raise

    def query_documents(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        Query documents from the vector store
        @param collection_name: str - Name of the collection
        @param query_text: str - Query text
        @param n_results: int - Number of results to return
        @param where: Dict - Filter conditions
        @returns: List[Dict[str, Any]] - List of matching documents
        """
        try:
            collection = self.client.get_collection(collection_name)
            
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i]
                })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to query vector store: {str(e)}")
            raise

    def delete_collection(self, collection_name: str):
        """
        Delete a collection from the vector store
        @param collection_name: str - Name of the collection to delete
        @returns: None
        """
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            raise
