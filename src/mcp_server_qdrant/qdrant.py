"""
Connector for the Qdrant vector database.
"""

from typing import Optional, Dict, List, Any
from uuid import uuid4
from datetime import datetime

from loguru import logger
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding


class QdrantConnector:
    """
    A connector for the Qdrant vector database.
    """

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        fastembed_model_name: str,
        qdrant_api_key: Optional[str] = None,
        qdrant_local_path: Optional[str] = None,
    ):
        """Initialize instance variables and create connections."""
        self._collection_name = collection_name
        self.embedding_model = TextEmbedding(fastembed_model_name)
        # Get vector size from model
        self.vector_size = len(next(self.embedding_model.embed(["test"])))
        
        # Инициализируем Qdrant клиент
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        logger.info(f"Initializing QdrantConnector with model: {fastembed_model_name}")
        
        # Создаем коллекцию если её нет
        self._ensure_collection_exists()
        
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        if not self.client.collection_exists(self._collection_name):
            logger.info(f"Creating collection {self._collection_name}")
            
            self.client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                )
            )
            logger.info(f"Collection {self._collection_name} created successfully")

    def get_embedding(self, text: str) -> list:
        """Get embedding for text using FastEmbed."""
        embeddings = list(self.embedding_model.embed([text]))
        return embeddings[0].tolist()

    def find_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for memories by semantic similarity."""
        logger.debug(f"Searching memories with query: {query}")
        try:
            # Получаем эмбеддинги для запроса
            query_vector = self.get_embedding(query)

            # Выполняем поиск
            search_result = self.client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=0.0
            )

            # Преобразуем результаты в нужный формат
            result = []
            for item in search_result:
                result_item = {
                    "id": item.id,
                    "score": item.score,
                    "metadata": item.payload
                }
                result.append(result_item)
            return result

        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}", exc_info=True)
            raise ValueError(f"Memory search error: {str(e)}")

    def add_message(self, message_text: str, user_id: int, username: str, message_id: int, date: datetime):
        """Add a message to the vector database."""
        logger.debug(f"Adding message from {username}: {message_text[:50]}...")
        vector = self.get_embedding(message_text)
        
        point = models.PointStruct(
            id=uuid4().hex,
            vector=vector,
            payload={
                "text": message_text,
                "user_id": user_id,
                "username": username,
                "message_id": message_id,
                "date": date.isoformat()
            }
        )
        
        self.client.upsert(
            collection_name=self._collection_name,
            points=[point]
        )

    def store_memory(self, information: str, metadata: Optional[Dict[str, Any]] = None):
        """Store a memory in the vector database."""
        logger.debug(f"Storing memory: {information[:100]}...")
        try:
            vector = self.get_embedding(information)
            
            payload = {"text": information}
            if metadata:
                payload.update(metadata)

            point = models.PointStruct(
                id=uuid4().hex,
                vector=vector,
                payload=payload
            )

            self.client.upsert(
                collection_name=self._collection_name,
                points=[point]
            )
            
            return {"status": "success", "message": "Memory stored successfully"}
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}", exc_info=True)
            raise ValueError(f"Memory storage error: {str(e)}")
