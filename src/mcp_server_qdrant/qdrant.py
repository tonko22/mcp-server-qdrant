"""
Connector for the Qdrant vector database.
"""

from typing import Optional, Dict, List, Any
from uuid import uuid4
from datetime import datetime

from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.qdrant_fastembed import QdrantFastembedMixin


class QdrantConnector(QdrantClient, QdrantFastembedMixin):
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
        
        # Инициализируем родительский класс QdrantClient
        QdrantClient.__init__(self, url=qdrant_url, api_key=qdrant_api_key)
        # Инициализируем FastEmbed
        QdrantFastembedMixin.__init__(self)
        
        logger.info(f"Initializing QdrantConnector with model: {fastembed_model_name}")
        try:
            logger.debug("Setting FastEmbed model")
            self.set_model(fastembed_model_name)
            logger.info("FastEmbed model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed model: {str(e)}", exc_info=True)
            raise

        # Получаем имя поля вектора
        self._vector_field_name = self.get_vector_field_name()
        logger.debug(f"Using vector field name: {self._vector_field_name}")

    async def find_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        logger.debug(f"Searching memories with query: {query}")
        try:
            # Выполняем поиск через query, который сам сгенерирует эмбеддинги
            logger.debug(f"Calling query with collection_name={self._collection_name}, query_text={query}, limit={limit}")
            try:
                # Проверяем, что модель инициализирована
                model = self._get_or_init_model(model_name=self.embedding_model_name)
                logger.debug(f"Using model: {model}")

                # Проверяем, что коллекция существует
                if not self.collection_exists(self._collection_name):
                    raise ValueError(f"Collection {self._collection_name} does not exist")

                search_result = self.query(
                    collection_name=self._collection_name,
                    query_text=query,
                    limit=limit
                )
                logger.debug(f"Query completed successfully, result type: {type(search_result)}")
            except AttributeError as ae:
                logger.error(f"Attribute error during query: {str(ae)}", exc_info=True)
                raise ValueError(f"Model initialization error: {str(ae)}")
            except ValueError as ve:
                logger.error(f"Value error during query: {str(ve)}", exc_info=True)
                raise
            except Exception as query_error:
                logger.error(f"Error during query execution: {str(query_error)}, type: {type(query_error)}", exc_info=True)
                raise ValueError(f"Query execution error: {str(query_error)}")

            logger.debug(f"Search completed, found {len(search_result)} results")
            logger.debug(f"First result example: {search_result[0] if search_result else 'No results'}")

            # Получаем текст из metadata
            result = [{"score": item.score, "text": item.metadata["text"]} for item in search_result]
            return result

        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}, type: {type(e)}", exc_info=True)
            raise ValueError(f"Memory search error: {str(e)}")

    async def add_message(self, message_text: str, user_id: int, username: str, message_id: int, date: datetime):
        """Add a message to the vector database."""
        logger.debug(f"Adding message from {username}: {message_text[:50]}...")
        embeddings = list(self.embed([message_text]))
        
        point = models.PointStruct(
            id=uuid4().hex,
            vector={self._vector_field_name: embeddings[0].tolist()},
            payload={
                "text": message_text,
                "user_id": user_id,
                "username": username,
                "message_id": message_id,
                "date": date.isoformat()
            }
        )

        await self.upsert(
            collection_name=self._collection_name,
            points=[point]
        )

    async def store_memory(self, information: str, metadata: Dict[str, Any] = None) -> None:
        """Store a memory in the vector database."""
        logger.debug(f"Storing memory: {information[:100]}...")
        try:
            embeddings = list(self.embed([information]))
            logger.debug(f"Generated embeddings: shape={len(embeddings)}x{len(embeddings[0])}")
            
            payload = {"text": information}
            if metadata:
                payload.update(metadata)

            point = models.PointStruct(
                id=uuid4().hex,
                vector={self._vector_field_name: embeddings[0].tolist()},
                payload=payload
            )

            await self.upsert(
                collection_name=self._collection_name,
                points=[point]
            )

        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}", exc_info=True)
            raise
