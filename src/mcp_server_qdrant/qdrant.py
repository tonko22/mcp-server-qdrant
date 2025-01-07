from typing import Optional
from loguru import logger
from qdrant_client import AsyncQdrantClient, models
import fastembed

class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the collection to use.
    :param fastembed_model_name: The name of the FastEmbed model to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        collection_name: str,
        fastembed_model_name: str,
        qdrant_local_path: Optional[str] = None,
    ):
        logger.debug(f"Initializing QdrantConnector with collection: {collection_name}, model: {fastembed_model_name}")
        
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._collection_name = collection_name
        self._fastembed_model_name = fastembed_model_name

        logger.debug(f"Creating AsyncQdrantClient with: url={self._qdrant_url}, path={qdrant_local_path}")
        self._client = AsyncQdrantClient(location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path)
        
        logger.debug(f"Setting up FastEmbed model: {fastembed_model_name}")
        self._model = fastembed.TextEmbedding(model_name=fastembed_model_name)
        logger.debug("FastEmbed model setup successful")
        
        logger.info(f"QdrantConnector initialized with collection: {self._collection_name}, model: {self._fastembed_model_name}")

    async def store_memory(self, information: str):
        """
        Store a memory in the Qdrant collection.
        :param information: The information to store.
        """
        logger.debug(f"Storing memory: {information[:100]}...")
        try:
            # Получаем эмбеддинги для текста
            embeddings = list(self._model.embed([information]))
            logger.debug(f"Generated embeddings: shape={len(embeddings)}x{len(embeddings[0])}")
            
            # Сохраняем в Qdrant
            await self._client.upsert(
                collection_name=self._collection_name,
                points=[
                    models.PointStruct(
                        id=hash(information),  # TODO: использовать более надежный способ генерации ID
                        vector=embeddings[0].tolist(),
                        payload={"text": information}
                    )
                ]
            )
            logger.info("Memory stored successfully")
        except Exception as e:
            logger.error(f"Error storing memory: {str(e)}")
            raise

    async def find_memories(self, query: str) -> list[str]:
        """
        Find memories in the Qdrant collection. If there are no memories found, an empty list is returned.
        :param query: The query to use for the search.
        :return: A list of memories found.
        """
        logger.debug(f"Searching memories with query: {query}")
        
        try:
            # Получаем эмбеддинги для запроса
            query_embeddings = list(self._model.embed([query]))
            logger.debug(f"Generated embeddings: shape={len(query_embeddings)}x{len(query_embeddings[0])}")
            
            # Ищем похожие документы
            search_result = await self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embeddings[0].tolist(),
                limit=10
            )
            logger.debug(f"Raw search result: {search_result}")
            
            # Извлекаем тексты из результатов
            memories = []
            for hit in search_result:
                logger.debug(f"Hit score: {hit.score}, payload: {hit.payload}")
                if hit.payload and "text" in hit.payload:
                    memories.append(hit.payload["text"])
            
            logger.info(f"Found {len(memories)} memories")
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return []
