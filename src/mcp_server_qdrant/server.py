"""
Server implementation for the Qdrant memory store.
"""

from typing import Optional, Dict, Any

import click
from loguru import logger
from mcp.server import Server, types, NotificationOptions
from mcp.server.models import InitializationOptions
import asyncio
import mcp

from .qdrant import QdrantConnector


async def serve(
    qdrant_url: str,
    collection_name: str,
    fastembed_model_name: str,
    qdrant_api_key: Optional[str] = None,
    qdrant_local_path: Optional[str] = None,
) -> Server:
    """
    Instantiate the server and configure tools to store and find memories in Qdrant.
    """
    logger.debug(f"Creating QdrantConnector with url={qdrant_url}, collection={collection_name}, model={fastembed_model_name}")
    try:
        qdrant = QdrantConnector(
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            fastembed_model_name=fastembed_model_name,
            qdrant_api_key=qdrant_api_key,
            qdrant_local_path=qdrant_local_path,
        )
        logger.debug("QdrantConnector created successfully")
    except Exception as e:
        logger.error(f"Failed to create QdrantConnector: {str(e)}", exc_info=True)
        raise

    server = Server("qdrant")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """Return the list of tools that the server provides."""
        logger.debug("Handling list_tools request")
        tools = [
            types.Tool(
                name="qdrant-store-memory",
                description=(
                    "Keep the memory for later use, when you are asked to remember something."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "information": {
                            "type": "string",
                        },
                    },
                    "required": ["information"],
                },
            ),
            types.Tool(
                name="qdrant-find-memories",
                description=(
                    "Look up memories in Qdrant. Use this tool when you need to: \n"
                    " - Find memories by their content \n"
                    " - Access memories for further analysis \n"
                    " - Get some personal information about the user"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for in the memories",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]
        logger.debug(f"Returning tools: {[tool.name for tool in tools]}")
        return tools

    @server.call_tool()
    async def handle_tool_call(
        name: str, arguments: Optional[Dict[str, Any]]
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        logger.debug(f"Tool call received: {name} with arguments: {arguments}")
        
        if name not in ["qdrant-store-memory", "qdrant-find-memories"]:
            logger.error(f"Unknown tool: {name}")
            raise ValueError(f"Unknown tool: {name}")

        try:
            if name == "qdrant-store-memory":
                if not arguments or "information" not in arguments:
                    logger.error("Missing required argument 'information'")
                    raise ValueError("Missing required argument 'information'")
                information = arguments["information"]
                logger.info(f"Storing memory: {information}")
                await qdrant.store_memory(information)
                return [types.TextContent(type="text", text=f"Remembered: {information}")]

            if name == "qdrant-find-memories":
                if not arguments or "query" not in arguments:
                    logger.error("Missing required argument 'query'")
                    raise ValueError("Missing required argument 'query'")
                query = arguments["query"]
                logger.info(f"Searching memories with query: {query}")
                try:
                    memories = await qdrant.find_memories(query)
                    logger.debug(f"Found {len(memories)} memories")
                    
                    # Добавляем детальное логирование
                    logger.debug("Found memories content:")
                    for i, memory in enumerate(memories):
                        logger.debug(f"Memory {i + 1}: {memory}")
                    
                    content = [
                        types.TextContent(type="text", text=f"Memories for the query '{query}'")
                    ]
                    content.extend(
                        types.TextContent(type="text", text=memory["text"]) for memory in memories
                    )
                    
                    # Логируем финальный ответ
                    logger.debug(f"Returning content: {content}")
                    return content
                except Exception as search_error:
                    logger.error(f"Error during search: {str(search_error)}", exc_info=True)
                    raise
        except Exception as e:
            logger.error(f"Error handling tool call: {str(e)}", exc_info=True)
            raise ValueError(str(e))

    return server


@click.command()
@click.option(
    "--qdrant-url",
    envvar="QDRANT_URL",
    required=True,
    help="URL of the Qdrant server",
)
@click.option(
    "--qdrant-api-key",
    envvar="QDRANT_API_KEY",
    help="API key for the Qdrant server",
)
@click.option(
    "--collection-name",
    envvar="COLLECTION_NAME",
    default="telegram_messages",
    help="Name of the collection to use",
)
@click.option(
    "--fastembed-model-name",
    envvar="FASTEMBED_MODEL_NAME",
    default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    help="Name of the FastEmbed model to use",
)
@click.option(
    "--qdrant-local-path",
    envvar="QDRANT_LOCAL_PATH",
    help="Path to the storage directory for the Qdrant client, if local mode is used",
)
def main(
    qdrant_url: str,
    qdrant_api_key: Optional[str],
    collection_name: str,
    fastembed_model_name: str,
    qdrant_local_path: Optional[str],
) -> None:
    """Run the server."""

    async def _run() -> None:
        logger.debug("Starting server")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.debug("Creating Qdrant server instance")
            try:
                server = await serve(
                    qdrant_url=qdrant_url,
                    collection_name=collection_name,
                    fastembed_model_name=fastembed_model_name,
                    qdrant_api_key=qdrant_api_key,
                    qdrant_local_path=qdrant_local_path,
                )
                logger.debug("Qdrant server instance created")
            except Exception as e:
                logger.error(f"Failed to create Qdrant server instance: {str(e)}", exc_info=True)
                raise

            logger.debug("Starting server run loop")
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="qdrant",
                    server_version="0.5.1",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    try:
        asyncio.run(_run())
    except Exception as e:
        logger.error(f"Failed to create or run stdio server: {str(e)}", exc_info=True)
        raise
