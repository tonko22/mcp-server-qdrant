from typing import Optional
import os
import sys

from loguru import logger
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions

import click
import mcp.types as types
import asyncio
import mcp

from .qdrant import QdrantConnector


def serve(
    qdrant_url: Optional[str],
    qdrant_api_key: str,
    collection_name: Optional[str],
    fastembed_model_name: str,
    qdrant_local_path: Optional[str],
) -> Server:
    """
    Instantiate the server and configure tools to store and find memories in Qdrant.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the collection to use.
    :param fastembed_model_name: The name of the FastEmbed model to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """
    try:
        logger.debug(f"Creating QdrantConnector with url={qdrant_url}, collection={collection_name}, model={fastembed_model_name}")
        qdrant = QdrantConnector(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=collection_name or "telegram_messages",
            fastembed_model_name=fastembed_model_name,
            qdrant_local_path=qdrant_local_path,
        )
    except Exception as e:  
        logger.error(f"Failed to create QdrantConnector: {str(e)}", exc_info=True)
        raise

    server = Server("qdrant")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """
        Return the list of tools that the server provides. By default, there are two
        tools: one to store memories and another to find them. Finding the memories is not
        implemented as a resource, as it requires a query to be passed and resources point
        to a very specific piece of data.
        """
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
        name: str, arguments: dict | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        logger.debug(f"Tool call received: {name} with arguments: {arguments}")
        
        if name not in ["qdrant-store-memory", "qdrant-find-memories"]:
            error_msg = f"Unknown tool: {name}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            if name == "qdrant-store-memory":
                if not arguments or "information" not in arguments:
                    error_msg = "Missing required argument 'information'"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                information = arguments["information"]
                logger.info(f"Storing memory: {information}")
                await qdrant.store_memory(information)
                return [types.TextContent(type="text", text=f"Remembered: {information}")]

            if name == "qdrant-find-memories":
                if not arguments or "query" not in arguments:
                    error_msg = "Missing required argument 'query'"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                query = arguments["query"]
                logger.info(f"Searching memories with query: {query}")
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
                    types.TextContent(type="text", text=memory) for memory in memories
                )
                
                # Логируем финальный ответ
                logger.debug(f"Returning content: {content}")
                return content
        except Exception as e:
            logger.error(f"Error handling tool call: {str(e)}", exc_info=True)
            raise

    return server


@click.command()
@click.option(
    "--qdrant-url",
    envvar="QDRANT_URL",
    required=False,
    help="Qdrant URL",
)
@click.option(
    "--qdrant-api-key",
    envvar="QDRANT_API_KEY",
    required=False,
    help="Qdrant API key",
)
@click.option(
    "--collection-name",
    envvar="COLLECTION_NAME",
    required=True,
    help="Collection name",
)
@click.option(
    "--fastembed-model-name",
    envvar="FASTEMBED_MODEL_NAME",
    required=True,
    help="FastEmbed model name",
    default="fast-paraphrase-multilingual-mpnet-base-v2",
)
@click.option(
    "--qdrant-local-path",
    envvar="QDRANT_LOCAL_PATH",
    required=False,
    help="Qdrant local path",
)
def main(
    qdrant_url: Optional[str],
    qdrant_api_key: str,
    collection_name: Optional[str],
    fastembed_model_name: str,
    qdrant_local_path: Optional[str],
):
    # Configure logger
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    log_dir = os.path.join(module_root, "logs")
    log_path = os.path.join(log_dir, "qdrant-server.log")
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Test file creation
    try:
        with open(log_path, 'a') as f:
            f.write("=== Starting new session ===\n")
        print(f"Successfully wrote to log file: {log_path}")
    except Exception as e:
        print(f"Failed to write to log file: {str(e)}")
        raise
    
    # Configure loguru
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
        colorize=True
    )
    
    # Add file handler
    sink_id = logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="1 week",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )
    
    try:
        logger.info("Starting Qdrant server")
        logger.info(f"Logs will be written to: {log_path}")
        
        # XOR of url and local path, since we accept only one of them
        if not (bool(qdrant_url) ^ bool(qdrant_local_path)):
            raise ValueError("Exactly one of qdrant-url or qdrant-local-path must be provided")

        async def _run():
            logger.debug("Creating stdio server")
            try:
                async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                    logger.debug("Stdio server created successfully")
                    
                    logger.debug("Creating Qdrant server instance")
                    server = serve(
                        qdrant_url,
                        qdrant_api_key,
                        collection_name,
                        fastembed_model_name,
                        qdrant_local_path,
                    )
                    logger.debug("Qdrant server instance created")
                    
                    logger.debug("Starting server run loop")
                    try:
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
                    except Exception as e:
                        logger.error(f"Server run loop failed: {str(e)}", exc_info=True)
                        raise
            except Exception as e:
                logger.error(f"Failed to create or run stdio server: {str(e)}", exc_info=True)
                raise

        logger.debug("Starting asyncio run")
        asyncio.run(_run())
    finally:
        try:
            logger.remove(sink_id)
        except ValueError:
            pass  # Игнорируем ошибку, если хендлер уже удален
