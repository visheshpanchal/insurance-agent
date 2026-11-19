import asyncio

from vectordb import QdrantConnection
from settings import QdrantSettings

async def main():
    qdb = QdrantSettings()
    async with QdrantConnection(qdb) as fp:
        pass

asyncio.run(main())

