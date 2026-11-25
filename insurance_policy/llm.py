from langchain_openai import OpenAIEmbeddings
from settings import settings


def get_embedding_model():
    return OpenAIEmbeddings(
        api_key=settings.openrouter_api_key,
        model=settings.embedding_model,
        base_url=settings.openrouter_baseurl,
        check_embedding_ctx_length=False,
    )
