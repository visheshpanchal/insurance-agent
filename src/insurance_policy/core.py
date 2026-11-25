from pathlib import Path

import anyio
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import SystemMessage, convert_to_openai_messages
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from memory import memory
from retrieval import retrieve_points
from settings import settings

NO_CONTEXT = "No relevant section was found in the uploaded policy for this question."
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class PolicyQANDA:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.model = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_baseurl,
        )
        self.list_of_policies = []
        self.last_points = []

    async def enhance_query(self, query: str):
        """
        This function take user query and return enhanced query based on policies and last conversation.

        """
        async with await anyio.open_file(f"{PROMPTS_DIR}/enhence-user-query.md") as fp:
            prompt = PromptTemplate.from_template(await fp.read(), template_format="mustache")

        _policies = "\n".join(self.list_of_policies)

        chain = prompt | self.model | JsonOutputParser()

        out = await chain.ainvoke(
            {
                "chat_history": memory.get_user_history(self.user_id),
                "policy_list": _policies,
                "user_question": query,
            }
        )
        return out

    async def build_context(self, question: str):
        try:
            rag_query = await self.enhance_query(question)
        except OutputParserException as e:
            logger.warning(
                "Query enhancer returned invalid JSON, searching raw question: {}", e
            )
            rag_query = {"needs_retrieval": True, "rewritten_question": question}

        if not isinstance(rag_query, dict):
            rag_query = {"needs_retrieval": True, "rewritten_question": question}

        if not rag_query.get("needs_retrieval", True):
            return ""

        queries = [rag_query.get("rewritten_question") or question]
        for sub in rag_query.get("sub_queries") or []:
            if isinstance(sub, dict) and sub.get("query"):
                queries.append(sub["query"])
        queries = list(dict.fromkeys(queries))

        self.last_points = await retrieve_points(queries, self.user_id)
        chunks = [
            point.payload["text"]
            for point in self.last_points
            if point.payload and point.payload.get("text")
        ]
        if not chunks:
            return NO_CONTEXT
        return "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks, 1))

    async def chat(self, question: str):
        policy_content = await self.build_context(question)

        if not memory.get(self.user_id):
            async with await anyio.open_file(
                f"{PROMPTS_DIR}/insurance-reviewer.md"
            ) as fp:
                prompt = await fp.read()

            user_prompt = f"""        
            I am sharing you data from I fetch that help to understand what he means.
            <context>
            {policy_content}
            </context>


            User has question regarding it's policy can help him to solve.
            {question}

            Answer the question using only the excerpts above. 
            Walk through the relevant clauses step by step, then give a clear conclusion.

            Please answer in politely don't recommend other policy. 
            Always add notes THIS IS AI GENERATED TEXT SO CROSS CHECK WITH INSURANCE AGENT. AI CAN MAKE MISTAKES.
            Note: Be polite when you anwser to user don't tell maybe sometimes content not available so tell to user as polite or professional way.
            """

            memory.set(self.user_id, SystemMessage(content=prompt))

            prompt = ChatPromptTemplate.from_messages(
                [("system", prompt), ("user", user_prompt)]
            )
            chain = prompt | self.model

        else:
            user = f"""
            <chat_history>
            {"Role\n"}
            </chat_history>
            I am sharing you data from I fetch that help to understand what he means.
            <context>
            {policy_content}
            </context>

            User has question regarding it's policy can help him to solve.
            {question}

            Answer the question using only the excerpts above. 
            Walk through the relevant clauses step by step, then give a clear conclusion.

            Please answer in politely don't recommend other policy. 
            Always add notes `THIS IS AI GENERATED TEXT SO CROSS CHECK WITH INSURANCE AGENT. AI CAN MAKE MISTAKES.`
            """

            prompt = ChatPromptTemplate(
                [
                    *convert_to_openai_messages(
                        memory.chat_history.get(self.user_id, [])
                    ),
                    ("user", user),
                ]
            )
            chain = prompt | self.model

        parts: list[str] = []
        async for chunk in chain.astream({}):
            for block in chunk.content_blocks:
                if block.get("type") == "text" and block.get("text"):
                    parts.append(block["text"])
                    yield block["text"]
        memory.add_turn(self.user_id, question, "".join(parts))


async def chat(query: str, user_id):
    pol = PolicyQANDA(user_id=user_id)
    async for stem in pol.chat(query):
        yield stem
