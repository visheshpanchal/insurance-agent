from langchain_core.exceptions import OutputParserException
from langchain_core.messages import SystemMessage, convert_to_openai_messages
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from memory import memory
from retrieval import retrieve
from settings import settings

NO_CONTEXT = "No relevant section was found in the uploaded policy for this question."


class PolicyQANDA:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.model = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_baseurl,
        )
        self.list_of_policies = []

    async def enhance_query(self, query: str):
        """
        This function take user query and return enhanced query based on policies and last conversation.

        """
        with open("./prompts/enhence-user-query.md") as fp:
            prompt = PromptTemplate.from_template(fp.read(), template_format="mustache")

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

        chunks = await retrieve(queries, self.user_id)
        if not chunks:
            return NO_CONTEXT
        return "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks, 1))

    async def chat(self, question: str):
        policy_content = await self.build_context(question)

        if not memory.get(self.user_id):
            with open(f"./prompts/insurance-reviewer.md", "r") as fp:
                prompt = fp.read()

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

        chain = chain.with_alisteners(on_end=memory.save_history)
        async for chunk in chain.astream({}):
            for block in chunk.content_blocks:
                if block.get("type") == "text" and block.get("text"):
                    yield block["text"]


async def chat(query: str, user_id):
    pol = PolicyQANDA(user_id=user_id)
    async for stem in pol.chat(query):
        yield stem
