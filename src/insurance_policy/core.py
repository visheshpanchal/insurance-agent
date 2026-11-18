import asyncio

from langchain_core.messages import AIMessage, HumanMessage, convert_to_openai_messages, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

from llm import ChatSettings, api_key, url
from retrieval import retrieve


class PolicyQANDA:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.history: dict[str, list[AIMessage | SystemMessage | HumanMessage]] = {}
        self.settings = ChatSettings()
        self.model = ChatOpenAI(model=self.settings.model, api_key=api_key, base_url=url)
        self.list_of_policies = []

    def get_user_history(self):
        _history = ""
        for history in self.history:
            if isinstance(history, AIMessage):
                role = "assistant"
            elif isinstance(history, HumanMessage):
                role = "user"

            _history += f"Role: {role}\n {history.content}\n\n"

        return _history

    async def enhance_query(self, query: str):
        """
        This function take user query and return enhanced query based on policies and last conversation.

        """
        with open("./prompts/enhence-user-query.md") as fp:
            prompt = PromptTemplate.from_template(fp.read(), template_format="mustache")

        _policies = "\n".join(self.list_of_policies)

        chain = prompt | self.model | JsonOutputParser()

        out = await chain.ainvoke({
            "chat_history": self.get_user_history(),
            "policy_list": _policies, "user_question": query
        })
        return out

    async def chat(self, question: str):
        rag_query = await self.enhance_query(question)
        policy_content = ""
        if rag_query.get("needs_retrieval"):
            rag_query = rag_query.get("rewritten_question")
            policy_content = await retrieve([rag_query])
            policy_content = policy_content[0].points[0].payload["text"]

        if not self.history.get(self.user_id):
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
            Always add notes `THIS IS AI GENERATED TEXT SO CROSS CHECK WITH INSURANCE AGENT. AI CAN MAKE MISTAKES.`
            """
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt),
                    ("user", user_prompt)
                ]
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

            prompt = ChatPromptTemplate([
                *convert_to_openai_messages(self.history),
                ("user", user)
            ])
            chain = prompt | self.model

        async for chunk in chain.astream({}):
            for block in chunk.content_blocks:
                if block.get("type") == "text" and block.get("text"):
                    yield block["text"]


async def chat(query: str, user_id):
    obj = PolicyQANDA(user_id=user_id)


if __name__ == "__main__":
    asyncio.run(chat())
