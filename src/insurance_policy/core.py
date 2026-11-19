from langchain_core.messages import convert_to_openai_messages, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI


from memory import memory
from retrieval import retrieve
from settings import settings


class PolicyQANDA:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.model = ChatOpenAI(model=settings.chat_model,
                                api_key=settings.openrouter_api_key, base_url=settings.openrouter_baseurl)
        self.list_of_policies = []

    async def enhance_query(self, query: str):
        """
        This function take user query and return enhanced query based on policies and last conversation.

        """
        with open("./prompts/enhence-user-query.md") as fp:
            prompt = PromptTemplate.from_template(fp.read(), template_format="mustache")

        _policies = "\n".join(self.list_of_policies)

        chain = prompt | self.model | JsonOutputParser()

        out = await chain.ainvoke({
            "chat_history": memory.get_user_history(self.user_id),
            "policy_list": _policies, "user_question": query
        })
        return out

    async def chat(self, question: str):
        rag_query = await self.enhance_query(question)
        policy_content = ""
        if rag_query.get("needs_retrieval"):
            rag_query = rag_query.get("rewritten_question")
            policy_content = await retrieve([rag_query], self.user_id)
            points = policy_content[0].points
            if points:
                policy_content = points[0].payload["text"]
            else:
                policy_content = "We won't be able to find policy content."

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
            Always add notes `THIS IS AI GENERATED TEXT SO CROSS CHECK WITH INSURANCE AGENT. AI CAN MAKE MISTAKES.`
            Note: Be polite when you anwser to user don't tell maybe sometimes content not available so tell to user as polite or professional way.
            """

            memory.set(self.user_id, SystemMessage(content=prompt))

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
                *convert_to_openai_messages(memory.chat_history.get(self.user_id, [])),
                ("user", user)
            ])
            chain = prompt | self.model

        chain.with_alisteners(on_end=memory.save_history)
        async for chunk in chain.astream({}):
            for block in chunk.content_blocks:
                if block.get("type") == "text" and block.get("text"):
                    yield block["text"]


async def chat(query: str, user_id):
    pol = PolicyQANDA(user_id=user_id)
    async for stem in pol.chat(query):
        yield stem
