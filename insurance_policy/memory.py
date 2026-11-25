from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class Memory:
    def __init__(self):
        self.chat_history: dict[
            str, list[AIMessage | SystemMessage | HumanMessage]
        ] = {}
        self.docs: dict[str, dict[str, str]] = {}

    def clear_history_chat(self, user_id):
        if not self.chat_history.get(user_id):
            return
        del self.chat_history[user_id]

    def save_doc(self, user_id: str, doc_name: str, doc: str):
        docs = self.docs.get(user_id, dict())
        docs[doc_name] = doc
        self.docs[user_id] = docs.copy()

    def get_doc(self, user_id: str, doc_name: str):
        docs = self.docs.get(user_id, dict())
        return docs.get(doc_name, "")

    def get_user_history(self, user_id: str):
        _history = ""
        history = self.chat_history.get(user_id, [])
        for his in history:
            if isinstance(his, AIMessage):
                role = "assistant"
            elif isinstance(his, HumanMessage):
                role = "user"
            else:
                continue
            _history += f"Role: {role}\n {his.content}\n\n"
        return _history

    def add_turn(self, user_id: str, question: str, answer: str):
        his = self.chat_history.get(user_id, [])
        his.extend([HumanMessage(content=question), AIMessage(content=answer)])
        self.chat_history[user_id] = his

    def get(self, user_id):
        return self.chat_history.get(user_id, None)

    def set(self, user_id, item):
        old = self.chat_history.get(user_id, [])
        old.append(item)
        self.chat_history[user_id] = old.copy()


memory = Memory()
