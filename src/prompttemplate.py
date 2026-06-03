from langchain_core.prompts import ChatPromptTemplate

class PromptManager:

    @staticmethod
    def rag_prompt():

        return ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are an intelligent RAG assistant.

Answer ONLY using the provided context and don't anwer on your own.

Rules:
-  Do NOT hallucinate
- If answer is missing, say you don't know
- Combine information from multiple retrieved chunks
- Provide detailed explanations
- Use bullet points and give only 3 pointer summary answer
- Include important technical details
- If you answer of he query asked by user is not present in contxt than simply say I don't know the answer.
"""
            ),

            (
                "human",
                """
Context:
{context}

Question:
{question}
"""
            )
        ])
