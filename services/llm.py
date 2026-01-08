from utils.llm_utils import llm
from utils.logger import logger


def generate_answer(question: str, contexts: list[str]) -> str:
    logger.info("Generating answer from LLM")
    logger.info("Question: %s", question)
    logger.info("Contexts: %s", contexts)
    try:
        prompt = f"""You are a helpful AI chatbot assistant.

       your task is understand the question and provide a concise and accurate answer based on the provided context. If the context does not contain sufficient information to answer the question, respond with "I don't know. No relevant context found."

        Context:
        {chr(10).join(contexts)}

        Question:
        {question}
        """

        response = llm.invoke(prompt)
        logger.info("Answer generated successfully")
        return response.content

    except Exception as e:
        logger.error("Error while generating answer: %s", str(e))
        raise
