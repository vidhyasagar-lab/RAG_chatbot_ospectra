from utils.llm_utils import llm


def generate_answer(question: str, contexts: list[str]) -> str:
    prompt = f"""You are a helpful AI chatbot assistant.
    Answer the user's question clearly and accurately using ONLY the information provided in the context below.
    If the answer is not present in the context, respond with: "The answer is not available in the provided context."

    Context:
    {chr(10).join(contexts)}

    Question:
    {question}
    """

    return llm.invoke(prompt).content
