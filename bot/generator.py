def generate_text_answer(question:str, qa_chain):
    """Get a text answer from the QA chain."""
    result = qa_chain(question)
    return result