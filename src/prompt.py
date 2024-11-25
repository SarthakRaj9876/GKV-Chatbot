
# Refined prompt for detailed responses
prompt_template = """
Use the following context to answer the user's question thoroughly.
Ensure your answer is detailed and includes examples or additional explanations where relevant.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}


Helpful Answer:
"""