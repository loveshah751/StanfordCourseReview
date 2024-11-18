PREPROMPT = "Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn't let caution get too much in the way of being useful.\n"

PROMPT = """"Use the following pieces of context to answer the question at the end in regards to topic provided.
The context may contains python snippets, text, or any other information. Include both text and code snippets in the answer if applicable.
you can reference the existing code directly from the context instead of writing new code.
No premable or postamble is required. If you don't know the answer, just say that you don't know, don't try to
make up an answer. Don't make up new terms or sample snippets which are not available in the context.
topic:{topic},
context:{context},
code_snippet:{code_snippet}
"""

END_PROMPT = "\n<|prompter|>{query}<|endoftext|><|assistant|>"


ASPECTPROMPT = """"Filter the following aspects to include only those relevant to course summary provided in the context.
 Return only the unique aspects, both semantically and syntactically, as a Python list.
 Do not include any code block formatting or additional text.
{aspect}
{course_summary}
"""

def get_llm_prompt_for_query(query_text, context, code_snippet, topic):
    llm_query = PREPROMPT + PROMPT.format(context="\n".join(context), code_snippet="\n".join(code_snippet), topic=topic)
    llm_query += END_PROMPT.format(query=query_text)
    return llm_query

def get_aspect_prompt(aspects, course_summary_context):
    aspect_query = ASPECTPROMPT.format(aspect="\n".join(aspects), course_summary="\n".join(course_summary_context))
    return aspect_query