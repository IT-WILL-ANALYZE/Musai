from langchain_classic import hub

def rag_prompt(): 
    '''
    <기본 RAG prompt>
    human
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    '''
    return hub.pull("rlm/rag-prompt")


def reduce_prompt(): 
    '''
    <기본 RAG prompt>
    human
    The following is set of summaries:
    {doc_summaries}
    Take these and distill it into a final, consolidated summary of the main themes. 
    Helpful Answer:
    '''
    return hub.pull("rlm/rag-prompt")
