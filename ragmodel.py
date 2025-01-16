# Implement a Retrieval-Augmented Generation (RAG) pipeline.
# This includes a retriever, document combination,
# and GPT model integration for answering queries.

from pprint import pprint
from llm import load_gpt_model, logger
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from vectorDB import get_retriever, load_chroma_store
from langchain_core.prompts import ChatPromptTemplate

# retriever
retriever = get_retriever()

def create_rag_pipeline():
    try:
        # load documents
        load_chroma_store()

        # prompt template
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}""")

        # load llm model
        llm = load_gpt_model()

        # chain
        combine_documents_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )

        # rag pipeline
        qa_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=combine_documents_chain
        )

        # print("Pipeline RAG successfully created.")
        logger.info('Pipeline RAG successfully created.')
        return qa_chain
    except Exception as e:
        # print(f'Pipeline RAG error: {e}')
        logger.exception(f'Pipeline RAG error: {e}')
        return None

class RAGPipeline:

    def __init__(self):
        self.qa_chain = create_rag_pipeline()

    def query(self, prompt: str) -> dict:
        if self.qa_chain is None:
            return {"response": "Pipeline RAG not available. Please check the rag initialization.", "sources": []}

        try:
            # make the query
            result = self.qa_chain.invoke({'input': prompt})

            # extract the response and the checked documents
            response = result["answer"]
            source_documents = result["context"]

            # Format the sources as a list of document content
            sources = [doc.page_content for doc in source_documents]
            logger.info('Risposta e documenti di origine ricevuti!')
            return {"response": response, "sources": sources}
        except Exception as e:
            print(f"Query error: {e}")
            logger.exception(e)
            return {"response": "Query error.", "sources": []}

if __name__ == "__main__":
    # Init
    rag_pipeline = RAGPipeline()

    while True:
        prompt = input("Make a question (or 'exit' to exit): ")
        if prompt.lower() == 'exit':
            break
        response = rag_pipeline.query(prompt)
        print(f"Response:\n")
        pprint(response['response'])
        print(f"\n\nSources:\n")
        pprint(response['sources'])