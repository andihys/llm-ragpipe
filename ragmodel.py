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

class RAGPipeline:
    def __init__(self):
        self.qa_chain = self._initialize_pipeline()

    @staticmethod
    def _initialize_pipeline():
        try:
            # Load document store
            load_chroma_store()

            # Define prompt template
            prompt = ChatPromptTemplate.from_template(
                """Answer the following question based only on the provided context:
                <context>
                {context}
                </context>
                Question: {input}"""
            )

            # Load language model
            llm = load_gpt_model()

            # Create document combination chain
            combine_documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

            # Create RAG pipeline
            qa_chain = create_retrieval_chain(retriever=get_retriever(), combine_docs_chain=combine_documents_chain)

            logger.info('Pipeline RAG successfully initialized.')
            return qa_chain
        except Exception as e:
            logger.exception(f'Pipeline RAG initialization error: {e}')
            return None

    def query(self, prompt: str) -> dict:
        if not self.qa_chain:
            return {"response": "Pipeline RAG not available. Please check the initialization.", "sources": []}

        try:
            result = self.qa_chain.invoke({'input': prompt})

            response = result.get("answer", "No response available.")
            sources = [doc.page_content for doc in result.get("context", [])]

            logger.info('Query processed successfully.')
            return {"response": response, "sources": sources}
        except Exception as e:
            logger.exception(f'Query processing error: {e}')
            return {"response": "Query processing error.", "sources": []}


if __name__ == "__main__":
    rag_pipeline = RAGPipeline()

    while True:
        user_input = input("Enter your question (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        response = rag_pipeline.query(user_input)
        print("\nResponse:\n")
        pprint(response['response'])
        print("\nSources:\n")
        pprint(response['sources'])
