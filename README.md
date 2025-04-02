# RAG Pipeline with Local and API-based LLM

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline. It integrates document retrieval, contextual response generation, and large language model (LLM) capabilities. The pipeline supports both **local LLMs** (e.g., Ollama) and LLMs accessed via an **API**.

## Features

- **Document Loading and Indexing**: 
  - Supports loading documents in `.txt` and `.pdf` formats.
  - Documents are stored and retrieved using **ChromaDB** with **Azure OpenAI embeddings**.

- **Flexible LLM Integration**:
  - Use an API-based LLM (e.g., OpenAI GPT models).
  - Use a local LLM like Ollama for enhanced flexibility.

- **RAG Workflow**:
  - Combines a retriever with a language model to generate context-aware answers.
  - Ensures the model generates responses only based on the retrieved context.

## Requirements

- Python 3.8 or later
- ChromaDB
- LangChain
- Pydantic
- Azure OpenAI API (if using Azure embeddings) or other API
- Ollama or other supported local LLMs (optional)

## Setup

1.**Configure Azure OpenAI Settings**:
   Update the `azure_settings` in the `config` module with your Azure deployment credentials.

2.**Prepare Documents**:
   Place your `.txt` and `.pdf` documents in the `documents/` directory for indexing.

3.**Run the Pipeline**:
   - Start the pipeline with:
     ```bash
     python ragmodel.py
     ```

## Usage

1. When prompted, input a query:
   ```
   Make a question (or 'exit' to exit): What is quantum computing?
   ```

2. The system will:
   - Retrieve relevant documents from the database.
   - Use the LLM to generate a response based on the retrieved context.
   - Return both the answer and the sources used.

## Example

Input:
```
What are the main principles of artificial intelligence?
```

Output:
```json
{
  "response": "Artificial intelligence (AI) involves principles such as machine learning, natural language processing, and computer vision.",
  "sources": [
    "Introduction to AI - Chapter 1",
    "AI Principles - Lecture Notes"
  ]
}
```

## Customization

- **Switching LLMs**:
  Modify the `load_gpt_model` function in `ragmodel.py` to switch between local or API-based models.

- **Document Directory**:
  Change the `DOC_PATH` variable in `vectorDB.py` to update the location of your document directory.

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License.
