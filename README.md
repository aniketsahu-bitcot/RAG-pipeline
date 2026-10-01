# RAG-pipeline
A FastAPI backend that lets you upload PDF documents and ask questions about them in a multi-turn conversation. It uses Retrieval-Augmented Generation (RAG): relevant passages are retrieved from the PDFs and passed to a local Ollama LLM, which answers using only that context. Everything runs locally — no external API keys required.
