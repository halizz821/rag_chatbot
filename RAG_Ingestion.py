import json
import os
import pickle
from typing import List, Generator
from pathlib import Path

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain components
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()


def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )
    return elements


def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )
    return chunks


def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"],
        "source_file": chunk.metadata.filename if hasattr(chunk.metadata, "filename") else None,
        "page_number": chunk.metadata.page_number if hasattr(chunk.metadata, "page_number") else None,
    }

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)
            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt_text = f"""You are an expert system designed to create highly searchable document
        representations for Retrieval-Augmented Generation (RAG).
        Below is the content extracted from a PDF. Convert ALL of this into a unified, highly searchable text block.
        TEXT CONTENT:
        {text}
        """

        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        prompt_text += """
        YOUR TASK: Generate a single, comprehensive search-optimized description of the content.
        Do NOT summarize; instead, **expand** the content into a searchable representation incorporating tables and image context.
        """

        message_content = [{"type": "text", "text": prompt_text}]

        for image_base64 in images:
            if image_base64:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                })

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"AI summary failed: {e}")
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def ingest_pdf_generator(file_path: str, persist_dir: str = "db/chroma_db", bm25_path: str = "db/bm25_docs.pkl"):
    """
    Generator that processes a single PDF, yielding status updates, and indexes into Chroma & BM25.
    Yields dictionary status updates.
    """
    try:
        os.makedirs(os.path.dirname(persist_dir), exist_ok=True)
        
        yield {"status": "partitioning", "message": f"Partitioning document: {os.path.basename(file_path)}"}
        elements = partition_document(file_path)
        
        yield {"status": "chunking", "message": f"Creating smart chunks from {len(elements)} elements..."}
        chunks = create_chunks_by_title(elements)
        
        langchain_documents = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            yield {"status": "summarizing", "message": f"Summarizing chunk {i + 1} of {total_chunks}..."}
            content_data = separate_content_types(chunk)
            
            if content_data["tables"] or content_data["images"]:
                enhanced_content = create_ai_enhanced_summary(
                    content_data["text"], content_data["tables"], content_data["images"]
                )
            else:
                enhanced_content = content_data["text"]
                
            doc = Document(
                page_content=enhanced_content,
                metadata={
                    "original_content": json.dumps({
                        "raw_text": content_data["text"],
                        "tables_html": content_data["tables"],
                        # Omitting base64 to save space in Chroma metadata
                        "source_file": content_data["source_file"],
                        "page_number": content_data["page_number"],
                    })
                },
            )
            langchain_documents.append(doc)
            
        yield {"status": "indexing", "message": "Adding documents to Chroma..."}
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            collection_name="rag_ingestion",
            embedding_function=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        vectorstore.add_documents(langchain_documents)
        
        yield {"status": "indexing", "message": "Updating BM25 sparse index..."}
        
        # We need all documents globally for BM25. We load existing, append, and save back.
        all_docs = []
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                all_docs = pickle.load(f)
                
        all_docs.extend(langchain_documents)
        
        with open(bm25_path, "wb") as f:
            pickle.dump(all_docs, f)

        yield {"status": "complete", "message": "Ingestion completed successfully."}
    except Exception as e:
        yield {"status": "error", "message": str(e)}

