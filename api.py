import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from RAG_Ingestion import ingest_pdf_generator
from retrieval import HybridRetriever

app = FastAPI()

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hybrid_retriever = HybridRetriever()

class ChatRequest(BaseModel):
    query: str

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    def event_generator():
        try:
            for update in ingest_pdf_generator(file_path):
                yield f"data: {json.dumps(update)}\n\n"
            
            # After ingestion is complete, reload the retriever
            hybrid_retriever.reload()
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Retrieve Hybrid context
    docs = hybrid_retriever.retrieve(request.query)
    
    if not docs:
        context_str = "No specific context found in the knowledge base."
    else:
        # context_str = "\n\n---\n\n".join([doc.page_content for doc in docs])
        # print(context_str)
        prompt_text = f"""Based on the following documents, please answer this question: {request.query}
                        CONTENT TO ANALYZE:
                        """
        source = []
        for i, chunk in enumerate(docs):
            prompt_text += f"--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])# convert the dumped json string back to dictionary
                source.append(original_data.get('source_file', 'Unknown'))
                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        source = list(set(source))
    # 2. Synthesize using gpt-4o
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    system_prompt = (
        "You are an expert AI assistant part of a RAG Ecosystem. "
        "Use the provided context to answer the user's query comprehensively and concisely. "
        "If you don't know the answer based on the context, state that clearly."
        "don't cite docoments inside the answer"
    )
    
    # We could implement a streaming response here as well, but we will return standard JSON for simplicity
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt_text)
        ])
        
        return {
            "answer": response.content,
            "sources": source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
