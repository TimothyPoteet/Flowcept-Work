from time import time
from langchain_sambanova import ChatSambaNovaCloud
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings  # or any other compatible embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from flowcept import Flowcept


class QAChain:

    def __init__(self):
        self.qa_chain = None
        self.tasks = None

    def ask(self, query, context=None):
    
        if context is None:
            context = "Each document represents a task. All tasks belong to a same workflow execution trace. "
            context += "The time the task started is stored in the started_at. The time the task ended is stored in the ended_at. The task duration is ended_at - started_at for each task "
        
        t0 = time()
        result = self.qa_chain({"query": f"{context}. {query}"})
        print(f"Q: {query}")
        print(result["result"])
        print(f"---------------- I took {time()-t0:.1f} s to answer this.")
        print("\n\n")
        return result

    def build(self, workflow_id):
    

        self.tasks = Flowcept.db.query({"workflow_id": workflow_id})
        docs = []
        for d in self.tasks:
            content = str(d)  # convert the dict to a string
            metadata = {"task_id": d.get("task_id", "unknown")}
            docs.append(Document(page_content=content, metadata=metadata))

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(docs, embedding_model)

        # STEP 5: Setup Retriever and LLM
        retriever = vectorstore.as_retriever()

        llm = ChatSambaNovaCloud(
            model='Llama-3.3-Swallow-70B-Instruct-v0.4',
            max_tokens=10024,
            temperature=0.7,
            top_p=0.01,
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        return self

