import pandas as pd
import inspect
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
        # Initialize the tracking DataFrame
        self.query_df = pd.DataFrame(columns=[
            'Query_ID', 'Query_Text', 'Query_Chars', 'Response_Text', 
            'Response_Chars', 'Response_Time', 'Accuracy'
        ])
    
    def ask(self, query, query_id=None, context=None):
        """
        Main ask method that can optionally take a query_id parameter
        If query_id is not provided, auto-generates one
        """
        # If no query_id provided, auto-generate one
        if query_id is None:
            query_id = f"Q_{len(self.query_df) + 1}"
        
        if context is None:
            context = "Each document represents a task. All tasks belong to a same workflow execution trace. "
            context += "The time the task started is stored in the started_at. The time the task ended is stored in the ended_at. The task duration is ended_at - started_at for each task "
        
        # Prepare full query text
        full_query = f"{context}. {query}"
        
        # Time the query
        t0 = time()
        result = self.qa_chain({"query": full_query})
        response_time = time() - t0
        
        # Extract response text
        response_text = result["result"]
        
        # Calculate character counts
        query_chars = len(query)
        response_chars = len(response_text)
        
        # Add to tracking DataFrame
        new_row = {
            'Query_ID': query_id,
            'Query_Text': query,
            'Query_Chars': query_chars,
            'Response_Text': response_text,
            'Response_Chars': response_chars,
            'Response_Time': response_time,
            'Accuracy': None  # To be filled manually
        }
        
        self.query_df = pd.concat([self.query_df, pd.DataFrame([new_row])], ignore_index=True)
        
        print(f"Q: {query}")
        print(response_text)
        print(f"---------------- I took {response_time:.1f} s to answer this.")
        print("\n\n")
        
        return result
    
    def ask_by_id(self, query_id, query, context=None):
        """
        Alternative method where you explicitly provide the query_id as first parameter
        This is kept for backward compatibility
        """
        return self.ask(query, query_id=query_id, context=context)
    
    def ask_with_id(self, query_id, query, context=None):
        """
        Alias for ask_by_id to match the naming used in benchmark_query_with_tracking
        """
        return self.ask(query, query_id=query_id, context=context)
    
    def update_accuracy(self, query_id, accuracy_score):
        """Method to manually update accuracy for a specific query"""
        mask = self.query_df['Query_ID'] == query_id
        if mask.any():
            self.query_df.loc[mask, 'Accuracy'] = accuracy_score
            print(f"Updated accuracy for {query_id}: {accuracy_score}")
        else:
            print(f"Query ID {query_id} not found")
    
    def get_query_stats(self):
        """Get summary statistics of all queries"""
        return self.query_df.describe()
    
    def export_queries(self, filename="query_results.csv"):
        """Export the query DataFrame to CSV"""
        self.query_df.to_csv(filename, index=False)
        print(f"Query results exported to {filename}")
    
    def build(self, workflow_id):
        self.tasks = Flowcept.db.query({"workflow_id": workflow_id})
        docs = []
        for d in self.tasks:
            content = str(d)  
            metadata = {"task_id": d.get("task_id", "unknown")}
            docs.append(Document(page_content=content, metadata=metadata))
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(docs, embedding_model)
        
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

# Usage examples:
# qa = QAChain()
# qa.build(your_workflow_id)

# Now you can use any of these approaches:
# result = qa.ask("Your query here")  # Auto-generates ID
# result = qa.ask("Your query here", query_id="Q_1_DF_L")  # Custom ID
# result = qa.ask_by_id("Q_1_DF_L", "Your query here")  # Original method
# result = qa.ask_with_id("Q_1_DF_L", "Your query here")  # For benchmark compatibility

# Update accuracy manually:
# qa.update_accuracy("Q_1_DF_L", 0.85)

# View the tracking DataFrame:
# print(qa.query_df)

# Export results:
# qa.export_queries("my_query_results.csv")