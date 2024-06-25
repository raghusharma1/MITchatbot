from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from inference import initialize_faiss, search_with_score
import gradio as gr


db = initialize_faiss()


# Define the chatbot function
def chatbot(query, score):
    if db is None:
        return "Faiss index loading failed"
    
    docs = search_with_score(db, query)
    #if docs[0][1] > score:
        
        #return "Search not conducted accurately"
    #else:
        #for doc,score in docs:
    print(docs)        
    return docs["answer"]

# Create a Gradio interface for the chatbot
gr.Interface(fn=chatbot, inputs=["text","slider"], outputs="html", title="MIT Chatbot").launch(share=True)

