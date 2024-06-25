from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
def initialize_faiss():
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    

    try:
        db = FAISS.load_local(folder_path="./faiss_db", index_name="Index", embeddings=embeddings, allow_dangerous_deserialization=True)
        print("Faiss index loaded ")
        return db
    except Exception as e:
        print("Faiss index loading failed in inference file \n", e)
        return None
    

def search_with_score(db, query):
    docs = db.similarity_search_with_score(query)
    searchresult = ' '.join([': '.join(map(str, doc)) for doc in docs])
    print(searchresult)
    llminput = query + searchresult
    llm = Ollama(model="llama3")
    prompt = ChatPromptTemplate.from_template("""You are an assistant for Manipal Institute of Technology, search information is provided after the query for context, Answer questions accurately and completely:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    output = retrieval_chain.invoke({"input": llminput})
    return output


