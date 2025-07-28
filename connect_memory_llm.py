import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv()
# Step 1 : Setup LLM (Mistral with huggingface)

HF_TOKEN=os.environ.get("HF_TOKEN")

HUGGINGFACE_REPO_ID='google/flan-t5-large'

from langchain_google_genai import ChatGoogleGenerativeAI

def load_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.5,
        top_p=0.95,
        max_new_tokens=512,
        return_full_text=False
    )
"""def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        top_p=0.95,
        max_new_tokens=512,
        return_full_text=False
    )
    return llm
"""
# which kind of precaution we should take for diabetes


# Step 2 : Connect LLM with FAISS and Create Chain

CUSTOM_PROMPT_TEMPLATE ="""
Use the pieces of information providede in the context to answer user's question.
If you don't know the answer, just say that you don't know, in a gentle way with add of quesstion, and do not try to fetch that information from anywhere
 don't try to make up an answer.
Do not include any information that is not in the context.

Context:{context}
Question: {question}

Start your answer with "Answer:"
Start the answer directly. No small talk, no preamble.Please.
"""
def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,
                          input_variables=["context","question"])
    return prompt
#load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH,
                    embedding_model,allow_dangerous_deserialization=True)
# Create Chain of QA
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':5}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE) }
) 


# Now invoke with a single query
user_query=input("Enter your question: ")
response=qa_chain.invoke({"query": user_query})
print("RESULT: ", response['result'])
print("SOURCE DOCUMENTS: ", response['source_documents'])
