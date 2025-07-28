import streamlit as st

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os


DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,
                          input_variables=["context","question"])
    return prompt

def load_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.5,
        top_p=0.95,
        max_tokens=1024
    )




def main():
    st.title("AI Medical Chatbot!")

    if 'message' not in st.session_state:
        st.session_state.message = []

    for message in st.session_state.message:
        st.chat_message(message['role']).markdown(message['content'])
        

    prompt=st.chat_input("Enter Your problem here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.message.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE ="""
            You are a knowledgeable and friendly medical assistant. Use the provided context to answer the user's question clearly and informatively.

            If the answer is present in the context, provide a detailed explanation.

            If you don’t know the answer from the context, politely say: 
            "I’m sorry, I couldn't find this information in the provided source."

            Context:
            {context}

            Question:
            {question}

            Answer:
            """


        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                raise ValueError("Vectorstore is not loaded properly.")
            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE) }
            ) 

            response = qa_chain.invoke({"query": prompt})
            result = response['result']
            source_documents =response['source_documents']
            # Cleaned result
            answer_md = f"### 🩺 Answer:\n{result.strip()}\n"

            # Clean source (show only 1st document, minimal info)
            if source_documents:
                source = source_documents[0]
                source_file = os.path.basename(source.metadata.get("file_path", ""))
                source_page = source.metadata.get("page", "N/A")
                source_md = f"\n<sup>📚 Source: {source_file} (Page {source_page})</sup>"
            else:
                source_md = "\n<sup>⚠️ Source not found</sup>"
            result_to_show = answer_md + source_md
            
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.message.append({'role': 'assistant', 'content': result_to_show})
        except Exception as e:
            st.error(f"Error loading vectorstore or creating QA chain: {e}")

            

        #response="Hi, I am a medical chatbot. I can answer your questions based on the medical knowledge I have been trained on. Please ask me anything related to health, diseases, treatments, or general medical advice."
        
        #st.session_state.message.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    main()