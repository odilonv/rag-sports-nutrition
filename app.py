import streamlit as st
from main import setup_rag, ask_question

st.set_page_config(page_title="AI Sports Nutritionist", page_icon="💪")
st.title("💪 AI Sports Nutritionist")
st.write("Ask your questions about recovery, hydration, or prep!")


@st.cache_resource
def load_models():
    return setup_rag()

retriever, rag_chain = load_models()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: What is the recommended protein intake?"):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Searching medical guidelines..."):
        answer, context = ask_question(prompt, retriever, rag_chain)
        
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("🔍 View RAG context/sources"):
            st.info(context)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
