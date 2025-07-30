import streamlit as st
import requests

st.set_page_config(page_title="Multimodal RAG for PDFs", layout="centered")
st.title("📄 Multimodal PDF RAG Agent")

st.markdown("Upload a PDF and ask any question based on its content.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
question = st.text_input("Enter your question")

if st.button("Ask"):
    if uploaded_file is None:
        st.error("Please upload a PDF file.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            try:
                response = requests.post(
                    "http://localhost:8000/answer",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    data={"question": question}
                )
                if response.status_code == 200:
                    st.success("Answer:")
                    st.write(response.json()["answer"])
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Request failed: {e}")