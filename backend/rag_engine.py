import os
from typing import List
from unstructured.partition.pdf import partition_pdf
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def parse_pdf_to_chunks(file_bytes: bytes) -> List[str]:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    chunks = partition_pdf(
        filename=tmp_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    os.remove(tmp_path)
    return [str(chunk) for chunk in chunks if str(chunk).strip()]

def get_gemini_model():
    return ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

def build_rag_chain(context_docs: List[str]):
    context = "\n\n".join(context_docs[:10])
    prompt = PromptTemplate.from_template(
        "You are an expert assistant. Use the context below to answer the user's question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    model = get_gemini_model()
    chain = {
        "context": RunnableLambda(lambda x: context),
        "question": RunnablePassthrough()
    } | prompt | model | StrOutputParser()
    return chain

def get_answer(file_bytes: bytes, question: str) -> str:
    chunks = parse_pdf_to_chunks(file_bytes)
    if not chunks:
        return "Failed to extract content from the PDF."
    chain = build_rag_chain(chunks)
    return chain.invoke({"question": question})