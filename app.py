import streamlit as st
from Helper import user_input, extract_text_from_pdf, get_text_chunks, get_vector_store

def preprocess_pdf(uploaded_files):
    progress = st.progress(0)
    all_text_chunks = []

    with st.spinner("Extracting and processing documents..."):
        for i, uploaded_file in enumerate(uploaded_files):
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(uploaded_file)
            progress.progress(int((i + 1) / len(uploaded_files) * 33))

            # Split the text into chunks
            text_chunks = get_text_chunks(pdf_text)
            all_text_chunks.extend(text_chunks)
            progress.progress(int((i + 1) / len(uploaded_files) * 66))

        # Create vector store
        vector_store = get_vector_store(all_text_chunks)
        progress.progress(100)

    st.success(f"Processed {len(uploaded_files)} documents successfully!")
    return vector_store

def create_ui():
    st.set_page_config(page_title="PDF Made Easy", page_icon=":book:", layout="centered")

    st.title("📚 PDF Made Easy!")
    st.sidebar.write("### Welcome to PDF Made Easy!")
    st.sidebar.write("Upload multiple PDFs, preprocess them, and chat with the documents effortlessly.")

    st.markdown("### Instructions")
    st.markdown(
        """
        1. Upload multiple PDF files.
        2. Wait for the documents to be processed.
        3. Ask questions to get detailed answers from the documents.
        """
    )

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Display a message informing the user that the files are being processed
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = preprocess_pdf(uploaded_files)

        st.markdown("### Chat with Your Documents")
        question = st.text_input("Ask a question:")

        if question:
            with st.spinner("Generating response..."):
                response, context_docs = user_input(question)
                output_text = response.get('output_text', 'No response')
                st.success("Response:")
                st.write(output_text)
        else:
            st.warning("Please enter a question to interact with the documents.")

    else:
        st.info("Please upload PDF files to get started.")


def main():
    create_ui()

if __name__ == "__main__":
    main()
