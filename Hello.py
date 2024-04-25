import streamlit as st
from transformers import pipeline

# Load the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_question(question, context):
    # Pass the question and context to the pipeline
    result = qa_pipeline(question=question, context=context)

    # Extract the answer and score from the result
    answer = result["answer"]
    score = result["score"]

    return answer, score

def main():
    st.title("Document Question Answering")

    # Get the document text from the user
    document_text = st.text_area("Enter the document text", height=200)

    # Get the question from the user
    question = st.text_input("Enter your question")

    if st.button("Answer"):
        if document_text.strip() == "":
            st.warning("Please enter the document text.")
        elif question.strip() == "":
            st.warning("Please enter a question.")
        else:
            # Answer the question
            answer, score = answer_question(question, document_text)

            # Display the answer and score
            st.subheader("Answer")
            st.write(answer)
            st.write(f"Score: {score:.2f}")

if __name__ == "__main__":
    main()
