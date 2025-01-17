# -*- coding: utf-8 -*-
import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo, get_query_constructor_prompt, StructuredQueryOutputParser
from langchain.retrievers.self_query.base import SelfQueryRetriever

# Set up OpenAI API key and base URL
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://api.perplexity.ai"

# Initialize embeddings and vectorstore
embeddings = OpenAIEmbeddings()
docs = [
    Document(
        page_content="A poor but big-hearted man takes orphans into his home. After discovering his scientist father's invisibility device, he rises to the occasion and fights to save his children and all of India from the clutches of a greedy gangster",
        metadata={"year": 2006, "director": "Rakesh Roshan", "rating": 7.1, "genre": "science fiction"},
    ),
    Document(
        page_content="The story of six young Indians who assist an English woman to film a documentary on the freedom fighters from their past, and the events that lead them to relive the long-forgotten saga of freedom",
        metadata={"year": 2006, "director": "Rakeysh Omprakash Mehra", "rating": 9.1, "genre": "drama"},
    ),
    Document(
        page_content="Three idiots embark on a quest for a lost buddy. This journey takes them on a hilarious and meaningful adventure through memory lane and gives them a chance to relive their college days",
        metadata={"year": 2009, "director": "Rajkumar Hirani", "rating": 9.4, "genre": "comedy"},
    ),
]
vectorstore = Chroma.from_documents(docs, embeddings)

# Define metadata for filtering
metadata_field_info = [
    AttributeInfo(name="genre", description="The genre of the movie.", type="string"),
    AttributeInfo(name="year", description="The year the movie was released", type="integer"),
    AttributeInfo(name="director", description="The name of the movie director", type="string"),
    AttributeInfo(name="rating", description="A 1-10 rating for the movie", type="float"),
]
document_content_description = "Brief summary of a movie"

# Initialize retriever
llm = ChatOpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(llm, vectorstore, document_content_description, metadata_field_info)

# Streamlit Frontend
def main():
    st.title("Self Query Retriever")
    st.write("Retrieve movies based on your specific queries!")

    # User input for query
    user_query = st.text_input("Enter your query:", "I want to watch a movie rated higher than 8")

    # Retrieve results based on query
    if st.button("Search"):
        try:
            results = retriever.invoke(user_query)
            if results:
                st.write("Results:")
                for doc in results:
                    st.write(f"Title: {doc.metadata.get('title', 'Unknown')}, "
                             f"Rating: {doc.metadata.get('rating', 'Unknown')}, "
                             f"Genre: {doc.metadata.get('genre', 'Unknown')}")
            else:
                st.write("No results found.")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
