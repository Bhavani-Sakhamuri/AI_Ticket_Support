import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AI Ticket Support", layout="wide")

st.title("AI Ticket Support - Grouping & Answer Retrieval")

# ----------------------------
# Step 1: Upload CSVs
# ----------------------------
st.sidebar.header("Upload CSV Files")
tickets_file = st.sidebar.file_uploader("Upload tickets CSV", type=["csv"])
answers_file = st.sidebar.file_uploader("Upload answers CSV", type=["csv"])

if tickets_file and answers_file:
    # Load CSVs
    tickets_df = pd.read_csv(tickets_file)
    answers_df = pd.read_csv(answers_file)

    st.success("Files loaded successfully!")

    # ----------------------------
    # Step 2: Embeddings
    # ----------------------------
    st.info("Generating embeddings...")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    ticket_embeddings = embed_model.encode(tickets_df['ticket_text'].tolist(), convert_to_tensor=True)
    answer_embeddings = embed_model.encode(answers_df['sample_question'].tolist(), convert_to_tensor=True)

    # ----------------------------
    # Step 3: Match tickets to answers
    # ----------------------------
    results = []

    for i, ticket_text in enumerate(tickets_df['ticket_text']):
        cos_scores = util.pytorch_cos_sim(ticket_embeddings[i], answer_embeddings)[0]
        best_idx = cos_scores.argmax()
        matched_category = answers_df.iloc[best_idx]['ticket_category']
        matched_answer = answers_df.iloc[best_idx]['answer']

        results.append({
            'ticket_id': tickets_df.iloc[i]['ticket_id'],
            'ticket_text': ticket_text,
            'category': matched_category,
            'answer': matched_answer
        })

    results_df = pd.DataFrame(results)

    st.subheader("Grouped Tickets with Answers")
    st.dataframe(results_df)

    # ----------------------------
    # Step 4: Optional: Group tickets by category
    # ----------------------------
    grouped = results_df.groupby('category')['ticket_text'].apply(list).reset_index()
    st.subheader("Tickets Grouped by Category")
    st.dataframe(grouped)

else:
    st.warning("Please upload both tickets CSV and answers CSV to proceed.")
