# AI Ticket Support System

This project implements an **AI-powered ticket support system** that groups support tickets and retrieves predefined answers using **semantic embeddings**.

## Features
- Group tickets based on **semantic similarity** using `sentence-transformers`.
- Retrieve **predefined answers** for each ticket from a CSV.
- Supports **multiple categories**: password issues, login issues, leave balance, attendance.
- Interactive **Streamlit interface** for uploading tickets and answers.

## Usage
1. Clone the repository:

git clone https://github.com/Bhavani-Sakhamuri/AI_Ticket_Support/

cd AI_Ticket_Support


2. Install dependencies:

pip install -r requirements.txt


3. Run the Streamlit app:

streamlit run app_ticket_support.py


4. Upload your `tickets.csv` and `ticket_answers.csv` in the sidebar.
5. View **grouped tickets** and **retrieved answers** interactively.

## Sample CSVs

* `tickets.csv` — Incoming user tickets
* `ticket_answers.csv` — Predefined questions and answers per category

## Tech Stack

* Python
* Streamlit
* Sentence-Transformers
* PyTorch
* Pandas

