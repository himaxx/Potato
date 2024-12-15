import streamlit as st
import requests

# Define the API endpoint
ENDPOINT = "https://your-api-endpoint"

# Function to fetch questions
def fetch_questions(endpoint):
    try:
        response = requests.get(endpoint)
        print(f"Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch questions: {e}")
        return None

# Streamlit UI
st.title("MCQ Quiz")

data = fetch_questions(ENDPOINT)
if data:
    for question in data:
        st.subheader(question["question"])
        options = question.get("options", [])
        selected_option = st.radio("Choose an answer:", options, key=question["question"])
        if st.button("Submit", key=f"submit-{question['question']}"):
            if selected_option == question["answer"]:
                st.success("Correct!")
            else:
                st.error(f"Wrong! The correct answer is: {question['answer']}")
else:
    st.error("No questions available.")
