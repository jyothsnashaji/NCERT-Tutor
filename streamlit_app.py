import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.markdown("""
<div style='padding: 15px; border-radius: 10px; background-color: #f0f8ff; text-align: center'>
    <h4 style='color: #2c3e50;'>🎓 TutorAI </strong></h4>
    <p style='color: #34495e;'>How can I help you?</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_box" not in st.session_state:
    st.session_state.input_box = ""
if "view" not in st.session_state:
    st.session_state.view = "chat"  # can be "chat" or "quiz"
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
    st.session_state.quiz_answers = {}

def add_message(sender, message):
    st.session_state.messages.append({"sender": sender, "message": message})

def evaluate_quiz(quiz):
    score = 0
    st.subheader("📊 Results")
    for i, q in enumerate(quiz):
        selected = st.session_state.quiz_answers.get(i)
        if not selected or selected == "-- Select an option --":
            result = f"⚠️ Not answered. Correct answer: {q['answer']}"
        else:
            is_correct = selected == q["answer"]
            result = "✅ Correct!" if is_correct else f"❌ Incorrect. Correct answer: {q['answer']}"
            if is_correct:
                score += 1
        st.markdown(f"**Q{i+1}: {q['question']}**")
        st.markdown(f"Your answer: {selected or 'None'} — {result}")

    st.success(f"**Your Score: {score} / {len(quiz)}**")

    if st.button("🔙 Back to Chat", key="back_to_chat_from_results"):
        st.session_state.view = "chat"

def display_quiz(quiz):
    st.session_state.view = "quiz"  # Ensure it stays in quiz view
    st.subheader("📋 Quiz Time!")

    for i, q in enumerate(quiz):
        options_with_placeholder = ["-- Select an option --"] + q["options"]
        selected = st.radio(
            f"{i+1}. {q['question']}",
            options=options_with_placeholder,
            key=f"quiz_q{i}"
        )
        st.session_state.quiz_answers[i] = selected

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("✅ Submit Quiz"):
            evaluate_quiz(quiz)

    with col2:
        if st.button("🔙 Back to Chat", key="back_to_chat_from_quiz"):
            st.session_state.view = "chat"

def submit_question():
    question = st.session_state.input_box.strip()
    if question:
        add_message("user", question)

        try:
            response = requests.post(f"{BACKEND_URL}/ask", json={"question": question, "session_id": "user123"})
        except Exception as e:
            add_message("assistant", f"❌ Backend error: {e}")
            st.session_state.input_box = ""
            return

        if response.status_code == 200:
            data = response.json()
            quiz = data.get("quiz", None)
            answer = data.get("answer", "No answer received.")
            docs = data.get("retrieved_documents", [])

            st.session_state.last_answer = answer
            st.session_state.last_docs = docs

            if quiz:
                st.session_state.quiz_data = quiz
                st.session_state.view = "quiz"
                add_message("assistant", "Generated quiz")
            else:
                add_message("assistant", answer)
        else:
            add_message("assistant", f"Error: {response.status_code} - {response.text}")

        # Clear input box
        st.session_state.input_box = ""


# Only show chat view if not in quiz mode
if st.session_state.view == "chat":
    # Display chat messages
    for msg in st.session_state.messages:
        if msg["sender"] == "user":
            st.markdown(
                f"<div style='text-align: right; background-color:#DCF8C6; padding:8px; margin:5px; border-radius:10px; max-width: 70%; float: right;'>{msg['message']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='text-align: left; background-color:#E8E8E8; padding:8px; margin:5px; border-radius:10px; max-width: 70%; float: left;'>{msg['message']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='clear: both;'></div>", unsafe_allow_html=True)

    # Input box
    st.text_input("Enter your question:", key="input_box", on_change=submit_question)

    if st.button("📚 Cite Sources"):
        docs = st.session_state.get("last_docs", [])
        if not docs:
            st.info("No relevant sources found.")
        else:
            st.markdown("**Retrieved Documents:**")
            for doc in docs:
                st.markdown(f"""
- **Page {doc['page']}** - ({doc['file']})
  > {doc['snippet']}
""")

# Show quiz view
elif st.session_state.view == "quiz":
    display_quiz(st.session_state.quiz_data)
