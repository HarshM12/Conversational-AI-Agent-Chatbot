import streamlit as st
from dotenv import load_dotenv
from agent.agent_core import get_agent_executor

load_dotenv()

st.title("Conversational AI Agent Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "intermediate_steps" not in st.session_state:
    st.session_state.intermediate_steps = []

print(">>> App started") 

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            agent_executor = get_agent_executor()
            result = agent_executor.invoke({
                "input": prompt,
                "chat_history": "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]])
            })
            
            st.markdown(result["output"])
            
            if "intermediate_steps" in result:
                st.session_state.intermediate_steps = result["intermediate_steps"]
                with st.expander("Agent Reasoning Steps"):
                    for step in result["intermediate_steps"]:
                        action, observation = step
                        st.markdown(f"**Thought/Action:** {action.log}")
                        st.markdown(f"**Observation:** {observation}")

    st.session_state.messages.append({"role": "assistant", "content": result["output"]})

st.write("### Provide Feedback")
print(">>> Rendering feedback section")
feedback = st.text_input(
    "Was this answer helpful? (leave a comment or type 👍 / 👎)",
    key="feedback_input"
)
print(f">>> Feedback input value: {feedback}")
if st.button("Submit Feedback", key="submit_feedback_button"):
    print(f">>> Submit Feedback button clicked with feedback: {feedback}")
    if feedback.strip():
        try:
            from agent.tools import collect_feedback
            fb_result = collect_feedback(feedback)
            st.success(fb_result)
        except ImportError as e:
            print(f">>> ImportError: {str(e)}")
            st.error(f"Error importing collect_feedback: {str(e)}")
    else:
        st.error("Please provide feedback before submitting.")