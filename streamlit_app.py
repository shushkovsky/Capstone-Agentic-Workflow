import streamlit as st
from agentic_workflow import main

st.set_page_config(page_title="Drug Discovery Assistant", page_icon="🧬", layout="centered")
st.title("🧪 Agentic Drug Discovery Assistant")
st.markdown(
    """
    Welcome! Ask any question related to **genes, drugs, or their targets**  
    and I’ll query the knowledge graph and return an intelligent answer.
    """
)

@st.cache_resource
def get_langgraph_invoker():
    return main()

def pretty_format(answer_dict):
    answer = answer_dict.get("answer", "")
    return answer.replace('\n', '\n\n')

langgraph_invoker = get_langgraph_invoker()

with st.form("question_form"):
    user_input = st.text_input("🔍 Type your question here", placeholder="e.g., What drugs are similar to Cetirizine?")
    submit = st.form_submit_button("Submit")

if submit and user_input.strip():
    with st.spinner("💡 Thinking..."):
        try:
            result = langgraph_invoker.invoke({"question": user_input})
            formatted_answer = pretty_format(result)
            st.markdown("### ✅ Answer")
            st.markdown(formatted_answer)

            cypher = result.get("cypher_statement", "")
            if cypher:
                with st.expander("⚙️ Show Cypher Query"):
                    st.code(cypher, language="cypher")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
