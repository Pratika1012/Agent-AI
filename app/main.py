import streamlit as st
from agents.llm_orchestration import LLMOrchestrator
from agents.research_agent import ResearchAgent
from Utils.token_counter import count_tokens
import os
import logging
from Utils.logger import setup_logger
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from agents.llm_orchestration import LLMOrchestrator

# ✅ Load Pinecone credentials from Streamlit secrets
pinecone_api = st.secrets["api_keys"].get("pinecone", "MISSING")
pinecone_env = st.secrets["pinecone_config"].get("environment", "MISSING")
pinecone_index = st.secrets["pinecone_config"].get("index_name", "MISSING")

print(f"Pinecone API Key: {pinecone_api}")
print(f"Pinecone Environment: {pinecone_env}")
print(f"Pinecone Index Name: {pinecone_index}")

# ✅ Correct config path
logger = setup_logger()
logger.info("✅ Logger setup completed successfully.")

# ✅ Initialize LLM Orchestrator and Research Agent
orchestrator = LLMOrchestrator()
research_agent = ResearchAgent()

# ✅ Streamlit UI
st.title("Agentic AI LLM Orchestration 🚀")
st.write("Efficiently route queries to different models with memory storage (Pinecone) & Research Agent.")

# ✅ Sidebar Mode Selection
with st.sidebar:
    st.header("Choose Mode 🎯")
    mode = st.radio(
        "Select mode:",
        ("LLM Orchestration", "View Memory", "Research Agent")
    )

# ✅ User Query Input
user_query = st.text_area("Enter your query here:", height=150)

if st.button("Generate Response"):
    if not user_query.strip():
        st.error("❌ Please enter a valid query.")
    else:
        with st.spinner("🔄 Generating response..."):

            # ✅ Mode: LLM Orchestration
            if mode == "LLM Orchestration":
                response = orchestrator.generate_response(user_query)
                model_used = orchestrator.select_model(user_query)

            # ✅ Mode: Research Agent
            elif mode == "Research Agent":
                response, model_info = research_agent.generate_response(user_query)

            # ✅ Display AI Response
            st.markdown("### AI Response ✅")
            st.write(response if response else "⚠️ No response generated.")

            st.markdown("---")

        # ✅ Token Usage & Cost Calculation
        if mode == "LLM Orchestration":
            input_tokens = count_tokens(user_query, model_used)
            output_tokens = count_tokens(response, model_used)
            total_tokens = input_tokens + output_tokens

            cost_per_1000_tokens = st.secrets["pricing_per_1000_tokens"].get(model_used, 0.01)
            total_cost = (total_tokens / 1000) * cost_per_1000_tokens

            st.markdown("### Token Usage & Cost 💰")
            st.write(f"**Model Used:** {model_used}")
            st.write(f"**Input Tokens:** {input_tokens}")
            st.write(f"**Output Tokens:** {output_tokens}")
            st.write(f"**Total Tokens:** {total_tokens}")
            st.write(f"**Estimated Cost:** ${total_cost:.4f}")

        else:
            st.markdown("### Token Usage & Cost 💰")
            st.write(f"**Model Used:** {model_info['model']}")
            st.write(f"**Input Tokens:** {model_info['total_input_tokens']}")
            st.write(f"**Output Tokens:** {model_info['total_output_tokens']}")
            st.write(f"**Total Tokens:** {model_info['total_tokens']}")

# ✅ Memory Retrieval UI (Using Pinecone)
if mode == "View Memory":
    st.markdown("## View Past Interactions 🔍")
    memory_query = st.text_input("Enter a query to check past responses:")

    if st.button("Retrieve Memory"):
        past_responses = orchestrator.memory.retrieve_similar(memory_query, k=3)  # Adjust `k` as needed
        if past_responses:
            st.markdown("### Retrieved Responses from Pinecone 🔄")
            for idx, response in enumerate(past_responses, 1):
                st.write(f"**Memory {idx}:** {response}")
        else:
            st.write("⚠️ No past responses found.")
