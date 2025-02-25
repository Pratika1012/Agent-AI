# import streamlit as st
# # from agents.research_agent import ResearchAgent
# from agents.llm_orchestration import LLMOrchestrator
# from Utils.token_counter import count_tokens

# import os

# CONFIG_PATH = r"C:\Users\HP\Desktop\Agent-AI\config\Configratution.json"

# # Initialize Agents
# # agent = ResearchAgent(CONFIG_PATH=CONFIG_PATH)

# orchestrator = LLMOrchestrator(CONFIG_PATH=CONFIG_PATH)


# st.title("Agentic AI Research Agent & LLM Orchestration 🧠🚀")
# st.write("Choose whether to use multi-step reasoning or LLM orchestration for Groq Cloud.")

# # Sidebar Selection
# with st.sidebar:
#     st.header("Choose Your Mode 🎯")
#     mode = st.radio(
#         "Select a mode:",
#         ("Research Agent", "LLM Orchestration (Groq Cloud)")
#     )

#     st.markdown("---")

#     if mode == "Research Agent":
#         prompting_style = st.selectbox(
#             "Choose a Prompting Technique",
#             ("Chain-of-Thought", "Tree-of-Thought", "Self-Consistency")
#         )

# user_query = st.text_area("Enter your query here:", height=150)

# if st.button("Generate Response"):
#     if user_query.strip() == "":
#         st.error("Please enter a valid query.")
#     else:
#         with st.spinner("Generating response..."):
#             if mode == "Research Agent":
#                 original_method = agent.choose_prompt_style
#                 agent.choose_prompt_style = lambda x: prompting_style.lower().replace("-", "_")
#                 response = agent.generate_response(user_query)
#                 agent.choose_prompt_style = original_method
#                 model_used = agent.model_name
#             else:
#                 response = orchestrator.generate_response(user_query)
#                 model_used = orchestrator.select_model(user_query)

#         st.markdown("### AI Response ✅")
#         st.write(response)

#         st.markdown("---")

#         # Token Calculation & Cost Display
#         input_tokens = count_tokens(user_query, model_used)
#         output_tokens = count_tokens(response, model_used)
#         total_tokens = input_tokens + output_tokens
#         cost_per_1000_tokens = orchestrator.config.get("pricing_per_1000_tokens", {}).get(model_used, 0.01)
#         total_cost = (total_tokens / 1000) * cost_per_1000_tokens

#         st.markdown("### Token Usage & Cost 💰")
#         st.write(f"**Model Used:** {model_used}")
#         st.write(f"**Input Tokens:** {input_tokens}")
#         st.write(f"**Output Tokens:** {output_tokens}")
#         st.write(f"**Total Tokens:** {total_tokens}")
#         st.write(f"**Estimated Cost:** ${total_cost:.4f}")



# import streamlit as st
# from agents.llm_orchestration import LLMOrchestrator
# from Utils.token_counter import count_tokens
# import os
# import requests  # Import if needed for debugging API responses

# # ✅ Correct config path
# CONFIG_PATH = r"C:\Users\HP\Desktop\Agent-AI\config\Configratution.json"

# # ✅ Initialize LLM Orchestrator
# orchestrator = LLMOrchestrator(CONFIG_PATH=CONFIG_PATH)

# # ✅ Streamlit UI
# st.title("Agentic AI LLM Orchestration 🚀")
# st.write("Efficiently route queries to different models on Groq Cloud.")

# # ✅ Sidebar Mode Selection
# with st.sidebar:
#     st.header("Choose Mode 🎯")
#     mode = st.radio(
#         "Select mode:",
#         ("LLM Orchestration (Groq Cloud)",),  # Removed "Research Agent"
#     )

# user_query = st.text_area("Enter your query here:", height=150)

# if st.button("Generate Response"):
#     if not user_query.strip():
#         st.error("❌ Please enter a valid query.")
#     else:
#         with st.spinner("🔄 Generating response..."):
#             response = orchestrator.generate_response(user_query)
#             model_used = orchestrator.select_model(user_query)

#         # ✅ Display AI Response
#         st.markdown("### AI Response ✅")
#         st.write(response if response else "⚠️ No response generated.")

#         st.markdown("---")

#         # ✅ Token Usage & Cost Calculation
#         input_tokens = count_tokens(user_query, model_used)
#         output_tokens = count_tokens(response, model_used)
#         total_tokens = input_tokens + output_tokens

#         # ✅ Fix: Safe Extraction of `pricing_per_1000_tokens`
#         cost_per_1000_tokens = orchestrator.config.get("pricing_per_1000_tokens", {}).get(model_used, 0.01)


#         total_cost = (total_tokens / 1000) * cost_per_1000_tokens

#         # ✅ Display Token Stats
#         st.markdown("### Token Usage & Cost 💰")
#         st.write(f"**Model Used:** {model_used}")
#         st.write(f"**Input Tokens:** {input_tokens}")
#         st.write(f"**Output Tokens:** {output_tokens}")
#         st.write(f"**Total Tokens:** {total_tokens}")
#         st.write(f"**Estimated Cost:** ${total_cost:.4f}")


# import requests

# api_key = "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"
# url = "https://api.groq.com/openai/v1/chat/completions"

# payload = {
#     "model": "mixtral-8x7b-32768",
#     "messages": [{"role": "user", "content": "Write a Python script for web scraping"}],
#     "max_tokens": 512,
#     "temperature": 0.1
# }

# headers = {
#     "Authorization": f"Bearer {api_key}",
#     "Content-Type": "application/json"
# }

# response = requests.post(url, json=payload, headers=headers)
# print(response.status_code)
# print(response.json())  # Print full API response


# import streamlit as st
# from agents.llm_orchestration import LLMOrchestrator
# from Utils.token_counter import count_tokens
# import os
# import logging
# from Utils.logger import setup_logger
# # ✅ Correct config path
# CONFIG_PATH = r"C:\Users\HP\Desktop\Agent-AI\config\Configratution.json"

# logger = setup_logger()

# # ✅ Example Log
# logger.info("✅ Logger setup completed successfully.")
# # # ✅ Initialize LLM Orchestrator
# orchestrator = LLMOrchestrator(CONFIG_PATH=CONFIG_PATH)

# # ✅ Streamlit UI
# st.title("Agentic AI LLM Orchestration 🚀")
# st.write("Efficiently route queries to different models on Groq Cloud.")

# # ✅ Sidebar Mode Selection
# with st.sidebar:
#     st.header("Choose Mode 🎯")
#     mode = st.radio("Select mode:", ("LLM Orchestration (Groq Cloud)",))

# user_query = st.text_area("Enter your query here:", height=150)

# if st.button("Generate Response"):
#     if not user_query.strip():
#         st.error("❌ Please enter a valid query.")
#     else:
#         with st.spinner("🔄 Generating response..."):
#             response = orchestrator.generate_response(user_query)
#             model_used = orchestrator.select_model(user_query)

#         # ✅ Display AI Response
#         st.markdown("### AI Response ✅")
#         st.write(response if response else "⚠️ No response generated.")

#         st.markdown("---")

#         # ✅ Token Usage & Cost Calculation
#         input_tokens = count_tokens(user_query, model_used)
#         output_tokens = count_tokens(response, model_used)
#         total_tokens = input_tokens + output_tokens

#         cost_per_1000_tokens = orchestrator.config.get("pricing_per_1000_tokens", {}).get(model_used, 0.01)
#         total_cost = (total_tokens / 1000) * cost_per_1000_tokens

#         # ✅ Display Token Stats
#         st.markdown("### Token Usage & Cost 💰")
#         st.write(f"**Model Used:** {model_used}")
#         st.write(f"**Input Tokens:** {input_tokens}")
#         st.write(f"**Output Tokens:** {output_tokens}")
#         st.write(f"**Total Tokens:** {total_tokens}")
#         st.write(f"**Estimated Cost:** ${total_cost:.4f}")


import streamlit as st
from agents.llm_orchestration import LLMOrchestrator
from agents.vector_db import VectorDB
from Utils.token_counter import count_tokens
import os
import logging
from Utils.logger import setup_logger

# ✅ Correct config path
CONFIG_PATH = r"C:\Users\HP\Desktop\Agent-AI\config\Configration.json"

logger = setup_logger()
logger.info("✅ Logger setup completed successfully.")

# ✅ Initialize LLM Orchestrator and Vector DB
orchestrator = LLMOrchestrator(CONFIG_PATH=CONFIG_PATH)

st.title("Agentic AI LLM Orchestration 🚀")
st.write("Efficiently route queries to different models on Groq Cloud.")

# ✅ Sidebar Mode Selection
with st.sidebar:
    st.header("Choose Mode 🎯")
    mode = st.radio("Select mode:", ("LLM Orchestration (Groq Cloud)", "Memory Retrieval"))

user_query = st.text_area("Enter your query here:", height=150)

if st.button("Generate Response"):
    if not user_query.strip():
        st.error("❌ Please enter a valid query.")
    else:
        with st.spinner("🔄 Generating response..."):
            if mode == "Memory Retrieval":
                response = orchestrator.memory.retrieve_similar(user_query, k=1)
                response_text = response[0] if response else "⚠️ No past response found."
            else:
                response_text = orchestrator.generate_response(user_query)

            model_used = orchestrator.select_model(user_query)

        # ✅ Display AI Response
        st.markdown("### AI Response ✅")
        st.write(response_text)

        st.markdown("---")

        # ✅ Token Usage & Cost Calculation
        input_tokens = count_tokens(user_query, model_used)
        output_tokens = count_tokens(response_text, model_used)
        total_tokens = input_tokens + output_tokens
        cost_per_1000_tokens = orchestrator.config.get("pricing_per_1000_tokens", {}).get(model_used, 0.01)
        total_cost = (total_tokens / 1000) * cost_per_1000_tokens

        # ✅ Display Token Stats
        st.markdown("### Token Usage & Cost 💰")
        st.write(f"**Model Used:** {model_used}")
        st.write(f"**Input Tokens:** {input_tokens}")
        st.write(f"**Output Tokens:** {output_tokens}")
        st.write(f"**Total Tokens:** {total_tokens}")
        st.write(f"**Estimated Cost:** ${total_cost:.4f}")

# ✅ Memory Retrieval UI
st.sidebar.markdown("---")
st.sidebar.subheader("Memory Lookup 🔍")
memory_query = st.sidebar.text_input("Search Past Conversations")

if st.sidebar.button("Retrieve Memory"):
    with st.spinner("🔄 Searching memory..."):
        memory_response = orchestrator.memory.retrieve_similar(memory_query, k=1)
        st.sidebar.markdown("### Retrieved Response ✅")
        st.sidebar.write(memory_response[0] if memory_response else "⚠️ No past response found.")
