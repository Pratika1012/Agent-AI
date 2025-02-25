# app/main.py

import streamlit as st
from agents.research_agent import ResearchAgent

# Initialize your agent
agent = ResearchAgent(config_path="config/Configration.json")

st.title("Agentic AI Research Agent 🧠")
st.write("This agent demonstrates multi-step reasoning using various prompting techniques.")

# Dropdown for prompting techniques
with st.sidebar:
    st.header("Prompting Techniques 🎯")
    prompting_style = st.selectbox(
        "Choose a Prompting Technique",
        ("Chain-of-Thought", "Tree-of-Thought", "Self-Consistency")
    )

# Displaying example queries based on selected prompting style
example_queries = {
    "Chain-of-Thought": "Summarize the key points from OpenAI’s latest research paper on multimodal models.",
    "Tree-of-Thought": "Compare and contrast OpenAI’s GPT models with Anthropic’s Claude models in terms of efficiency and alignment strategies.",
    "Self-Consistency": "If I want a model that prioritizes safety, should I choose GPT-4 or Claude 2?"
}

selected_example = example_queries[prompting_style]

# Display example query clearly for user guidance
st.markdown(f"**Example Query ({prompting_style}):**")
st.info(selected_example)

# User input area, pre-populated with the example
user_query = st.text_area("Enter your query here:", value=selected_example, height=150)

if st.button("Generate Response"):
    if user_query.strip() == "":
        st.error("Please enter a valid query.")
    else:
        with st.spinner("Generating refined response..."):
            # Temporarily override the dynamic prompting method to use selected style
            original_method = agent.choose_prompt_style
            agent.choose_prompt_style = lambda x: prompting_style.lower().replace("-", "_")
            
            response = agent.generate_response(user_query)

            # Restore original method after generation
            agent.choose_prompt_style = original_method
        
        st.markdown("### Refined Response ✅")
        st.write(response)
