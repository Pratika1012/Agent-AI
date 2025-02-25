# # app/agent/research_agent.py

# import os
# import json
# import time
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from Utils.logger import setup_logger
# from Utils.token_counter import count_tokens

# #CONFIG_PATH = r"C:\Users\HP\Desktop\Agent-AI\config\Configratution.json"

# class ResearchAgent:
#     def __init__(self, CONFIG_PATH=r"C:\Users\HP\Desktop\Agent-AI\config\Configratution.json"):
#         # Logger setup
#         self.logger = setup_logger()

#         with open(CONFIG_PATH, "r") as f:
#             self.config = json.load(f)  # ✅ Store config in self.config

#         os.environ["OPENAI_API_KEY"] = self.config["api_keys"]["openai"]

#         self.model_name = self.config["models"]
#         self.llm = ChatOpenAI(
#             models="gpt-4o",
#             temperature=self.config["generation_config"]["gpt-4o"]["temperature"],
#             max_tokens=self.config["generation_config"]["gpt-4o"]["max_tokens"],
#             top_p=self.config["generation_config"]["gpt-4o"]["top_p"]
#         )

#         # Prompt Templates (unchanged)
#         self.chain_of_thought_template = PromptTemplate(
#             input_variables=["query"],
#             template="Let's think step by step to answer:\n{query}\n\nStep 1:"
#         )

#         self.self_consistency_template = PromptTemplate(
#             input_variables=["query"],
#             template="Provide multiple perspectives for:\n{query}\n\nAnswer 1:"
#         )

#         self.tree_of_thought_template = PromptTemplate(
#             input_variables=["query"],
#             template="Break down into sub-questions and synthesize response:\n{query}\n\nSub-question 1:"
#         )

#         self.refinement_template = PromptTemplate(
#             input_variables=["query", "initial_response"],
#             template=(
#                 "Based on previous answer, refine response:\n\n"
#                 "Query: {query}\n\n"
#                 "Initial Answer: {initial_response}\n\n"
#                 "Refined Answer:"
#             )
#         )

#     def choose_prompt_style(self, query: str) -> str:
#         classification_prompt = f"""
#         Classify the following query into one of the three categories based on complexity and intent:

#         1. Tree-of-Thought: Queries involving comparisons, contrasts, or multiple sub-topics.
#         2. Self-Consistency: Queries requiring a decision-making process, weighing pros and cons, or providing multiple perspectives.
#         3. Chain-of-Thought: General queries requiring step-by-step reasoning or explanations.

#         Clearly respond with ONLY the category name (Tree-of-Thought, Self-Consistency, or Chain-of-Thought).

#         Query: "{query}"

#         Category:
#         """

#         style_response = self.llm.predict(classification_prompt).strip().lower()

#         if "tree-of-thought" in style_response:
#             return "tree_of_thought"
#         elif "self-consistency" in style_response:
#             return "self_consistency"
#         elif "chain-of-thought" in style_response:
#             return "chain_of_thought"
#         else:
#             self.logger.warning("Unable Style:'{style_response}'. Defaulting to Chain-of-Thought.")
#             return "chain_of_thought"
    
    


#     def generate_initial_response(self, query: str) -> str:
#         style = self.choose_prompt_style(query)
#         chain = LLMChain(llm=self.llm, prompt=getattr(self, f"{style}_template"))

#         # Track tokens and time
#         start_time = time.time()
#         response = chain.run(query=query)
#         duration = time.time() - start_time

#         input_tokens = count_tokens(query, self.model_name)
#         output_tokens = count_tokens(response, self.model_name)
#         total_tokens = input_tokens + output_tokens

#         # Logging details
#         self.logger.info(
#                 f"[Initial Response]\n"
#                 f"• Prompting Style: {style}\n"
#                 f"• Input Tokens: {input_tokens}\n"
#                 f"• Output Tokens: {output_tokens}\n"
#                 f"• Total Tokens: {total_tokens}\n"
#                 f"• Time Taken: {duration:.2f} seconds\n"
# )

#         return response

#     def refine_response(self, query: str, initial_response: str) -> str:
#         chain = LLMChain(llm=self.llm, prompt=self.refinement_template)

#         start_time = time.time()
#         refined_response = chain.run(query=query, initial_response=initial_response)
#         duration = time.time() - start_time

#         combined_input = query + initial_response
#         input_tokens = count_tokens(combined_input, self.model_name)
#         output_tokens = count_tokens(refined_response, self.model_name)
#         total_tokens = input_tokens + output_tokens

#         self.logger.info(
#                 f"[Refined Response]\n"
#                 f"• Input Tokens: {input_tokens}\n"
#                 f"• Output Tokens: {output_tokens}\n"
#                 f"• Total Tokens: {total_tokens}\n"
#                 f"• Time Taken: {duration:.2f} seconds\n"
# )
#         return refined_response

#     def generate_response(self, query: str) -> str:
#         initial_response = self.generate_initial_response(query)
#         refined_response = self.refine_response(query, initial_response)
#         return refined_response
