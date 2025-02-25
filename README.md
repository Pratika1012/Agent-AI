# Agentic AI Research Agent

## Overview

This project implements an AI-powered research agent designed to retrieve, analyze, and summarize information while optimizing multi-step reasoning, memory retention, and efficient orchestration of large language models (LLMs). It leverages advanced prompting styles and integrates with Groq Cloud for multi-LLM orchestration. An optional vector database integration allows for memory retention across sessions.

## Project Structure

- **src/**: Contains the source code for the project.
  - **agent/**: Core agent logic including dynamic prompting and LLM orchestration.
  - **memory/**: Vector DB integration for memory retention.
  - **utils/**: Utility modules for logging and token counting.
- **logs/**: Directory where runtime logs are stored.
- **config.json**: Configuration file for storing sensitive keys (e.g., API key).
- **requirements.txt**: List of required Python packages.
- **README.md**: This file.

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/agentic-ai-research-agent.git
   cd agentic-ai-research-agent
