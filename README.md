# RedGuard AI — LLM Guardrail Chatbot

A hands-on implementation of AI guardrails using 
NeMo Guardrails + LangChain + Ollama (local LLM)

## Features
- Jailbreak Protection
- Prompt Injection Blocking
- Harmful Content Filtering
- Toxic Input Detection
- Sensitive Data Output Protection
- 100% local — no API costs

## Stack
- NeMo Guardrails 0.20.0
- LangChain + LangChain-Ollama
- Ollama (deepseek-llm:7b)
- Python 3.11

## Test Results
26/26 tests passed — 100% guardrail coverage

## Setup
pip install -r requirements.txt
ollama pull deepseek-llm:7b
python app.py

## Author
Security Researcher — RedGuard AI Project
