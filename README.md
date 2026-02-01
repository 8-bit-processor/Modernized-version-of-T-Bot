# T-bot (remake 2025)

T-bot is a modern remake of a python AI project "thinkebot" made in 2021

*** This is Unfinished- I am not working on this now but I will come back someday to implement propositional logic and more ***

The original thinkerbot used hardcoded parts of speech and grammar rules to develop semantical meaning 
acquiring and storing word and phrases using object oriented programming and CSV files. With zero dependencies and could be
run offline it used hardcoded scripting and logically assembled them into new sentence objects.  

## Overview

`T-bot 2025` is remake using a pipeline that combines LLM-based linguistic analysis with vector embeddings and lightweight mathematical reasoning. Because we are using Agent Ollama embeddings instead of hardcoded rules the logic system has more autonomous capability:

- Instead of a scripted realm of statements, it accepts any natural language input
- Instead of loops and hardcoded grammar rules it extracts structured linguistic objects and explicit facts (including conditional implications)
- Instead of CSV files it Generates embeddings for inputs and persists analyses into a local knowledge base (`tbot_analysis_data.json`)
- More robustly applies semantic and logical reasoning (similarity checks, implication extraction, modus ponens/tollens)

This repository contains the same concepts used in the original t-bot for building a semantic reasoning assistant except that becuase we are using Ollama(LLM) t-bot 2025 more capably bridges that barrier between natural-language structure with symbolic-style logic.

## Quick Start

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the application:

```powershell
python main.py
```

## Notes

- The project expects a local Ollama server for chat/embeddings. Ensure Ollama is running and accessible at its default API endpoint if you use the provided `ollama` integration.
- The knowledge base (`tbot_analysis_data.json`) persists analyses; for testing you can clear it to start fresh.
