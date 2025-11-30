# T-bot (remake 2025)

T-bot is a modern remake of a python AI project "thinkebot" made in 2021
The original thinkerbot could analyze sentences using hardcoded parts of speech and grammar rules
acquiring and storing word and phrase objects into objects and CSV files and develop 
a semantical meaning and logically assemble them into new sentence objects 

## Overview

`T-bot 2025` is remake using a pipeline that combines LLM-based linguistic analysis with vector embeddings and lightweight mathematical reasoning. Because we are using Agent Ollama embeddings instead of hardcoded rules the logic system has more capability:

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

## Repository Prep for GitHub

Included files to prepare this project for GitHub:

- `.gitignore` — ignores virtual environments, caches, and local data files
- `requirements.txt` — captures runtime dependencies
- `LICENSE` — MIT license (default; change if you prefer another license)

## Recommendations before publishing

- Review `SYSTEM_PROMPT` in `ollama_module.py` to ensure it contains no sensitive or proprietary content.
- Decide whether to include the knowledge base (`tbot_analysis_data.json`) in the repo — typically it's better to keep this out of version control. The provided `.gitignore` excludes it by default.
- Add unit tests (mocking LLM and embedding calls) so CI can validate behavior without depending on external services.
- Consider adding a GitHub Actions workflow for linting and running tests on push.

## Next steps I can take

- Add a basic `ci.yml` GitHub Actions workflow for lint/tests.
- Initialize a local Git repository and make the first commit.
- Add unit tests and a tiny test harness that mocks `ollama` calls.


Tell me which of the above you want me to do next.
