# Symptex

A chatbot designed for medical students, simulating doctor-patient interactions with the goal of improving the user's medical
history-taking skills.

## Prerequisites

- [Docker](https://docs.docker.com/get-started/get-docker/)
- Running ILuVI PostgreSQL database (see ILuVI repository)
- Browser of your choice to interact with Symptex
- Access to an API key for the [KISSKI ChatAI service](https://kisski.gwdg.de/leistungen/2-02-llm-service/)

## Getting Started

1. In the `api/chains/` folder, create an `.env` file to define and store the following required (sensitive) keys:

```env
CHATAI_API_URL=https://chat-ai.academiccloud.de/v1
CHATAI_API_KEY=insert_api_key
# For testing, optional:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=insert_langsmith_key
```
2. In the root folder create an `.env` file to contain the folder where the Befunde are. The corresponding variable is:
```env
HOST_ANAMNESIS_PATH={path to Befunde}
```
3. Run `docker compose up --build` in the project's root directory.
4. Interact with Symptex locally through [Streamlit frontend URL](http://localhost:8501).

## Endpoints

- Streamlit frontend: <http://localhost:8501>
- API: <http://localhost:8000>

## Features

- Simulation of multiple patient conditions in the context of medical history-taking: default, alzheimer, schwerhörig (hearing impairment), verdrängung (denial of symptoms)
- Configurable patient talkativeness levels/verbosity: kurz angebunden, ausgewogen, ausschweifend
- Provision of performance feedback for increased pedagogical value
- Multiple LLM models supported (see [KISSKI ChatAI models](https://docs.hpc.gwdg.de/services/saia/index.html))
- Chat session management through ILuVI PostgreSQL database
- ILuVI Patient file integration

## Project Structure

```
symptex/
│
├── api/
│   ├── app/                      # API logic
│   │   ├── main.py               # FastAPI entry point
│   │   ├── db/                   # Database models and connection
│   │   │   ├── db.py             # Database configuration
│   │   │   └── models.py         # SQLAlchemy models
│   │   └── routers/
│   │       └── chat.py           # Chat-specific routes
│   │
│   ├── chains/                   # Chain logic
│   │   ├── chat_chain.py         # Main chat chain definition
│   │   ├── eval_chain.py         # Evaluation chain for feedback
│   │   ├── prompts.py            # Behavior prompts for different conditions
│   │   ├── patient_data.py       # Patient data definitions for testing
│   │   └── formatting.py         # Patient data formatting utilities
│   │
│   ├── tests/                    # Test files
│   │
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── frontend.py               # Streamlit frontend
│   ├── requirements.txt          # Dependencies for Streamlit frontend
│   ├── assets/                   # Frontend assets (images, etc.)
│   └── Dockerfile
│
├── docker-compose.yml
└── README.md
```

TODOs: Connect the upload of Befunde with ILVI's backend