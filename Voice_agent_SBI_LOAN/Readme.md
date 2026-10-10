# SBI Loan Voice Agent

A browser-based voice assistant built with FastAPI and the OpenAI Agents SDK.

## Requirements

- An OpenAI API key
- A microphone and a  browser

## Run on Windows (PowerShell)

Open PowerShell in this folder, then create and activate a virtual environment
outside the repository:

```powershell
py -m venv $venv
& "$venv\Scripts\Activate.ps1"
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create or edit the local `.env` file in this folder and add your API key:

```text
OPENAI_API_KEY=your_openai_api_key
```

Do not share or commit `.env`. If a key was previously included in Git history
or exposed, revoke it and use a newly generated key.

Start the web server:

```powershell
python -m uvicorn server:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser, allow
microphone access, and start the call. Stop the server with `Ctrl+C`.

