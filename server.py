"""
Loan Voice Agent - OpenAI Agents SDK edition (browser-based)

Browser mic --WebSocket--> FastAPI --RealtimeSession--> OpenAI Realtime API
Browser speaker <-WebSocket-- FastAPI <-- agent audio + transcripts

Run:  uvicorn server:app --reload     then open http://localhost:8000
"""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agents import Agent, Runner
from agents.guardrail import GuardrailFunctionOutput, OutputGuardrail
from agents.realtime import RealtimeAgent, RealtimeRunner

GUARDRAIL_LOG = Path(__file__).with_name("guardrail_trips.txt")
guardrail_logger = logging.getLogger("loan_guardrail")
guardrail_logger.setLevel(logging.WARNING)
if not guardrail_logger.handlers:
    guardrail_handler = logging.FileHandler(GUARDRAIL_LOG, encoding="utf-8")
    guardrail_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    guardrail_logger.addHandler(guardrail_handler)
guardrail_logger.propagate = False

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY not set - put it in .env or export it.")

# ---------------------------------------------------------------
# 1) The agent's brain - edit these for your bank
# ---------------------------------------------------------------
BANK_NAME = "Apex Bank"

IN_SCOPE = [
    "Only speak in english",
    "Loan products offered: personal loans",
    "Interest rates: current starting rates and how interest is calculated",
    "Eligibility criteria: age, minimum income, employment status, and credit score requirements",
    "Documentation required to apply: ID proof, address proof, income proof, and bank statements",
    "Fees and charges: processing fees, prepayment/foreclosure charges, and late-payment penalties",
    "Loan amounts, minimum and maximum limits, and general loan terms and conditions",
    "Co-applicant, guarantor, and collateral/security requirements",
    "Balance transfer and loan refinancing options",
    "Questions about minimum credit scores for personal-loan eligibility are on topic, provide specific credit score requirments if available"
]

EXCLUDED = [
    "Loans other than personal loans",
    "Investment, trading, or wealth-management advice",
    "Approving, rejecting, or guaranteeing any loan",
    "Legal, tax advice",
    "General chit-chat, trivia, or any questions unrelated to bank loans",
    "Branch timings, ATM locations, careers, or HR-related questions",
    "Unrelated questions about technical support, website issues, or mobile app problems",
    "Any other trivia, entertainment, or off-topic questions not related to bank loans",
]

in_scope_list = "\n".join(f"  - {item}" for item in IN_SCOPE)
excluded_list = "\n".join(f"  - {item}" for item in EXCLUDED)

INSTRUCTIONS = f"""You are the Loan Assistant for {BANK_NAME}.
You are a voice agent answering calls from customers.

Your ONLY purpose is to answer the caller's questions about the bank's LOAN products and loan-related topics:

IN SCOPE - you MAY discuss:
{in_scope_list}

OUT OF SCOPE - you must NOT discuss:
{excluded_list}

RULES:
- Answer ONLY loan-related questions. If the caller asks about anything on the
  excluded list or any other topic, politely say you can only help with
  loan-related questions, and offer to list the loan topics you can assist with.
- Never invent rates, fees, or policies that are not in this prompt. If you
  don't know a specific figure, say it varies and the caller should confirm
  with a loan officer or the bank's website.
- Never ask for or repeat sensitive personal data (full account numbers, PINs,
  passwords). If the caller provides such data, tell them not to share it.
- Keep answers short (1-3 sentences) and natural for speech.
- Be polite and professional at all times.
"""

GREETING_PROMPT = (
    f"Greet the caller as the {BANK_NAME} Loan Assistant in one short sentence, "
    "then briefly state which loan topics you can help with and mention that "
    "you can only answer loan-related questions."
)

# ---------------------------------------------------------------
# 2) Off-topic guardrail
#    Realtime guardrails run on the agent's OUTPUT transcript. A small
#    classifier agent checks every reply; if it is off-topic, the
#    guardrail trips and the SDK interrupts the response.
# ---------------------------------------------------------------
class ScopeCheck(BaseModel):
    on_topic: bool
    reason: str


scope_checker = Agent(
    name="Loan scope checker",
    model="gpt-4o-mini",
    instructions=(
        f"You review replies from a {BANK_NAME} loan voice assistant. "
        "Set on_topic=true for replies that discuss any in-scope personal-loan topic, "
        "including loan products, rates, eligibility and credit-score requirements, "
        "documentation, processing fees, prepayment or foreclosure charges, "
        "late-payment penalties, loan limits and terms, guarantors or collateral, "
        "balance transfers, and refinancing. These topics are on-topic even when the "
        "reply gives general information rather than a bank-specific figure. Also set "
        "on_topic=true for a greeting or a polite refusal/redirect to loan topics. "
        "Set on_topic=false only when the reply gives substantive information about "
        "an excluded or unrelated topic, or promises loan approval or a guaranteed "
        "personalized rate.\n"
        f"Excluded topics:\n{excluded_list}"
    ),
    output_type=ScopeCheck,
)


async def loan_scope_guardrail(context, agent, output: str) -> GuardrailFunctionOutput:
    result = await Runner.run(scope_checker, output)
    check: ScopeCheck = result.final_output
    if not check.on_topic:
        guardrail_logger.warning("Guardrail triggered: %s", check.reason)
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=not check.on_topic,
    )


# ---------------------------------------------------------------
# 3) Agent + Runner
# ---------------------------------------------------------------
agent = RealtimeAgent(name="Loan Assistant", instructions=INSTRUCTIONS)

runner = RealtimeRunner(
    starting_agent=agent,
    config={
        "model_settings": {
            "model_name": "gpt-realtime",
            "voice": "alloy",
            "modalities": ["audio"],
            "input_audio_format": "pcm16",   # 24 kHz, 16-bit mono
            "output_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "whisper-1",
                "language": "en",
            },
            "turn_detection": {"type": "server_vad", "interrupt_response": True},
        },
        "output_guardrails": [OutputGuardrail(guardrail_function=loan_scope_guardrail)],
        "guardrails_settings": {"debounce_text_length": 80},
    },
)

# ---------------------------------------------------------------
# 4) Web server
# ---------------------------------------------------------------
app = FastAPI()
INDEX = Path(__file__).parent / "static" / "index.html"


@app.get("/")
async def index():
    return FileResponse(INDEX)


def extract_messages(history) -> list[dict]:
    """Turn the SDK's history items into simple {role, text} dicts for the UI."""
    messages = []
    for item in history:
        if getattr(item, "type", None) != "message":
            continue
        text = ""
        for part in getattr(item, "content", None) or []:
            text += getattr(part, "text", None) or getattr(part, "transcript", None) or ""
        text = text.strip()
        if text and text != GREETING_PROMPT:
            messages.append({"id": item.item_id, "role": item.role, "text": text})
    return messages


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session = await runner.run()

    async with session:
        await session.send_message(GREETING_PROMPT)  # agent speaks first

        async def browser_to_agent():
            """Mic audio (binary frames) from the browser -> the model."""
            while True:
                data = await ws.receive_bytes()
                await session.send_audio(data)

        async def agent_to_browser():
            """Events from the model -> the browser."""
            async for event in session:
                if event.type == "audio":
                    await ws.send_bytes(event.audio.data)
                elif event.type == "audio_interrupted":
                    await ws.send_json({"type": "interrupted"})
                elif event.type == "history_updated":
                    await ws.send_json(
                        {"type": "history", "messages": extract_messages(event.history)}
                    )
                elif event.type == "guardrail_tripped":
                    await ws.send_json({"type": "interrupted"})
                    reasons = [
                        result.output.output_info.reason
                        for result in event.guardrail_results
                        if isinstance(
                            getattr(result.output, "output_info", None), ScopeCheck
                        )
                        and result.output.output_info.reason.strip()
                    ]
                    reason = "; ".join(reasons) or (
                        "The reply was classified as outside the allowed loan topics."
                    )
                    await ws.send_json({
                        "type": "notice",
                        "text": f"Off-topic reply blocked. Reason: {reason}",
                    })
                elif event.type == "error":
                    await ws.send_json({"type": "error", "text": str(event.error)})

        tasks = [
            asyncio.create_task(browser_to_agent()),
            asyncio.create_task(agent_to_browser()),
        ]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except WebSocketDisconnect:
            pass
        finally:
            for t in tasks:
                t.cancel()
