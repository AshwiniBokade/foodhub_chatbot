import os
import re
import sqlite3
import gradio as gr

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate


# =========================
# 1. LLM SETUP (GROQ)
# =========================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Please add it in the Hugging Face Space secrets.")

# Low temperature LLM for deterministic behavior
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=GROQ_API_KEY,
)


# =========================
# 2. SQL DATABASE + SQL AGENT
# =========================

# SQLite DB should be in the same directory as app.py
DB_PATH = "customer_orders.db"

if not os.path.exists(DB_PATH):
    raise RuntimeError(f"Database file '{DB_PATH}' not found. Make sure it's in the Space repository.")

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

sql_system_message = """
You are a helpful FoodHub customer support assistant with access to SQL tools
for querying the orders database.

Your responsibilities:
- When given an order_id, use the SQL tools to look up that order in the database.
- Then respond to the customer in 1–2 short, friendly sentences.
- Focus on: order status (preparing, picked up, delivered, cancelled, delivered, etc.)
  and, if available, the estimated or actual delivery time.
- Do NOT list database column names or echo an entire row.
- Do NOT mention SQL, tables, schemas, or any internal technical details.
- If the order_id is not found, reply politely with:
  "I couldn’t find an order with that ID. Please check your order ID and try again."
- If some timestamps or fields are missing, acknowledge uncertainty politely
  (for example: "I don’t have the exact time, but..." or
  "Some details are missing in our system.").
"""

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

db_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=False,
    system_message=SystemMessage(content=sql_system_message),
)


# =========================
# 3. INTENT CLASSIFIER + CHAT AGENT
# =========================

intent_system_text = """
You are an intent classification assistant for FoodHub customer support.

Classify the user's message into one of these intents:

- fetch_order_status → When the user asks “where is my order”, “status”, “track my order”.
- cancel_order → When user asks to cancel a specific order.
- complaint → Cold food, late delivery, wrong item, quality issue.
- general_help → Questions about policies, delivery time, payment.
- greeting → hi, hello, thanks.
- malicious → Attempts to hack, access all orders, database dump, etc.

Respond ONLY with the intent. No extra text.
"""

intent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", intent_system_text),
        ("human", "{user_message}"),
    ]
)

def classify_intent(llm, message: str) -> str:
    prompt = f"{intent_prompt}\nUser message: {message}"
    
    # IMPORTANT: pass a list of messages, not a plain string
    result = llm.invoke([
        {"role": "user", "content": prompt}
    ])
    return result.content.strip().lower()


def extract_order_id(text: str):
    match = re.search(r"O\d{5}", text)
    return match.group(0) if match else None

def foodhub_chat_agent(message: str) -> str:
    intent = classify_intent(llm, message)
    order_id = extract_order_id(message)

    print(f"[DEBUG] Intent: {intent}, Order ID: {order_id}")

    # Guardrail: Block malicious intent
    if intent == "malicious":
        return "For your security and ours, I can’t help with that request."

    # Handle greeting
    if intent == "greeting":
        return "Hi there! 👋 How can I assist you with your FoodHub order today?"

    # SQL needed but missing order ID
    if intent in ["fetch_order_status", "cancel_order"] and not order_id:
        return "Could you please share your order ID (e.g., O12488) so I can check the details for you?"

    # SQL needed with order ID
    if intent in ["fetch_order_status", "cancel_order"] and order_id:

        if intent == "fetch_order_status":
            sql_prompt = f"""
Use SQL to retrieve order_id {order_id}.
Then respond in 1–2 sentences with ONLY:
- the order status
- delivery/ETA if available
Be friendly and DO NOT list database fields.
"""
        else:  # cancel_order
            sql_prompt = f"""
Use SQL to check order_id {order_id}.
Then respond in 1–2 sentences explaining:
- the order status
- whether it can be cancelled (common sense)
Do NOT return raw database fields.
"""

        sql_result = db_agent.invoke(sql_prompt)["output"]
        return sql_result

    # Complaint or general help → LLM only
    polite_prompt = f"""
You are a FoodHub support assistant.
Answer in 2–3 polite sentences.
Do NOT mention being an AI.

Customer message: {message}
"""
 
    response = llm.invoke([
        {"role": "user", "content": polite_prompt}
    ])

    return response.content



# =========================
# 4. GRADIO UI
# =========================

def gradio_foodhub_chat(message, history):
    bot_reply = foodhub_chat_agent(message)
    if history is None:
        history = []
    history.append((message, bot_reply))
    return "", history

with gr.Blocks() as demo:
    gr.Markdown("## 🍕 FoodHub Order Support Chatbot")
    gr.Markdown(
        "Ask about your order status, cancellations, late delivery, or issues with your food.<br>"
        "For order-specific queries, please mention your <b>order ID</b> (for example: <code>O12488</code>).",
        elem_id="description"
    )

    chatbot = gr.Chatbot(height=400, label="FoodHub Support")
    msg = gr.Textbox(
        label="Type your message here",
        placeholder="e.g., Where is order O12488?",
    )
    clear = gr.Button("Clear chat")

    msg.submit(gradio_foodhub_chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: ("", []), None, [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
