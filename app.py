import os
import re
import sqlite3
from typing import Optional, Dict, Any

import gradio as gr
from groq import Groq

# =====================================================
# 1. LLM SETUP (GROQ / LLAMA 3.3 70B)
# =====================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Please add it in your Hugging Face Space "
        "Settings → Variables and secrets."
    )

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.3-70b-versatile"


def groq_chat(system_text: str, user_text: str, temperature: float = 0.2) -> str:
    """
    Helper to call Groq chat completions with system + user messages.
    """
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
    )
    return resp.choices[0].message.content.strip()


# =====================================================
# 2. SQLITE DATABASE UTILITIES
# =====================================================

DB_PATH = "customer_orders.db"

if not os.path.exists(DB_PATH):
    raise RuntimeError(
        f"Database file '{DB_PATH}' not found. Make sure it is in the Space repository."
    )


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_orders_table_name(conn) -> Optional[str]:
    """
    Auto-detect the orders table name (assumes a single main table).
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1;"
    )
    row = cur.fetchone()
    return row["name"] if row else None


def fetch_order_by_id(order_id: str) -> Optional[Dict[str, Any]]:
    """
    Return a dictionary for the given order_id, or None if not found.
    """
    conn = get_connection()
    try:
        table = get_orders_table_name(conn)
        if not table:
            return None
        cur = conn.cursor()
        cur.execute(
            f"SELECT * FROM {table} WHERE order_id = ?",
            (order_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        conn.close()


# =====================================================
# 3. INTENT CLASSIFICATION
# =====================================================

INTENT_SYSTEM_PROMPT = """
You are an intent classification assistant for FoodHub customer support.

You must classify the customer's message into exactly ONE of these intents:

- fetch_order_status  → “where is my order”, “track my order”, “status of order O123”
- cancel_order        → asking to cancel a specific order
- complaint           → cold food, late delivery, wrong item, quality issues, refund requests
- general_help        → questions about policies, delivery time ranges, payment, how app works
- greeting            → hi, hello, hey, thanks, good morning, etc.
- malicious           → hacking attempts, requests for all orders, all customer data, database dump

Return ONLY the intent string (e.g., fetch_order_status). No extra words.
"""


def classify_intent(message: str) -> str:
    user_text = f"Customer message: {message}\n\nIntent:"
    intent = groq_chat(INTENT_SYSTEM_PROMPT, user_text, temperature=0.0)
    return intent.strip().lower()


ORDER_ID_PATTERN = re.compile(r"O\\d{5}")


def extract_order_id(text: str) -> Optional[str]:
    match = ORDER_ID_PATTERN.search(text)
    return match.group(0) if match else None


# =====================================================
# 4. BUSINESS LOGIC FOR RESPONSES
# =====================================================

def build_status_message(order: Dict[str, Any]) -> str:
    status = str(order.get("order_status", "")).lower()
    delivery_eta = order.get("delivery_eta")
    delivery_time = order.get("delivery_time")

    if status == "delivered":
        base = "Your order has already been delivered."
        if delivery_time:
            base += f" It was delivered at {delivery_time}."
        return base

    if status in ("preparing food", "preparing"):
        base = "Your order is currently being prepared in the restaurant."
        if delivery_eta:
            base += f" It is expected to be delivered around {delivery_eta}."
        return base

    if status in ("picked up", "out for delivery"):
        base = "Your order has been picked up and is on its way to you."
        if delivery_eta:
            base += f" It should reach you around {delivery_eta}."
        return base

    if status in ("canceled", "cancelled"):
        return "This order has been cancelled."

    return f"The current status of your order is '{order.get('order_status', 'unknown')}'."


def can_order_be_cancelled(order: Dict[str, Any]) -> str:
    status = str(order.get("order_status", "")).lower()

    if status == "delivered":
        return (
            "This order has already been delivered, so it can no longer be cancelled. "
            "If there is an issue, you can raise a complaint in the app."
        )
    if status in ("canceled", "cancelled"):
        return "This order is already cancelled in our system."
    if status in ("picked up", "out for delivery"):
        return (
            "Your order has already been picked up by the delivery partner, "
            "so cancellation may not be possible at this stage."
        )

    # For preparing/placed/etc.
    return (
        "We can try to cancel this order as it is still being processed. "
        "Please confirm that you want to cancel, and we'll proceed."
    )


SUPPORT_GENERAL_SYSTEM_PROMPT = """
You are FoodHub Support Assistant for an online food delivery app.

Guidelines:
- Answer in 2–3 short, friendly sentences.
- Be clear, polite, and helpful.
- Do NOT say you are an AI model.
- Do NOT invent order details or data.
"""


def handle_general_or_complaint(message: str) -> str:
    user_text = f"Customer message: {message}"
    return groq_chat(SUPPORT_GENERAL_SYSTEM_PROMPT, user_text, temperature=0.4)


# =====================================================
# 5. MAIN CHAT CONTROLLER
# =====================================================

def foodhub_chat_agent(message: str, history=None) -> str:
    """
    Main function used by Gradio ChatInterface.
    We ignore `history` here, since we handle each message independently.
    """
    message = message.strip()
    if not message:
        return "Please type a message so I can help you with your order."

    intent = classify_intent(message)
    order_id = extract_order_id(message)

    print(f"[DEBUG] Intent: {intent}, Order ID: {order_id}")

    # Guardrail: block malicious
    if intent == "malicious":
        return "For your security and ours, I’m not able to help with that request."

    # Greetings
    if intent == "greeting":
        return "Hi there! 👋 How can I help you with your FoodHub order today?"

    # Fetch order status / cancel order needs an ID
    if intent in ("fetch_order_status", "cancel_order") and not order_id:
        return (
            "I can help with that. Please share your order ID "
            "(for example: O12488) so I can check the details."
        )

    # If we have an order ID, query DB
    if intent in ("fetch_order_status", "cancel_order") and order_id:
        order = fetch_order_by_id(order_id)
        if not order:
            return (
                f"I couldn’t find any order with ID {order_id}. "
                "Please double-check the ID and try again."
            )

        status_msg = build_status_message(order)

        if intent == "fetch_order_status":
            return f"For order {order_id}: {status_msg}"

        # cancel_order
        cancel_msg = can_order_be_cancelled(order)
        return f"For order {order_id}: {cancel_msg}"

    # Complaint or general_help → pure LLM
    return handle_general_or_complaint(message)


# =====================================================
# 6. GRADIO UI (ChatInterface handles history format)
# =====================================================

with gr.Blocks() as demo:
    gr.Markdown("## 🍕 FoodHub Order Support Chatbot")
    gr.Markdown(
        "Ask about your order status, cancellations, late delivery, or issues with your food.<br>"
        "For order-specific queries, please mention your <b>order ID</b> "
        "(for example: <code>O12488</code>)."
    )

    gr.ChatInterface(
        fn=foodhub_chat_agent,
        title="FoodHub Order Assistant",
        description="Chat with FoodHub support about your orders.",
        textbox=gr.Textbox(
            label="Type your message",
            placeholder="e.g., Where is order O12488?",
            lines=1,
        ),
        # ChatInterface includes a built-in Clear button
    )


if __name__ == "__main__":
    demo.launch()
