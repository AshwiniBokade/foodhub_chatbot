# foodhub_chatbot
FoodHub – AI-Powered Order Support Chatbot (LLM + SQL Agent)

FoodHub is an online food delivery platform where customers frequently ask about order status, delays, cancellations, and delivery issues.
This project builds an AI-powered customer support chatbot that:

✅ Uses Groq LLM (llama-3.3-70b-versatile)
✅ Connects to a SQLite database via a SQL Agent
✅ Retrieves real order details safely
✅ Applies guardrails for malicious users
✅ Generates polite, short, customer-friendly responses
✅ Can be deployed on Hugging Face Spaces

Objective

To automate customer support queries using an LLM + SQL Agent pipeline that can:
Fetch order status from the database
Handle order cancellation requests
Respond politely to complaints
Reject unauthorized/malicious requests
Operate like a real food delivery chatbot
