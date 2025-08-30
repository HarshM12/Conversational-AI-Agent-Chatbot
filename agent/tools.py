from langchain_core.tools import tool
import json
import os
import json
from difflib import get_close_matches
import difflib
from pathlib import Path
import json

faq_path = os.path.join(os.path.dirname(__file__), "../data/faqs.json")
with open(faq_path, "r", encoding="utf-8") as f:
    FAQ_DATA = json.load(f)


@tool
def lookup_faq(query: str) -> dict:
    """Lookup a frequently asked question by keyword or text and return the best matching answer."""
    questions = list(FAQ_DATA.keys())
    match = difflib.get_close_matches(query, questions, n=1, cutoff=0.4)
    if match:
        return {"answer": FAQ_DATA[match[0]], "is_final": True}
    return {"answer": "No matching FAQ found. Please provide more details.", "is_final": True}


@tool
def collect_feedback(feedback: str) -> str:
    """Collect feedback from the user and store it in a log file. Returns a confirmation message."""
    import os
    import datetime

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    log_path = os.path.join(data_dir, "feedback.log")

    try:
        os.makedirs(data_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] Feedback: {feedback}\n")
        
        print(f">>> Feedback written to: {log_path}")
        return "Feedback collected successfully. Thank you!"
    except Exception as e:
        print(f">>> Error writing feedback to {log_path}: {str(e)}")
        return f"Error collecting feedback: {str(e)}"


@tool
def track_order(order_id: str | dict) -> dict:
    """Return the status of an order by its ID."""
    import json
    from pathlib import Path

    if isinstance(order_id, str):
        try:
            data = json.loads(order_id)
            order_id = data.get("order_id", None)
        except json.JSONDecodeError:
            pass

    if not order_id:
        raise ValueError("Order ID must be provided")

    orders_path = Path.cwd() / "data" / "orders.json"
    print(f">>> Loading orders from: {orders_path}")
    with open(orders_path, "r", encoding="utf-8") as f:
        orders = json.load(f)

    order_id_str = str(order_id).strip()
    status = orders.get(order_id_str, "Order ID not found.")
    return {"answer": status, "is_final": True}
