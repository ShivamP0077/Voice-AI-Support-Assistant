import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_orders():
    """Load orders from data/orders.json."""
    path = os.path.join(DATA_DIR, "orders.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_policies():
    """Load policies from data/policies.json."""
    path = os.path.join(DATA_DIR, "policies.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_orders(orders):
    """Convert each order into a dictionary containing text and metadata."""
    chunks = []
    for order in orders:
        delivery_info = ""
        if order.get("delivery_date"):
            delivery_info = f"delivered on {order['delivery_date']}."
        elif order.get("expected_delivery"):
            delivery_info = f"expected delivery on {order['expected_delivery']}."
            
        return_info = f" Return window: {order['return_window_days']} days." if order.get("return_window_days") else ""
            
        text = (
            f"Order {order['order_id']} for user {order['user_id']}: "
            f"{order['item']}, status: {order['status']}, "
            f"{delivery_info}{return_info}"
        ).strip()
        
        chunks.append({
            "text": text,
            "type": "order",
            "user_id": order["user_id"]
        })
    return chunks


def chunk_policies(policies):
    """Convert each policy section into a dict containing text and metadata."""
    chunks = []
    
    if "returns" in policies:
        ret = policies["returns"]
        conds = ", ".join(ret.get("conditions", []))
        text = f"Returns Policy: are returns allowed? {ret.get('allowed')}. Window: {ret.get('window_days')} days. Conditions: {conds}."
        chunks.append({"text": text, "type": "policy", "user_id": None})
        
    if "refunds" in policies:
        ref = policies["refunds"]
        text = f"Refunds Policy: method is {ref.get('method')}. Processing time takes {ref.get('processing_time_days')} days."
        chunks.append({"text": text, "type": "policy", "user_id": None})
        
    if "support_hours" in policies:
        text = f"Support Hours Policy: Support is available during {policies['support_hours']}."
        chunks.append({"text": text, "type": "policy", "user_id": None})

    return chunks


def load_all_chunks():
    """Load orders and policies, return all text chunks ready for embedding."""
    orders = load_orders()
    policies = load_policies()

    order_chunks = chunk_orders(orders)
    policy_chunks = chunk_policies(policies)

    all_chunks = order_chunks + policy_chunks
    return all_chunks
