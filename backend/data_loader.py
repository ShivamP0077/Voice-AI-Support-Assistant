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
    """Convert each order into a flat text chunk for embedding.

    Example output:
      "Order ORD1001 for user U1: Wireless Headphones, status delivered
       on 2026-03-20. Return window: 7 days."
    """
    chunks = []
    for order in orders:
        delivery_info = (
            f"on {order['delivery_date']}"
            if order["delivery_date"]
            else "not yet delivered"
        )
        chunk = (
            f"Order {order['order_id']} for user {order['user_id']}: "
            f"{order['product']} (qty: {order['quantity']}, ${order['price']}), "
            f"status {order['status']} {delivery_info}. "
            f"Return window: {order['return_window_days']} days."
        )
        chunks.append(chunk)
    return chunks


def chunk_policies(policies):
    """Convert each policy section into a flat text chunk for embedding.

    Example output:
      "Return Policy: Items can be returned within 7 days of delivery..."
    """
    chunks = []
    for key, policy in policies.items():
        chunk = f"{policy['title']}: {policy['content']}"
        chunks.append(chunk)
    return chunks


def load_all_chunks():
    """Load orders and policies, return all text chunks ready for embedding."""
    orders = load_orders()
    policies = load_policies()

    order_chunks = chunk_orders(orders)
    policy_chunks = chunk_policies(policies)

    all_chunks = order_chunks + policy_chunks
    return all_chunks
