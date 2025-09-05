# ops.py
class VirtualOp:
    """Wrap an existing atomic op so it has its own identity/name."""
    def __init__(self, base_op, name: str):
        self.base_op = base_op
        self.name = name
    def __call__(self, X):
        return self.base_op(X)
