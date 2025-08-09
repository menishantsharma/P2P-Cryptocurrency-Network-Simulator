from id_manager import IDManager
from typing import Any

class Transaction:
    """
    Represents a transaction in the blockchain network.
    """
    def __init__(self, sender: Any, receiver: Any, amount: int):
        self.id = IDManager.get_txn_id()
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
    
    def __str__(self):
        return f"TxnID:{self.id}: {self.sender.id} pays {self.receiver.id} {self.amount} coins"
    
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Transaction) and self.id == other.id