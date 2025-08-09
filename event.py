from id_manager import IDManager

class Event:
    """
    Represents an event in the simulation.
    Attributes:
        - sender: The sender of the event
        - receiver: The receiver node of the event or None
        - type: One of "TXN_SEND", "TXN_RECV", "BLOCK_SEND", "BLOCK_RECV"
        - txn: The transaction object if the event is a transaction event
        - block: The block object if the event is a block
        - scheduled_time: The time at which the event is scheduled

    """
    def __init__(self, sender, receiver, type, txn, block):
        self.id = IDManager.get_event_id()
        self.sender = sender
        self.receiver = receiver
        self.type = type
        self.txn = txn
        self.block = block
    
    def __lt__(self, other):
        return id(self) < id(other)