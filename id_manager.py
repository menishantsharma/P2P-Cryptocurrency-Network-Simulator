class IDManager:
    block_counter = 0
    txn_counter = 0
    event_counter = 0

    @classmethod
    def get_block_id(cls):
        cls.block_counter += 1
        return cls.block_counter
    
    @classmethod
    def get_txn_id(cls):
        cls.txn_counter += 1
        return cls.txn_counter
    
    @classmethod
    def get_event_id(cls):
        cls.event_counter += 1
        return cls.event_counter