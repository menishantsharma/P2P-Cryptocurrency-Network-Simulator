from id_manager import IDManager

class Block:
    """
    Represents a block in the blockchain
    """
    # txns - set of transactions in the block
    def __init__(self, miner, parent, txns):
        self.id = IDManager.get_block_id()
        self.miner = miner
        self.parent = parent
        self.txns = txns

    def get_block_size(self) -> float:
        """
        Return the block size (in kbytes). For simulation: coinbase is 1 kB; each txn is 1 kB.
        """
        if self.miner:
            return 2 + len(self.txns)
        
        return 1 + len(self.txns)

    def __str__(self):
        parent_id = self.parent.id if self.parent else None
        miner_id = self.miner.id if self.miner else None
        return (f"Block Id: {self.id} | Miner Id: {miner_id} | Parent Id: {parent_id} "
                f"Txns: {[str(txn) for txn in self.txns]}")