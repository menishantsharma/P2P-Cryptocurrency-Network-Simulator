import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Any, Set, Dict, Optional


class Node:
    """
    Represents a peer in the cryptocurrency network.
    Each node:
      - Has a unique id and labels: slow/fast and low/high CPU.
      - Maintains a transaction pool and a tree of blocks (with arrival times).
      - Forwards transactions (avoiding loops) and performs block validation.
    """

    def __init__(self, id: int, is_slow: bool, is_low_cpu: bool, hashing_power: float):
        self.id: int = id
        self.peers: Set["Node"] = set()
        self.txn_pool: Set = set()
        self.blockchain: Dict[int, Any] = None
        self.tail: Any = None
        self.orphaned_blocks: Set = set()
        self.is_slow: bool = is_slow
        self.is_low_cpu: bool = is_low_cpu
        self.hashing_power: float = hashing_power
        self.block_pool: Set = set()
        self.block_tree: Dict[int, tuple] = None

    def get_balance(self, node: "Node", last_block: Optional[Any] = None) -> int:
        """
        Calculate the balance of a node in the blockchain network
        """
        balance: int = 0
        cur_block = last_block if last_block else self.tail

        while cur_block:
            for txn in cur_block.txns:
                if txn.sender == node:
                    balance -= txn.amount
                if txn.receiver == node:
                    balance += txn.amount
            
            if cur_block.miner == node:
                balance += 50

            cur_block = cur_block.parent

        return balance
    
    def get_unused_txns(self) -> Set:
        """
        Returns transactions in the pool that are not already in the blockchain
        """
        used_txns = set()
        cur_block = self.tail

        while cur_block:
            used_txns.update(cur_block.txns)
            cur_block = cur_block.parent
        
        return self.txn_pool - used_txns
    
    def validate_block(self, block: Any) -> bool:
        """
        Validate the block by ensuring that no sender's total spending exceeds their balance
        """

        total_txn_amount = defaultdict(int)

        for txn in block.txns:
            total_txn_amount[txn.sender.id] += txn.amount

        for sender_id, total_amount in total_txn_amount.items():
            sender = next(txn.sender for txn in block.txns if txn.sender.id == sender_id)
            sender_balance = self.get_balance(sender, block.parent)

            if sender_balance < total_amount:
                # print(f"Invalid block: {sender.id} has insufficient balance as balance is {sender_balance} and total amount is {total_amount}")
                return False
            
        return True
    
    def get_block_length(self, block: Any) -> int:
        """
        Returns the number of blocks from the given block back to genesis
        """

        length: int = 0
        cur_block = block

        while cur_block:
            length += 1
            cur_block = cur_block.parent
        
        return length
    
    def draw_blockchain(self):
        """
        Draw the blockchain network as a directed graph
        """
        graph = nx.DiGraph()

        for block in self.blockchain.values():
            graph.add_node(block.id)
            if block.parent is not None:
                graph.add_edge(block.id, block.parent.id)

        try:
            pos = nx.nx_agraph.graphviz_layout(graph, prog="dot", args="-Grankdir=BT")  # Top-to-bottom layout
        except:
            print("Please install graphviz to view the blockchain tree more clearly.")
            pos = nx.spring_layout(graph)  # Fallback if Graphviz is unavailable

        plt.figure(figsize=(8, 10))
        nx.draw(
            graph,
            pos,
            with_labels=False,
            labels=nx.get_node_attributes(graph, 'label'),
            node_size=10,
            node_color='black',
            font_size=10,
            font_weight='bold',
            edge_color='red',
            arrows=True,
            arrowstyle='-|>',
            arrowsize=10,
        )
        
        plt.title("Blockchain Tree (Top-to-Bottom Layout)")
        plt.axis("off")
        plt.show()

    def calc_link_latency(self, peer: "Node", message_size: float, rho) -> float:
        """
        Calculate latency between self and peer given a message_size (in kbytes) using an exponential distribution and fixed parameters.
        """

        # Message size in kbytes
        cij = 5 if self.is_slow or peer.is_slow else 100  
        # Propagation delay in seconds
        dij: float = np.random.exponential(96/(cij * 1024))
        latency: float = dij + rho[self.id][peer.id] + message_size / (cij * 128)
        return latency
    
    def get_mined_blocks(self, node) -> int:
        """
        Get the number of blocks mined by the node
        """
        count: int = 0
        cur_block = self.tail
        while cur_block:
            if cur_block.miner == node:
                count += 1
            cur_block = cur_block.parent
        return count
    
    def print_blockchain(self):
        """
        Print the blockchain of the node
        """
        cur_block = self.tail
        while cur_block:
            print(f"Block Id: {cur_block.id} | Miner: {cur_block.miner.id if cur_block.miner else None} | Parent: {cur_block.parent.id if cur_block.parent else None}\nTxns: {[(txn.sender.id, txn.receiver.id, txn.amount) for txn in cur_block.txns]}")
            cur_block = cur_block.parent

    def record_block(self, block: Any, arrival_time: float):
        """
        Record the arrival of a block in the block tree
        """
        parent = block.parent if block.parent else None
        self.block_tree[block.id] = (parent, arrival_time)

    def export_blocktree(self):
        """
        Export the block tree to a file
        """
        filename = f"node{self.id}_blocktree.txt"

        with open(filename, "w") as f:
            f.write("Node,Parent,ArrivalTime\n")
            for block_id, (parent, arrival_time) in self.block_tree.items():
                parent_id = parent.id if parent else None
                f.write(f"{block_id},{parent_id},{arrival_time}\n")
    
    def get_branches(self):
        """
        Get the branches of the blockchain
        """
        longest_chain = set()
        cur_block = self.tail

        while cur_block:
            longest_chain.add(cur_block.id)
            cur_block = cur_block.parent

        # Find all leaf nodes in the blockchain

        graph = nx.DiGraph()
        for block in self.blockchain.values():
            graph.add_node(block.id)
            if block.parent is not None:
                graph.add_edge(block.parent.id, block.id)
            
        leaf_block_ids = [x for x in graph.nodes() if graph.out_degree(x)==0 and graph.in_degree(x)==1]
        leaf_blocks = [self.blockchain[block_id] for block_id in leaf_block_ids]
        leaf_blocks.remove(self.tail)

        branches = []
        while leaf_blocks:
            branch = set()
            leaf_block = leaf_blocks.pop()
            cur_block = leaf_block

            while cur_block:
                if cur_block.id in longest_chain:
                    break
                branch.add(cur_block.id)
                cur_block = cur_block.parent
            
            branches.append(branch)

        return branches

    
    def __str__(self):
        return f"Node Id: {self.id} | Peers: {[peer.id for peer in self.peers]} | Tail: {self.tail.id} | Balance: {self.get_balance(self)} | Slow: {self.is_slow} | Low CPU: {self.is_low_cpu} | Hashing Power: {self.hashing_power}"
