from block import Block
import numpy as np
from node import Node
import networkx as nx
from transaction import Transaction
from event import Event
import heapq
import argparse           

# -------------------------------
# Get Inputs from User
# -------------------------------

parser = argparse.ArgumentParser(description='Simulate a blockchain network')
parser.add_argument('--z0', type=int, help='Percentage of slow nodes', required=True)
parser.add_argument('--z1', type=int, help='Percentage of low cpu nodes', required=True)
parser.add_argument('--n', type=int, help='Number of nodes', required=True)
parser.add_argument('--T', type=float, help='Maximum simulation time in seconds', required=True)
parser.add_argument('--ttx', type=float, help='Mean transaction interval in seconds', required=True)
parser.add_argument('--tblk', type=float, help='Mean block interval in seconds', required=True)

args = parser.parse_args()

num_nodes: int = args.n
z0: int = args.z0                        # percentage of slow nodes
z1: int = args.z1                        # percentage of low CPU nodes
max_time: int = args.T                   # mean transaction interval in seconds
mean_txn_interval = args.ttx        # mean block interval in seconds
mean_block_interval = args.tblk     # maximum simulation time in seconds
events = []                         # priority queue for events

# ρij: propagation delay matrix in seconds (uniform between 0.01 and 0.5)
rho = np.random.uniform(0.1, 5, size=(num_nodes, num_nodes)) 

# -------------------------------
# Create Nodes
# -------------------------------

def get_shuffled_boolean_list(size, percentage):
    """
    Generate a shuffled list of booleans with a given percentage of True values.
    """
    num_true = size * percentage // 100
    bool_list = [True] * num_true + [False] * (size - num_true)
    np.random.shuffle(bool_list)
    return bool_list

slow_nodes = get_shuffled_boolean_list(num_nodes, z0)
low_cpu_nodes = get_shuffled_boolean_list(num_nodes, z1)

# Calculate hashing power: high-CPU nodes get 10; low-CPU nodes get 1
hashing_power = np.where(low_cpu_nodes, 1, 10)
hashing_power_fractions = hashing_power / np.sum(hashing_power)
hashing_power_fractions = np.round(hashing_power_fractions, 2)
nodes = [Node(i, slow_nodes[i], low_cpu_nodes[i], hashing_power_fractions[i]) for i in range(num_nodes)]

# -------------------------------
# Connect Nodes
# -------------------------------

min_degree = min(3, num_nodes - 1)
max_degree = min(num_nodes - 1, 6)

# Generate a random graph with the given degree sequence and ensure it is connected
while True:
    degree_sequence = np.random.randint(min_degree, max_degree + 1, num_nodes)
    
    if nx.is_graphical(degree_sequence):
        try:
            graph = nx.random_degree_sequence_graph(degree_sequence)
            if nx.is_connected(graph):
                break
        except Exception:
            continue
    
for node in nodes:
    peers = set(graph.neighbors(node.id))

    for peer_id in peers:
        node.peers.add(nodes[peer_id])

# -------------------------------
# Create funding transactions
# -------------------------------

system_node = Node(-1, False, False, 0)
funding_txns = [Transaction(system_node, node, 10000) for node in nodes]
genesis = Block(system_node, None, set(funding_txns))

for node in nodes:
    node.tail = genesis
    node.blockchain = {genesis.id: genesis}
    node.block_tree = {genesis.id: (None, 0)}

# ------------------------------------------
# Helper Function to Schedule Events
# ------------------------------------------

def schedule_txn(sender, current_time):
    """
    Schedule the next transaction for a node.
    """
    next_time = current_time + np.random.exponential(mean_txn_interval)
    receiver = np.random.choice(nodes)
    coins = np.random.randint(1, 10)
    new_txn = Transaction(sender, receiver, coins)
    new_event = Event(sender, None, "TXN_SEND", new_txn, None)
    heapq.heappush(events, (next_time, new_event))

def schedule_block(sender, current_time):
    """
    Schedule the next block for a node.
    """
    next_time = current_time + np.random.exponential(mean_block_interval / sender.hashing_power)
    # print(sender.is_low_cpu, current_time, next_time)
    txnToAdd = sender.get_unused_txns()

    # Validate transactions (remove those where sender's balance is insufficient).
    for txn in list(txnToAdd):
        if sender.get_balance(txn.sender, sender.tail) < txn.amount:
            txnToAdd.remove(txn)
    
    # Limit the number of transactions to 1022
    txnToAdd = set(list(txnToAdd)[:1022])
    new_block = Block(sender, sender.tail, txnToAdd)
    new_event = Event(sender, None, "BLOCK_SEND", None, new_block)
    heapq.heappush(events, (next_time, new_event))

# Schedule initial events for each node
for node in nodes:
    schedule_txn(node, 0)
    schedule_block(node, 0)

# ------------------------------------------
# Main Simulation Loop
# ------------------------------------------

print("Wait a moment it will take time bcoz it is python :)...")
while events:
    current_time, event = heapq.heappop(events)

    if current_time > max_time:
        break
    
    if event.type == "TXN_SEND":
        # Send transaction to all peers
        for peer in event.sender.peers:
            latency =  event.sender.calc_link_latency(peer, 1, rho)
            new_event = Event(event.sender, peer, "TXN_RECV", event.txn, None)
            heapq.heappush(events, (current_time + latency, new_event))

        # Next transaction
        schedule_txn(event.sender, current_time)

    elif event.type == "TXN_RECV":
        if event.txn not in event.receiver.txn_pool:
            event.receiver.txn_pool.add(event.txn)

            # Send transaction to all peers
            for peer in event.receiver.peers:
                latency = event.receiver.calc_link_latency(peer, 1, rho)
                new_event = Event(event.receiver, peer, "TXN_RECV", event.txn, None)
                heapq.heappush(events, (current_time + latency, new_event))

    elif event.type == "BLOCK_SEND":
        # Only add block if its parent matches the current tail and it validates

        if event.block.parent == event.sender.tail and event.sender.validate_block(event.block):
            event.sender.blockchain[event.block.id] = event.block
            event.sender.tail = event.block

            event.sender.record_block(event.block, current_time)

            # Send block to all peers
            for peer in event.sender.peers:
                latency = event.sender.calc_link_latency(peer, event.block.get_block_size(), rho)
                new_event = Event(event.sender, peer, "BLOCK_RECV", None, event.block)
                heapq.heappush(events, (current_time + latency, new_event))

        # Mine next block
        schedule_block(event.sender, current_time)

    elif event.type == "BLOCK_RECV":
        if event.block not in event.receiver.block_pool:
            event.receiver.block_pool.add(event.block)

            if event.block.parent not in event.receiver.blockchain.values():
                event.receiver.orphaned_blocks.add(event.block)
            else:
                # Process orphaned blocks
                queue = [event.block]

                last_block = event.block

                while queue:
                    cur_block = queue.pop(0)                        

                    if not event.receiver.validate_block(cur_block):
                        continue

                    event.receiver.blockchain[cur_block.id] = cur_block
                    event.receiver.record_block(cur_block, current_time)

                    if event.receiver.get_block_length(cur_block) > event.receiver.get_block_length(last_block):
                        last_block = cur_block

                    for peer in event.receiver.peers:
                        latency = event.receiver.calc_link_latency(peer, cur_block.get_block_size(), rho)
                        new_event = Event(event.receiver, peer, "BLOCK_RECV", None, cur_block)
                        heapq.heappush(events, (current_time + latency, new_event))
                    
                    orphans_to_remove = []
                    for orphan in list(event.receiver.orphaned_blocks):
                        if orphan.parent == cur_block:
                            queue.append(orphan)
                            orphans_to_remove.append(orphan)
                    
                    for orphan in orphans_to_remove:
                        event.receiver.orphaned_blocks.remove(orphan)

                if(event.receiver.get_block_length(last_block) > event.receiver.get_block_length(event.receiver.tail)):
                    event.receiver.tail = last_block
                    schedule_block(event.receiver, current_time)



# ------------------------------------------
# End of Simulation
# ------------------------------------------
for node in nodes:
    node.export_blocktree()

# -------------------------------
# Analysis
# -------------------------------

# Please install graphviz before using this function
nodes[0].draw_blockchain()

length_of_longest_chain = nodes[0].get_block_length(nodes[0].tail) - 1
print()
print("Length of the longest chain in blocktree (excluding genesis): ",length_of_longest_chain)

total_blocks_mined = len(nodes[0].blockchain) - 1

print("Number of mined blocks which are added in blocktree (exluding genesis and rejected blocks): ", total_blocks_mined)
print("Fraction of mined blocks present in the longest chain to the total number of blocks in blocktree): ", round(length_of_longest_chain/total_blocks_mined, 2))

print()

low_cpu_slow_nodes_block_count = 0
low_cpu_fast_nodes_block_count = 0
high_cpu_slow_nodes_block_count = 0
high_cpu_fast_nodes_block_count = 0

for node in nodes:
    if node.is_low_cpu and node.is_slow:
        low_cpu_slow_nodes_block_count += nodes[0].get_mined_blocks(node)
    
    if node.is_low_cpu and not node.is_slow:
        low_cpu_fast_nodes_block_count += nodes[0].get_mined_blocks(node)

    if not node.is_low_cpu and node.is_slow:
        high_cpu_slow_nodes_block_count += nodes[0].get_mined_blocks(node)

    if not node.is_low_cpu and not node.is_slow:
        high_cpu_fast_nodes_block_count += nodes[0].get_mined_blocks(node) 

print()

print("Fraction of blocks mined by low CPU and slow nodes to the total number of blocks present in the longest chain: ", round(low_cpu_slow_nodes_block_count/length_of_longest_chain, 2))
print("Fraction of blocks mined by low CPU and fast nodes to the total number of blocks present in the longest chain: ", round(low_cpu_fast_nodes_block_count/length_of_longest_chain, 2))
print("Fraction of blocks mined by high CPU and slow nodes to the total number of blocks present in the longest chain: ", round(high_cpu_slow_nodes_block_count/length_of_longest_chain, 2))
print("Fraction of blocks mined by high CPU and fast nodes to the total number of blocks present in the longest chain: ", round(high_cpu_fast_nodes_block_count/length_of_longest_chain, 2))

print()
for node in nodes:
    temp = nodes[0].get_mined_blocks(node)
    print(f"Fraction of blocks mined by node {node.id} which is {'slow' if node.is_slow else 'fast'} and {'low cpu' if node.is_low_cpu else 'high cpu'} to the total number of blocks present in the longest chain: {round(temp/length_of_longest_chain, 2)}")

print()

branches = nodes[0].get_branches()
branch_lengths = [len(branch) for branch in branches]
print("Length of each branch (branches are tree of blocks which is not part of the longest chain): ", branch_lengths)
if(len(branch_lengths) > 0):
    print("Average length of branches: ", round(np.mean(branch_lengths), 2))

print()