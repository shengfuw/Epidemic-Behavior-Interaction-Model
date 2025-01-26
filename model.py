import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

class Node: 

    def __init__(self, node_id, a, b, gamma, eta) -> None:
        """
        Initialize a Node with an ID and initial state.
        
        Args:
        - node_id (int): Unique identifier for the node.
        - state_awareness (int): 0 for unaware (U), 1 for aware (A).
        - state_disease (int): 0 for susceptible (S), 1 for infected (I).
        """

        self.id = node_id
        self.state_disease = 0
        self.state_awareness = 0

        self.new_state_disease = 0
        self.new_state_awareness = 0

        self.physical_neighbors = []
        self.virtual_neighbors = []

        # self.aware_therold = np.random.uniform(0.05, 0.95)
        # self.aware_therold = np.random.normal(0.5, 0.25)
        # self.aware_therold = 0.4
        self.aware_therold = np.random.beta(a, b)

        self.infected_time = -1

        self.gamma = gamma
        self.eta = eta

    def add_physical_neighbor(self, neighbor_node):
        self.physical_neighbors.append(neighbor_node)
    
    def remove_physical_neighbor(self, neighbor_node):
        self.physical_neighbors.remove(neighbor_node)

    def add_virtual_neighbor(self, neighbor_node):
        self.virtual_neighbors.append(neighbor_node)

    def remove_virtual_neighbor(self, neighbor_node):
        self.virtual_neighbors.remove(neighbor_node)

    def get_physical_neighbors_id(self):
        return [neighbors.id for neighbors in self.physical_neighbors]

    def get_virtual_neighbors_id(self):
        return [neighbors.id for neighbors in self.virtual_neighbors]

    def turn_into_aware(self, time: int):
        self.new_state_awareness = 1
        self.forget_time = time + np.random.exponential(60) # rate of forgetting

    def turn_into_unaware(self):
        self.new_state_awareness = 0
        self.forget_time = -1
        
    def turn_into_infectious(self, time: int):
        self.new_state_disease = 1
        self.infected_time = time

    def turn_into_susceptible(self): # recover
        self.new_state_disease = 0
        self.infected_time = -1

    def update_to_new_state(self):
        self.state_disease = self.new_state_disease
        self.state_awareness = self.new_state_awareness

    def rewire(self, all_susceptible_nodes):
        for neighbor in random.sample(self.physical_neighbors, len(self.physical_neighbors)):
            # break the link with the infected neighbor at probability gamma
            if neighbor.state_disease == 1 and np.random.rand() < self.gamma:
                
                # rewire to a neighbor of a neighbor (nodes at distance 2) at probability eta
                if np.random.rand() < self.eta:
                    new_links_pool = set()
                    for n in self.physical_neighbors:
                        if n.id != neighbor.id:
                            new_links_pool.update(n.physical_neighbors) 
                    [new_links_pool.discard(n) for n in list(new_links_pool) if n.state_disease == 1] # remove the infected modes
                # rewire to random susceptible node at probability 1 - eta
                else:
                    new_links_pool = set(all_susceptible_nodes)
                
                new_links_pool = new_links_pool - set(self.physical_neighbors)
                new_links_pool.discard(self)
                # print(self.id, neighbor.id, [n.id for n in list(new_links_pool)], [n.id for n in all_susceptible_nodes])

                if len(new_links_pool) > 0:
                    new_neighbor = random.choice(list(new_links_pool))
                    self.add_physical_neighbor(new_neighbor)
                    new_neighbor.add_physical_neighbor(self)

                    self.remove_physical_neighbor(neighbor)
                    neighbor.remove_physical_neighbor(self)

                    return 1
        return 0

class TwoLayerNetwork:
    def __init__(self, N, k, a, b, beta_a, beta_u, alpha, infection_period, gamma, eta, threshold_formula, psi="", lambda_="", rho="", network_type="ER", initial_infected_ratio=0.01, intial_aware_ratio=0.1):
        """
        Initialize a two-layer network with N nodes in each layer.

        Args:
        - N (int): Number of nodes in each layer.
        - rewiring_percentage (float): Percentage of edges to rewire in the virtual network
        - beta_a (float): Transmission probability for aware nodes.
        - beta_u (float): Transmission probability for unaware nodes.
        - alpha (float): Probability of recovery.
        - infection_period (int): Duration of the infection period.
        - gamma (float): Probability of rewiring the physical network.
        - eta (float): Probability of rewiring to a neighbor of a neighbor.
        """
        # Create N nodes 
        self.nodes = [Node(node_id=i, a=a, b=b, gamma=gamma, eta=eta) for i in range(N)]

        # Infect a fraction of the nodes
        num_infected = int(initial_infected_ratio * N)
        infected_nodes = np.random.choice(self.nodes, num_infected, replace=False)
        for node in infected_nodes:
            node.turn_into_infectious(time=0)
        
        # Make a fraction of the nodes aware
        num_aware = int(intial_aware_ratio * N)
        aware_nodes = np.random.choice(self.nodes, num_aware, replace=False)
        for node in aware_nodes:
            node.turn_into_aware(0)

        for node in self.nodes:
            node.update_to_new_state()

        # Create the physical network
        p = k/N
        if network_type == "ER":
            G = nx.erdos_renyi_graph(n=N, p=p)
        elif network_type == "BA":
            G = nx.barabasi_albert_graph(n=N, m=k)
        elif network_type == "WS":
            G = nx.connected_watts_strogatz_graph(n=N, k=k, p=0.5)
        else:
            raise ValueError("Invalid network type. Choose 'ER', 'BA', or 'WS'.")
        
        self.create_physical_network(G)

        # Rewire the network by performing double edge swaps
        # num_swaps = int((1-overlap_percentage) * G.number_of_edges())
        # nx.double_edge_swap(G, nswap=num_swaps)
        self.create_virtual_network(G)

        # Set the parameters of the model
        self.N = N
        self.k = k
        self.p = p
        self.a = a
        self.b = b
        
        self.beta_a = beta_a
        self.beta_u = beta_u
        self.alpha = alpha
        self.infection_period = infection_period  
        self.gamma = gamma
        self.eta = eta

        self.threshold_formula = threshold_formula
        self.psi = psi
        self.lambda_ = lambda_
        self.rho = rho
        if self.rho == 0:
            self.rho = 0.01
        
        self.round = 0
        self.rewiring_count = 0

        self.infection_counts = [self.count_infected()]
        self.awreness_counts = [self.count_aware()]
        
        self.adjust_awareness_thresholds = [] # for recording the adjusted awareness thresholds

    def create_physical_network(self, G):
        """
        Create the physical network by adding physical neighbors to each node.

        Args:
        - G (nx.Graph): NetworkX graph representing the physical network.
        """
        for node in self.nodes:
            for neighbor in G[node.id]:
                node.add_physical_neighbor(self.nodes[neighbor])
    
    def create_virtual_network(self, G):
        """
        Create the virtual network by adding virtual neighbors to each node.

        Args:
        - G (nx.Graph): NetworkX graph representing the virtual network.
        """
        for node in self.nodes:
            for neighbor in G[node.id]:
                node.add_virtual_neighbor(self.nodes[neighbor])
    
    def update_health_state(self):
        """
        Update the health state of each node in the network.
        """
        for node in random.sample(self.nodes, len(self.nodes)):
            # If the node is susceptible: S -> I
            if node.state_disease == 0: 
                # 1. count the number of infected neighbors
                num_infected_neighbors = sum([neighbor.state_disease == 1 for neighbor in node.physical_neighbors])

                # 2. calculate the probability of transmission
                if node.state_awareness == 1:
                    transmission_prob = 1 - (1 - self.beta_a) ** num_infected_neighbors
                else:
                    transmission_prob = 1 - (1 - self.beta_u) ** num_infected_neighbors

                # 3. set the state of the node to infected with probability 'transmission_prob'
                if np.random.rand() < transmission_prob:
                    node.turn_into_infectious(time=self.round)

            # If the node is infected: I -> S
            elif node.state_disease == 1: 
                if self.round - node.infected_time >= self.infection_period: 
                    if np.random.rand() < self.alpha:
                        node.turn_into_susceptible()
    
    def update_awareness_state(self):
        """
        Update the awareness state of each node in the network.
        """
        if self.threshold_formula == "simple": # simple contagion
            for node in random.sample(self.nodes, len(self.nodes)):
            # If the node is unaware: U -> A
                if node.state_awareness == 0: 
                    # 1. count the number of awared neighbors
                    num_awared_neighbors = sum([neighbor.state_awareness == 1 for neighbor in node.virtual_neighbors])

                    # 2. calculate the probability of transmission
                    transmission_prob = 1 - (1 - self.psi) ** num_awared_neighbors

                    # 3. set the state of the node to awared with probability 'transmission_prob'
                    if np.random.rand() < transmission_prob:
                        node.turn_into_aware(time=self.round)

                # If the node is awared: I -> S
                elif node.state_awareness == 1: 
                    if self.round >= node.forget_time: 
                        node.turn_into_unaware()
        
        elif self.threshold_formula == "complex": # complex contagion
            global_infection_rate = self.count_infected() / len(self.nodes)
            
            for node in random.sample(self.nodes, len(self.nodes)):
            
                if len(node.virtual_neighbors) == 0:
                    continue

                # 1. count the proportion of aware neighbors
                proportion_aware_neighbors = sum([neighbor.state_awareness == 1 for neighbor in node.virtual_neighbors]) / len(node.virtual_neighbors) 

                # 2. decide the threshold for turning into aware
                adjusted_threshold = node.aware_therold * (1 - self.lambda_*global_infection_rate) * (1/self.rho)
                adjusted_threshold = np.clip(adjusted_threshold, 0, 1)
                self.adjust_awareness_thresholds.append(adjusted_threshold)

                # 3. set the state of the node to aware if the threshold is met
                if proportion_aware_neighbors > adjusted_threshold:
                    node.turn_into_aware(time=self.round)
                else:
                    node.turn_into_unaware()
        
    def update_physical_network(self):
        """
        Update the network structure at the physical layer.
        """
        # get all susceptible nodes
        all_susceptible_nodes = [node for node in self.nodes if node.state_disease == 0]

        for node in random.sample(self.nodes, len(self.nodes)):
            # If the node is susceptible
            if node.state_disease == 0:
                self.rewiring_count += node.rewire(all_susceptible_nodes)
    
    def update_virtual_network(self):
        """
        Update the network structure at the virtual layer.
        """
        for node in self.nodes:
            node.virtual_neighbors = node.physical_neighbors.copy()
    
    def count_infected(self):
        return sum(node.state_disease == 1 for node in self.nodes)
    
    def count_aware(self):
        return sum(node.state_awareness == 1 for node in self.nodes)
    
    def update_one_round(self):
        self.round += 1
    
        self.adjust_awareness_thresholds = [] # for recording the adjusted awareness thresholds

        self.update_health_state()
        self.update_awareness_state()
        for node in self.nodes:
            node.update_to_new_state()

        self.update_physical_network()
        self.update_virtual_network()

        # Record the number of infected and aware nodes
        self.infection_counts.append(self.count_infected())
        self.awreness_counts.append(self.count_aware())
            
    
    def run_simulation(self, num_rounds, print_info=True, plot_network=True):
        if print_info: self.print_info()
        for t in range(num_rounds):
            pi = self.count_infected() / len(self.nodes)

            self.update_one_round()
            
            if plot_network and (t % 300 == 0):
                print(f"Time step: {self.round}")
                print(f"Awareness count: {self.count_aware()}")
                plt.figure(figsize=(3, 2))
                plt.hist(self.adjust_awareness_thresholds, bins=50, edgecolor='black')
                plt.xlim(0, 1) 
                plt.title(f'Global Infection rate={pi:.2f}')
                plt.show()

                self.plot_network("physical")
        if print_info: self.print_info()


    def get_network(self, layer):
        """
        Get the network of the specified layer.

        Args:
        - layer (str): "physical" or "virtual"
        """
        G = nx.Graph()
        edge_list = []

        for node in self.nodes:
            if node.state_awareness == 1 and node.state_disease == 1:
                color = 'skyblue' # A and I
            elif node.state_awareness == 0 and node.state_disease == 1:
                color = 'darkblue' # U and I
            elif node.state_awareness == 1 and node.state_disease == 0:
                color = 'gray' # A and S
            else: 
                color = 'black' # U and S
            G.add_node(node.id, color=color)

            if layer == "physical":
                for neighbor in node.physical_neighbors:
                    edge_list.append((node.id, neighbor.id))
        
            elif layer == "virtual":
                for neighbor in node.virtual_neighbors:
                    edge_list.append((node.id, neighbor.id))

        G.add_edges_from(edge_list)
        return G
    

    def plot_network(self, layer="physical", component_num=2):
        G = self.get_network(layer)

        connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
        largest_components = connected_components[:component_num]
        subgraphs = [G.subgraph(component).copy() for component in largest_components]
        pos = nx.spring_layout(G, seed=42) 
        plt.figure(figsize=(8, 6))
        for subgraph in subgraphs:
            node_colors = nx.get_node_attributes(subgraph, 'color').values()
            nx.draw(
                subgraph,
                pos, 
                with_labels=False,
                node_color=node_colors,
                node_size=2.5,
                edge_color='lightgray',
                width=0.5
            )
        
        colors = ['skyblue', 'darkblue', 'gray', 'black']
        labels = ['I+A: Infected and Adopting', 'I+N: Infected and Not Adopting', 'S+A: Susceptible and Adopting', 'S+N: Susceptible and Not Adopting']
        for color, label in zip(colors, labels):
            plt.scatter([], [], color=color, label=label)

        plt.legend(title="Node Status", loc="best")
        plt.title(f"Largest {component_num} Components of the Graph")
        plt.show()
        

    def print_info(self):
        print(f"Parameters: N={self.N}, k={self.k}, a={self.a}, b={self.b}, beta_a={self.beta_a}, beta_u={self.beta_u}, alpha={self.alpha}, infection_period={self.infection_period}, gamma={self.gamma}, eta={self.eta}, threshold_formula={self.threshold_formula}, psi={self.psi}, lambda={self.lambda_}, rho={self.rho}, network_type={self.network_type}, initial_infected_ratio={self.initial_infected_ratio}, intial_aware_ratio={self.intial_aware_ratio}")
        print(f"Number of Infected, Aware: {self.count_infected()}, {self.count_aware()}")
        print(f"Physical, Virtual network: {self.get_network('physical').number_of_edges()}, {self.get_network('virtual').number_of_edges()} edges")
        print(f"The number of isolated nodes: {nx.number_of_isolates(self.get_network('physical'))}")
        print(f"Clustering coefficient, Transitivity: {nx.average_clustering(self.get_network('physical')):.6f}, {nx.transitivity(self.get_network('physical')):.6f}")
        print('-'*30)


    def plot_prevalence_trend(self, ax=None):
        time_steps = self.round
        infection_proportion = np.array(self.infection_counts) / self.N
        awareness_proportion = np.array(self.awreness_counts) / self.N
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))  # Create a new figure if no axes provided
        ax.plot(range(time_steps+1), infection_proportion, label='I: Infected population')
        ax.plot(range(time_steps+1), awareness_proportion, label='A: Adopting population', color='orange')

        # ax.title('EBIM: SIS + NAN Model')
        ax.set_xlabel('Time step (Iteration)')
        ax.set_ylabel('Percentage of agents')
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        # if ax.figure:
        #     plt.show()

    def get_statistics(self, type, window_size=500):
        if type == "infection":
            y = np.array(self.infection_counts) / self.N
        elif type == "awareness":
            y = np.array(self.awreness_counts) / self.N
        else:
            raise ValueError("Invalid type. Choose 'infection' or 'awareness'.")
        
        last_segment = y[-window_size:]

        # amplitu: Max-Min
        amplitude = last_segment.max() - last_segment.min()
        
        # standard deviation
        std = last_segment.std()

        # average the last 10 time steps
        average = y[-10:].mean()   

        return y[-1], average, amplitude, std
    
def plot_heatmap(data, title, fig=None, ax=None, cmap='inferno', sigma=1.5):

    smoothed_data = gaussian_filter(data, sigma=sigma) # Apply Gaussian smoothing

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure if no axes provided

    im = ax.imshow(smoothed_data, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap=cmap)
    fig.colorbar(im, ax=ax, label=title)  
    ax.set_title("Heatmap of " + title)
    ax.set_xlabel('Lambda (λ)')
    ax.set_ylabel('Rho (ρ)')
    return im