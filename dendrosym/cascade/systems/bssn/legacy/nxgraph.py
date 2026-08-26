"""
@author : Milinda Fernando
@brief  : Compute a directed graph from a sympy expression.
    1. Rename custom functions to symbols
    2. Graphs are merged from multiple expressions
NetworkX is required, and used to store the graph.
"""

from operator import is_
import matplotlib

matplotlib.use("Agg")
import sympy as sympy
import networkx as nx
import random
from collections import namedtuple

from typing import Dict, List, Optional, Any, Set, Tuple, Union

Community_Data = namedtuple("Community_Data", ["storage", "order", "adjacencies"])


def save_any_graph(G, filename="graph.png", title=None, ordering=None):
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = nx.spring_layout(G)

    if ordering is not None:
        label_map = {
            node: f"{i}: {str(G.nodes[node].get('func', node))}"
            for i, node in enumerate(ordering)
        }
    else:
        label_map = {node: str(G.nodes[node].get("func", node)) for node in G.nodes()}

    plt.figure(figsize=(10, 10))
    nx.draw_networkx(
        G,
        pos,
        labels=label_map,
        font_size=8,
        node_size=400,
        arrows=True,
        with_labels=True,
    )
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved graph as {filename}")


class ExpressionGraph:
    def __init__(self):
        self._sympy_expr: Dict[str, sympy.Expr] = dict()

        self._G_: nx.DiGraph = nx.DiGraph()
        self._hash_to_expr: Dict[int, sympy.Expr] = dict()
        self._output_expr_map: Dict[str, sympy.Expr] = dict()

        # this maps expression objects to node ids
        self._expr_to_id: Dict[sympy.Expr, int] = dict()

    def get_expr_from_hash(self, node_hash: int) -> Optional[sympy.Expr]:
        """
        Retrieves the original SymPy expression from the hash. Returns None if it isn't found
        """
        return self._hash_to_expr.get(node_hash, None)

    def map_outputs_to_blocks(self, blocks):
        """ """

        # reverse lookup for node hash to block idx
        node_to_block_idx = {}
        for i, (subgraph, _) in enumerate(blocks):
            for node_hash in subgraph.nodes():
                node_to_block_idx[node_hash] = i
        # print(node_to_block_idx)

        # map each output expression to its block
        output_to_block_map = {}
        for var_name, expr in self._sympy_expr.items():
            expr_hash = hash(expr)

            # figure out which block has it
            block_idx = node_to_block_idx.get(expr_hash)

            if block_idx is not None:
                output_to_block_map[var_name] = block_idx

            else:
                print(f"WARNING: couldn't find a block for output '{var_name}'")
                pass

        return output_to_block_map

    def __pre_traversal_1(self, expr, node_list, edge_list):
        """
        Preorder traversal of the expression converting undefined functions to sympy symbols
        """
        if isinstance(expr.func, sympy.core.function.UndefinedFunction):
            sym_name = str(expr.func)
            for a in expr.args:
                sym_name = sym_name + "_" + str(a)

            node_list.append(sympy.Symbol(sym_name))
        else:
            node_list.append(expr)

        for arg in expr.args:
            if isinstance(arg.func, sympy.core.function.UndefinedFunction):
                f = arg.func
                sym_name = str(f)
                for a in arg.args:
                    sym_name = sym_name + "_" + str(a)

                node_list.append(sympy.Symbol(sym_name))
                edge_list.append((expr, sympy.Symbol(sym_name)))
            else:
                edge_list.append((expr, arg))
                self.__pre_traversal_1(arg, node_list, edge_list)

    def __pre_traversal_2(self, expr):
        """
        Keep undefined function references as it is pruning but not renaming.
        """
        if expr in self._expr_to_id:
            return self._expr_to_id[expr]

        expr_hash = hash(expr)

        while expr_hash in self._hash_to_expr:
            if self._hash_to_expr[expr_hash] == expr:
                break

            # otherwise it's a collision and we'll perturb it to work
            expr_hash += 1

        # fix up the mappings
        self._expr_to_id[expr] = expr_hash
        self._hash_to_expr[expr_hash] = expr

        # then add to graph
        self._G_.add_node(expr_hash, func=expr.func, args=expr.args, eval=False)

        # recurse on children and add edges
        for arg in expr.args:
            arg_hash = self.__pre_traversal_2(arg)
            self._G_.add_edge(expr_hash, arg_hash)

        return expr_hash

    def __preorder_traversal__(self, expr):
        """
        Pre order traversal and returns the node list and edge list
        """
        expr_list = list()
        edge_list = list()
        self.__pre_traversal_2(expr, expr_list, edge_list)
        return [expr_list, edge_list]

    def add_expression(self, expr, expr_name):
        """
        Generate a networkx graph for a given expression
        """

        # recursively add the full set of expressions to the graph
        expr_hash = self.__pre_traversal_2(expr)
        self._sympy_expr[str(expr_name)] = expr

        # grab variable names for this node, and if the expression hash is re-used we fix it up
        if "vnames" not in self._G_.nodes[expr_hash]:
            self._G_.nodes[expr_hash]["vnames"] = []

        self._G_.nodes[expr_hash]["vnames"].append(str(expr_name))

        return expr_hash

    # WHOLE SET

    def add_expressions(self, outs, vnames, suffix_idx="[pp]"):
        """
        Adds list of sympy expressions
        """
        mi = [0, 1, 2, 4, 5, 8]
        midx = ["00", "01", "02", "11", "12", "22"]

        num_e = 0
        for i, e in enumerate(outs):
            if type(e) is list:
                num_e = num_e + len(e)
                for j, ev in enumerate(e):
                    expr_name = vnames[i] + "" + str(j) + str(suffix_idx)
                    # print("processing expr : %d var name %s[%s]" %(i,vnames[i],str(j)))
                    self.add_expression(ev, expr_name)
            elif type(e) is sympy.Matrix:
                num_e = num_e + len(e)
                for j, k in enumerate(mi):
                    expr_name = vnames[i] + "" + str(midx[j]) + str(suffix_idx)
                    # print("processing expr : %d var name %s[%s]" %(i,vnames[i],midx[j]))
                    self.add_expression(e[k], expr_name)
            else:
                num_e = num_e + 1
                # print("processing expr : %d var name %s" %(i,vnames[i]))
                expr_name = vnames[i] + str(suffix_idx)
                self.add_expression(e, expr_name)

    def composed_graph(self, verbose=False):
        """
        compute the composed graph for all the added expressions
        """

        if verbose:
            print("Full graph (built incrementally)")
            print(f"  Nodes: {self._G_.number_of_nodes()}")
            print(f"  Edges: {self._G_.number_of_edges()}")

        print(
            f"Composed graph: {self._G_.number_of_nodes()} nodes, {self._G_.number_of_edges()} edges"
        )

        return self._G_

    def set_output_expressions(self, mapping):
        """
        Stores a dictionary mapping of the final output variable names to their corresponding post-CSE sympy expressions.
        """
        self._output_expr_map = mapping

    def plot_adjmatrix(self):
        """
        plots the adjacency matrix for the composed big G
        """
        import matplotlib.pyplot as plt

        print(">> Plotting adjacency matrix...")
        A = nx.adjacency_matrix(self._G_)
        plt.spy(A, markersize=3)

        # New stuff (trying to get it to work on terminal)
        plt.title("Adjacency Matrix")
        plt.savefig("adj_matrix_plot.png", dpi=300)
        print(">> Saved plot to 'adj_matrix_plot.png'")

    def draw_graph(self, expr_name):
        """
        plots the graph for a given sympy expression
        """
        import matplotlib.pyplot as plt

        g = self._nx_graphs[expr_name]

        labels = {
            node: str(data["func"]) if "func" in data else str(node)
            for node, data in g.nodes(data=True)
        }

        pos = nx.planar_layout(g)
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(g, pos, node_size=500)
        nx.draw_networkx_edges(g, pos, arrows=True)
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=6)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{expr_name}_graph.png", bbox_inches="tight")
        plt.close()
        print(f"Saved graph as {expr_name}_graph.png")

    def save_all_graphs(self, output_dir="."):
        """
        Draws and saves a PNG image for each added expression graph
        """
        import os
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        for expr_name, g in self._nx_graphs.items():
            plt.figure(figsize=(8, 6))
            nx.draw_networkx(g, pos=nx.planar_layout(g), font_size=6)
            output_path = os.path.join(output_dir, f"graph_{expr_name}.png")
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            print(f"Saved graph: {output_path}")

    def debug_print_nodes(graph, expr_name):
        print(f"\n=== Debug: Nodes in expression graph '{expr_name}' ===")
        g = graph._nx_graphs[expr_name]

        for node, data in g.nodes(data=True):
            print(f"Node: {node!r}")
            for key, value in data.items():
                print(f"  {key}: {value!r}")
            print()

    def save_composed_graph(self, filename="composed_graph.png"):
        import matplotlib.pyplot as plt

        if not hasattr(self, "_G_"):
            print("No composed graph exists. Run composed_graph() first.")
            return

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self._G_)

        # Create meaningful labels
        labels = {}
        for node, data in self._G_.nodes(data=True):
            if "func" in data:
                func_name = data["func"].__name__
                if func_name == "Symbol" and data["args"]:
                    labels[node] = str(data["args"][0])
                elif func_name == "Integer" and not data["args"]:
                    labels[node] = str(node)
                else:
                    labels[node] = func_name
            else:
                labels[node] = str(node)

        nx.draw_networkx(
            self._G_,
            pos,
            labels=labels,
            font_size=8,
            node_size=400,
            arrows=True,
            with_labels=True,
        )

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved composed graph as {filename}")

    def get_graph(self, expr_name):
        """
        Get the computational graph for a given expression
        """
        g = self._nx_graphs[expr_name]
        return g

    ############################################################################
    ############################################################################
    # New Stuff
    ############################################################################
    ############################################################################

    def count_dependents(self):
        """
        Adds a 'num_dependents' attribute to each node in the composed graph.
        This is the number of nodes that directly depend on the current node
        (i.e., the number of immediate parents in the graph).
        """
        if not hasattr(self, "_G_"):
            raise ValueError("Call composed_graph() before computing dependents.")

        for node in self._G_.nodes:
            dependents = list(self._G_.predecessors(node))
            self._G_.nodes[node]["num_dependents"] = len(dependents)

    def max_storage(self, ordering):
        """
        Takes in a topological ordering and returns the maximum number of temporary
        variables needed throughout the traversal computation
        """
        if not hasattr(self, "_G_"):
            raise ValueError("Call composed_graph() first.")

        uses_remaining = {n: self._G_.in_degree(n) for n in self._G_.nodes}
        active_storage = set()
        max_storage = 0
        for node in ordering:
            is_computed_temp = self._G_.out_degree(node) > 0

            if is_computed_temp:
                active_storage.add(node)

            max_storage = max(max_storage, len(active_storage))

            for dep in self._G_.successors(node):
                if dep in uses_remaining:
                    uses_remaining[dep] -= 1

                    if uses_remaining[dep] == 0:
                        if dep in active_storage:
                            active_storage.remove(dep)

        return max_storage, []

    def random_dfs_sort(self, relevant_nodes=None, seed=None):
        """
        returns a valid reverse topological sort using randomized dfs.
        pass a seed for deterministic output.
        """
        if not hasattr(self, "_G_"):
            raise ValueError("Call composed_graph() first.")

        rng = random.Random(seed)

        visited = set()
        ordering = []

        nodes_to_include = set(relevant_nodes) if relevant_nodes is not None else None

        def dfs(node):
            if node in visited:
                return
            visited.add(node)

            children = list(self._G_.successors(node))
            rng.shuffle(children)

            for child in children:
                dfs(child)

            if nodes_to_include is None or node in nodes_to_include:
                ordering.append(node)

        # start from leaves (in_degree == 0 in our edge convention)
        leaves = [n for n in self._G_.nodes if self._G_.in_degree(n) == 0]
        rng.shuffle(leaves)

        for r in leaves:
            dfs(r)

        # catch disconnected components
        all_nodes = list(self._G_.nodes)
        rng.shuffle(all_nodes)
        for node in all_nodes:
            if node not in visited:
                dfs(node)

        return ordering

    def plot_storage_trace(
        self, storage_data, filename="storage_trace.png", output_dir="."
    ):
        """
        (Not really using anymore)
        Saves a line plot of the storage usage trace over time.
        """
        import os
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        plt.figure(figsize=(10, 5))
        plt.plot(storage_data, marker="o", markersize=2, linewidth=1)
        plt.title("Storage Usage Over Time")
        plt.xlabel("Computation Step")
        plt.ylabel("Storage Used")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f">> Saved storage trace plot to: {output_path}")

    def get_communities(self):
        """
        Returns a list of communities (each as a set of nodes) using greedy modularity.
        Converts the directed composed graph to undirected for community detection.
        """
        from networkx.algorithms.community import greedy_modularity_communities

        if not hasattr(self, "_G_"):
            raise ValueError("Call composed_graph() before getting communities.")

        # Community detection requires an undirected graph
        undirected = self._G_.to_undirected()
        communities = list(greedy_modularity_communities(undirected))

        print(f"Detected {len(communities)} communities.")
        for i, comm in enumerate(communities):
            print(f"   - Community {i}: {len(comm)} nodes")

        return communities

    def get_cluster_io(self, cluster_nodes):
        """
        Given a set of cluster nodes, returns:
        - inputs: nodes outside the cluster with edges into the cluster
        - outputs: nodes inside the cluster with edges to outside
        """
        if not hasattr(self, "_G_"):
            raise ValueError("Call composed_graph() first.")

        inputs = set()
        outputs = set()

        for node in cluster_nodes:
            # Find INPUTS: check dependencies of successors
            for succ in self._G_.successors(node):
                if succ not in cluster_nodes:
                    # succ is a dependency that isn't in this block!
                    # therefore it's an INPUT
                    inputs.add(succ)

            # Find OUTPUTS: check dependencies of predecessors
            for pred in self._G_.predecessors(node):
                if pred not in cluster_nodes:
                    # pred is an expression outside this block that *uses* 'node'.
                    # therefore its an output
                    outputs.add(node)
                    # only add it once
                    break

        return inputs, outputs

    def analyze_subgraph_storage(self, community):
        """
        Outputs # of nodes, edges, inputs, outputs, and min storage approximation
        for a given subgraph
        """
        subgraph = self._G_.subgraph(community).copy()
        if not nx.is_connected(subgraph.to_undirected()):
            print("WARNING: Community is not connected")

        sub_expr_graph = ExpressionGraph()
        sub_expr_graph._G_ = subgraph
        order, storage = sub_expr_graph.optimize_storage_genetic()

        # Print out attributes
        print(
            f"Nodes: {subgraph.number_of_nodes()} | Edges: {subgraph.number_of_edges()}"
        )
        inputs, outputs = self.get_cluster_io(community)
        print(f"Inputs: {len(inputs)} | Outputs: {len(outputs)}")
        print(f"Min storage approximation: {storage}")

        return Community_Data(storage, order, inputs | outputs), sub_expr_graph

    def generate_clusters(
        self, blocks=None, storage_max=33, storage_min=27, generation="", original=True
    ):  # remaining_graph = None,
        """
        Generates blocks/clusters
        """
        # set-up
        import os

        if not hasattr(self, "_G_"):
            raise ValueError("Call composed_graph() before analyzing clusters.")

        if original:
            print("Running quick estimate of graph storage...")
            quick_order = self.random_dfs_sort()
            estimated_storage, _ = self.max_storage(quick_order)

            print(f"Quick estimate of graph: {estimated_storage}")

            if estimated_storage > storage_max:
                print(
                    f"Full graph optimal storage ({estimated_storage}) exceends limit ({storage_max})"
                )
                print("Proceeding with clustering heuristic")
            else:
                print(
                    "Estimate of graph was within storage minimum, running full analysis to be sure..."
                )
                full_c_data, full_c_graph = self.analyze_subgraph_storage(
                    self._G_.nodes()
                )
                full_storage = full_c_data.storage

                if full_storage <= storage_max:
                    print(
                        f"Full graph optimal storage ({full_storage}) is within limit ({storage_max})"
                    )
                    print("Bypassing clustering and returning full graph as one block")
                    return [(self._G_, full_c_data)]
                else:
                    print(
                        "Full graph storage exceeds limit! Proceeding with clustering heuristic"
                    )

        if blocks is None:
            blocks = []

        # clusters not yet part of a block
        bachelors = dict()

        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(self._G_.to_undirected()))

        for i, comm in enumerate(communities):
            label = f"{generation}.{i}" if generation else str(i)
            print(f"\n== Community {label} ==")

            c_data, c_graph = self.analyze_subgraph_storage(comm)
            storage = c_data.storage
            subgraph = c_graph._G_

            if storage > storage_max:
                c_graph.generate_clusters(
                    generation=label,
                    blocks=blocks,
                    storage_max=storage_max,
                    storage_min=storage_min,
                    original=False,
                )
            elif storage >= storage_min:
                blocks.append((subgraph, c_data))
            else:
                bachelors[subgraph] = c_data

        # creating dictionary of node origins...
        node_origins = dict()
        b_snapshot = set(bachelors.keys())
        for subgraph in b_snapshot:
            for node in subgraph:
                node_origins[node] = subgraph

        for subgraph in list(b_snapshot):
            adj_nodes = bachelors[subgraph].adjacencies
            # remap to subgraphs and normalize
            remapped = {
                node_origins[n]
                for n in list(adj_nodes)
                if n in node_origins and node_origins[n] is not subgraph
            }

            remapped.intersection_update(b_snapshot)
            adj_nodes.clear()
            adj_nodes.update(remapped)

        # EXPERIMENTAL
        import heapq

        while bachelors:
            ####################################################################
            # LOOP SET-UP
            ####################################################################
            # important graphs
            subgraph = None
            neighbor = None
            c_graph = None
            c_data = None

            # extract min
            count = float("inf")
            for b in bachelors:
                if count > len(bachelors[b].adjacencies):
                    count = len(bachelors[b].adjacencies)
                    subgraph = b

            print(f"[Debug] Chose subgraph with {count} adjacencies")

            # subgraph attributes
            data = bachelors[subgraph]
            adj = data.adjacencies

            ####################################################################
            # functions (ON LOOP VARIABLES)
            ####################################################################
            # add subgraph to blocks
            def add_block():
                blocks.append((subgraph, data))
                del bachelors[subgraph]  # bachelor removal
                for other_subgraph in bachelors:  # adjacency removal
                    bachelors[other_subgraph].adjacencies.discard(subgraph)

                print(f"[Debug] Added graph w/ {data.storage} storage")

            # return data on a combined graph
            def combine_graph():
                # candidate_data = bachelors[neighbor]
                combined_nodes = set(subgraph.nodes()) | set(neighbor.nodes())
                combined_data, combined_graph = self.analyze_subgraph_storage(
                    combined_nodes
                )

                print(f"[Debug] Trial Combo: graph w/ {combined_data.storage} storage")

                return combined_data, combined_graph._G_

            # adds combination to bachelor, replaces trace of components
            def insert_combo_bachelor():
                del bachelors[subgraph]
                del bachelors[neighbor]

                bachelors[c_graph] = c_data

                # replace w / new...
                for b in bachelors:
                    old_size = len(bachelors[b].adjacencies)
                    bachelors[b].adjacencies.discard(subgraph)
                    bachelors[b].adjacencies.discard(neighbor)

                    if old_size != len(bachelors[b].adjacencies):
                        bachelors[b].adjacencies.add(c_graph)
                        bachelors[c_graph].adjacencies.add(b)

                print(f"[Debug] Saved combo")

            ####################################################################
            # iterative process
            ####################################################################
            if count == 0:
                add_block()
            elif count == 1:
                neighbor = next(iter(adj))
                c_data, c_graph = combine_graph()
                if c_data.storage > storage_max:
                    add_block()
                else:
                    insert_combo_bachelor()
            else:
                add_block()
                # working on integrating code below here...

                # # neighbor pq (inverse size)
                # neighbor_heap = []
                # for n in adj:
                #     c = len(bachelors[n].adjacencies)
                #     heapq.heappush(neighbor_heap, (c, id(n), n))

                # done = False
                # while neighbor_heap and not done:
                #     _, _, neighbor = heapq.heappop(neighbor_heap)
                #     c_data, c_graph = combine_graph(neighbor)

                #     if (c_data.storage > storage_max): # too much
                #         continue
                #     if (c_data.storage >= storage_min): # terminate
                #         add_block_combo()
                #         done = True
                #     else: # keep going
                #         insert_combo_bachelor()
                #         # have to update since we keep doing here...
                #         subgraph = c_graph._G_
                #         data = c_data

        return blocks

    ############################################################################
    # Top Sort Selection
    ############################################################################

    def build_block_graph(
        self, blocks: List[Tuple[nx.DiGraph, Community_Data]]
    ) -> nx.DiGraph:
        """
        Build a directed graph whose nodes are block indices (0..len(blocks)-1)
        and whose edges represent data dependencies between blocks.

        There is an edge i -> j if there exists an edge u -> v in the original
        composed graph self._G_ with u in block i and v in block j.

        NOTE: This graph may have cycles even if self._G_ is a DAG, because
        clustering can create mutual dependencies between blocks. We handle
        that in block_traversal_order via SCC condensation.
        """
        B = nx.DiGraph()
        for idx, (g, c_data) in enumerate(blocks):
            B.add_node(
                idx,
                num_nodes=g.number_of_nodes(),
                storage=c_data.storage,
            )

        # Map each original node hash to its block index
        node_to_block = {}
        for idx, (g, _) in enumerate(blocks):
            for n in g.nodes():
                if n in node_to_block:
                    raise RuntimeError(f"Node {n!r} appears in multiple blocks.")
                node_to_block[n] = idx

        # Add edges between blocks based on edges in the original global graph
        for u, v in self._G_.edges():
            bu = node_to_block.get(u)
            bv = node_to_block.get(v)

            # If this happens, some node in self._G_ wasn't assigned to any block
            if bu is None or bv is None:
                continue

            # If the edge connects two different blocks, record the dependency
            if bu != bv:
                B.add_edge(bu, bv)

        # Optional debug: report if block graph has cycles
        # if not nx.is_directed_acyclic_graph(B):
        #     print("[Block Graph] Note: block graph is not a DAG; "
        #           "will resolve via SCC condensation in block_traversal_order().")

        return B

    def block_traversal_order(self, blocks: List[Tuple[nx.DiGraph, Community_Data]]):
        """
        Returns:
          block_order:  a list of block indices in a DAG-consistent order of
                       *SCC components*; for multi-block SCCs, their members
                       are appended together in that component's position.
          global_order: a flat list of node hashes (original graph nodes) in a
                        traversal that respects all dependencies in self._G_.

        Strategy:
          1) Build the block-level graph B (blocks as nodes, edges = deps).
          2) Condense B into a DAG of strongly-connected components (SCCs).
          3) Topologically sort the SCC-DAG.
          4) For each SCC:
             - if it has 1 block: use that block's internal ordering
               (c_data.order if available, else topo-sort of that block).
             - if it has multiple blocks: union all their nodes, build a
               subgraph, and compute an ordering on that combined subgraph
               (using optimize_storage_genetic).
                - > May want to recurse on this later
        """
        if not blocks:
            return [], []

        # 1) Build block-level graph
        B = self.build_block_graph(blocks)

        # 2) Condense into SCC-DAG
        #    C's nodes are 0..k-1, each has attribute "members" = set of block indices
        C = nx.condensation(B)

        # Topologically sort SCCs
        scc_order = list(nx.topological_sort(C))

        global_order = []
        block_order = []

        for comp in scc_order:
            members = C.nodes[comp]["members"]  # set of block indices in this SCC
            members_sorted = sorted(members)

            if len(members_sorted) == 1:
                # Simple case: single block SCC
                idx = members_sorted[0]
                block_order.append(idx)

                g, c_data = blocks[idx]

                # If we already computed a valid order for this block, use it
                if (
                    c_data.order is not None
                    and len(c_data.order) == g.number_of_nodes()
                ):
                    local_order = list(c_data.order)
                else:
                    # Fallback: topo-sort inside the block’s subgraph
                    local_order = list(nx.topological_sort(g))
                    # NOTE: this might fail if we are don't have a DAG, might want a fallback?

                global_order.extend(local_order)

            else:
                # Multi-block SCC: combine all their nodes and compute a joint order
                print(
                    f"[Block Order] SCC with {len(members_sorted)} blocks: {members_sorted}"
                )

                # Record these blocks in block_order (flat)
                block_order.extend(members_sorted)

                # Union of nodes in all these blocks
                combined_nodes = set()
                for idx in members_sorted:
                    g, _ = blocks[idx]
                    combined_nodes.update(g.nodes())

                # Build subgraph of the composed graph on those nodes
                subgraph = self._G_.subgraph(combined_nodes).copy()

                # Use a fresh ExpressionGraph to run the genetic optimizer on the SCC
                sub_expr = ExpressionGraph()
                sub_expr._G_ = subgraph

                # Re-run optimization on the fused blocks, i opted for less aggressive options, but code could do defaults
                combined_order, combined_storage = sub_expr.optimize_storage_genetic(
                    pop_size=50, generations=10
                )

                print(
                    f"[Block Order] Combined SCC storage (approx): {combined_storage}"
                )
                global_order.extend(combined_order)

        return block_order, global_order

    def optimize_storage_genetic(
        self, initialization_coefficient=3, pop_size=1000, generations=50
    ):
        """
        Genetic algorithm for selecting ideal ordering.
        Returns ordering and max_storage of that ordering.
        """

        def fitness(ordering):
            return -self.max_storage(ordering)[0]

        # creating initial population
        initial_pool = [
            self.random_dfs_sort() for _ in range(pop_size * initialization_coefficient)
        ]
        scored_pool = [(fitness(ind), ind) for ind in initial_pool]
        scored_pool.sort(reverse=True)
        population = [ind for _, ind in scored_pool[:pop_size]]

        # precompute this so it's not calculating out degrees during each cross over
        base_out_degrees = {n: self._G_.out_degree(n) for n in self._G_.nodes}

        # defining crossover
        def crossover(p1, p2):
            import heapq
            from collections import defaultdict

            # index maps for each parent
            index1 = {node: i for i, node in enumerate(p1)}
            index2 = {node: i for i, node in enumerate(p2)}

            # make sure we copy it, because we don't want to edit the base version
            out_degree_counts = base_out_degrees.copy()

            queue = []
            for node, out_deg in out_degree_counts.items():
                if out_deg == 0:
                    # find the leaves inputs
                    prio = min(
                        index1.get(node, float("inf")), index2.get(node, float("inf"))
                    )
                    heapq.heappush(queue, (prio, node))

            result = []
            while queue:
                _, node = heapq.heappop(queue)
                result.append(node)

                # iterate over the consumers:
                for consumer in self._G_.predecessors(node):
                    out_degree_counts[consumer] -= 1
                    # if dependencies are all met:
                    if out_degree_counts[consumer] == 0:
                        prio = min(
                            index1.get(consumer, float("inf")),
                            index2.get(consumer, float("inf")),
                        )
                        heapq.heappush(queue, (prio, consumer))

            return result

        # actual genetic algorithm
        best_ordering = None
        best_fitness = float("-inf")

        # process generations
        for gen in range(generations):
            scored = [(fitness(ind), ind) for ind in population]
            scored.sort(reverse=True)

            # updating best
            if scored[0][0] > best_fitness:
                best_fitness = scored[0][0]
                best_ordering = scored[0][1][:]

            # Select top 50% to reproduce
            survivors = [ind for _, ind in scored[: pop_size // 2]]

            # Generate children
            children = []
            while len(children) < pop_size // 2:
                p1, p2 = random.sample(survivors, 2)
                child = crossover(p1, p2)
                children.append(child)

            population = survivors + children
            # print(f"Generation {gen}: best storage = {-best_fitness}")

        return best_ordering, -best_fitness

    def partition_by_register_pressure(self, max_regs=25, num_trials=10):
        if not hasattr(self, "_G_"):
            raise ValueError(
                "Please be sure to call composed_graph() before this function!"
            )

        if num_trials > 1:
            best_blocks = None
            best_scratch_count = float("inf")

            for trial in range(num_trials):
                candidate = self._partition_by_register_pressure_single(max_regs, seed=trial)
                # fewer inter-block outputs = less data shuffling between blocks
                scratch_count = sum(len(b["outputs"]) for b in candidate)
                if scratch_count < best_scratch_count:
                    best_scratch_count = scratch_count
                    best_blocks = candidate

            print(f"[Register Partitioner] Best of {num_trials} trials: {len(best_blocks)} blocks, {best_scratch_count} inter-block outputs")
            return best_blocks

        return self._partition_by_register_pressure_single(max_regs)

    def _partition_by_register_pressure_single(self, max_regs=25, seed=None):
        sorted_nodes = self.random_dfs_sort(seed=seed)

        blocks = []
        current_block_nodes = []

        # this effectively tracks global remaining uses to know what var is really "dead"
        # this also keeps us from counting a var as live if we just consumed the last use
        global_uses_remaining = {n: self._G_.out_degree(n) for n in sorted_nodes}

        # set of vals currently held in registers (computed locally or loaded inputs)
        live_set = set()

        for node in sorted_nodes:
            # identify inputs required by this node, dependencies are the successors, we consume children
            children = list(self._G_.successors(node))

            # inputs we need to bring in (if not live)
            new_inputs = {p for p in children if p not in live_set}

            # identify what dies immediately
            dying_children = set()
            for c in children:
                # last global use = dead, but don't decrement yet
                if global_uses_remaining[c] == 1:
                    dying_children.add(c)

            # estimate peak pressure for this instruction
            peak_pressure = len(live_set) + len(new_inputs) + 1

            # decision time, do we cut the graph or keep?

            # if pressure is too high and we made progress in the block
            if peak_pressure > max_regs and len(current_block_nodes) > 0:
                # we commit it all
                subgraph = self._G_.subgraph(current_block_nodes).copy()
                inputs, outputs = self.get_cluster_io(current_block_nodes)

                blocks.append(
                    {
                        "subgraph": subgraph,
                        "inputs": inputs,
                        "outputs": outputs,
                        "id": f"RegBlock_{len(blocks)}",
                        "data": Community_Data(peak_pressure, [], []),
                    }
                )

                # reset for the new block
                current_block_nodes = []

                # live set effectively spills to memory (scratchpad), for next block these should be inputs
                live_set.clear()

                # re-calculate inputs for the current node (since we reset)
                # they all are inputs from memory
                new_inputs = set(children)

            # commit everything
            current_block_nodes.append(node)

            # update live set
            live_set.update(new_inputs)
            live_set.add(node)

            # update usage counts and kill dead variables
            for c in children:
                global_uses_remaining[c] -= 1
                if global_uses_remaining[c] == 0:
                    if c in live_set:
                        live_set.remove(c)

        # append the final block
        if current_block_nodes:
            subgraph = self._G_.subgraph(current_block_nodes).copy()
            inputs, outputs = self.get_cluster_io(current_block_nodes)
            blocks.append(
                {
                    "subgraph": subgraph,
                    "inputs": inputs,
                    "outputs": outputs,
                    "id": f"RegBlock_{len(blocks)}",
                    "data": Community_Data(0, [], []),
                }
            )

        return blocks
