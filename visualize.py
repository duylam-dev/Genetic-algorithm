import matplotlib.pyplot as plt
import networkx as nx


def plot_pareto_2d(hof):
    costs = [ind.fitness.values[0] for ind in hof]
    accepted = [
        -ind.fitness.values[1] for ind in hof
    ]  # đổi dấu vì bạn lưu là -accepted

    plt.figure(figsize=(8, 6))
    plt.scatter(costs, accepted, c="blue", label="Pareto front")
    plt.xlabel("Total Resource Cost")
    plt.ylabel("Number of Accepted Requests")
    plt.title("Pareto Front: Cost vs Accepted Requests")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pareto_front.png")


def draw_network_graph(problem, filename="network_graph.png"):
    G = problem.G
    pos = nx.spring_layout(G, seed=42, k=5.0)  # Spread mạnh hơn

    plt.figure(figsize=(20, 15))  # Hình to hơn

    nx.draw_networkx_edges(G, pos, edge_color="black", width=1)
    nx.draw_networkx_nodes(
        G, pos, node_color="lightblue", node_size=600, edgecolors="black"
    )
    nx.draw_networkx_labels(
        G, pos, font_size=8, font_color="black"
    )  # Nhỏ lại để khỏi đè

    plt.title("Input Network Graph", fontsize=16)
    plt.axis("off")
    plt.savefig(filename, format="png", bbox_inches="tight")
    plt.close()
    print(f"Graph saved to {filename}")
