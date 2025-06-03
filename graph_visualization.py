import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from textwrap import wrap


class DraggableGraphWithSidebarLegend:
    def __init__(self, G, pos, labels, sentences, title="Граф схожести предложений", summary_indices=None):
        self.G = G
        self.pos = pos
        self.labels = labels
        self.sentences = sentences
        self.title = title
        self.summary_indices = summary_indices or []

        self.node_main_color = '#4e79a7'
        self.node_summary_color = '#f28e2c'
        self.edge_cmap = self.create_custom_cmap()
        self.font_color = '#333333'

        self.fig = plt.figure(figsize=(16, 10), facecolor='#f5f5f5')
        self.ax_graph = self.fig.add_axes([0.05, 0.05, 0.65, 0.90])
        self.ax_legend = self.fig.add_axes([0.71, 0.05, 0.28, 0.90])

        self.dragging_node = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.draw_graph(redraw_legend=True)
        plt.show()

    def create_custom_cmap(self):
        colors = ['#f0f0f0', '#aec7e8', '#1f77b4', '#023047']
        return LinearSegmentedColormap.from_list("custom_cmap", colors)

    def draw_graph(self, redraw_legend=True):
        self.ax_graph.clear()
        self.ax_graph.set_facecolor('#ffffff')


        weights = [d['weight'] for _, _, d in self.G.edges(data=True)] if self.G.edges else [0.5]
        min_weight, max_weight = min(weights), max(weights)
        norm = plt.Normalize(min_weight, max_weight)

        for (u, v, d) in self.G.edges(data=True):
            weight = d['weight']
            color = self.edge_cmap(norm(weight))
            width = 0.5 + 3.0 * norm(weight)
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax_graph,
                edgelist=[(u, v)],
                width=width,
                edge_color=color,
                alpha=0.9
            )

        node_colors = [self.node_summary_color if i in self.summary_indices
                       else self.node_main_color for i in self.G.nodes()]
        node_sizes = [3500 if i in self.summary_indices else 2800 for i in self.G.nodes()]
        border_colors = ['#ffffff' for _ in self.G.nodes()]

        self.nodes = nx.draw_networkx_nodes(
            self.G, self.pos, ax=self.ax_graph,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors=border_colors,
            linewidths=2.5,
            alpha=0.95
        )
        self.nodes.set_picker(True)
        self.nodes.set_pickradius(10)

        nx.draw_networkx_labels(
            self.G, self.pos, labels=self.labels,
            ax=self.ax_graph,
            font_size=10,
            font_weight='bold',
            font_color='white'
        )

        self.ax_graph.set_title(self.title,
                                fontsize=18,
                                fontweight='bold',
                                pad=20,
                                color=self.font_color)
        self.ax_graph.axis('off')

        if redraw_legend:
            self.draw_legend()

    def draw_legend(self):
        self.ax_legend.clear()
        self.ax_legend.set_facecolor('#ffffff')
        self.ax_legend.axis('off')

        self.ax_legend.text(
            0.01, 0.97, "Предложения:",
            fontsize=14,
            fontweight='bold',
            ha='left',
            va='top',
            color=self.font_color
        )

        y_position = 0.93
        base_font_size = 9
        line_height = 0.009
        wrap_width = 80

        for i, sentence in enumerate(self.sentences):
            if y_position < 0.02:
                break

            is_summary = i in self.summary_indices
            text_color = self.node_summary_color if is_summary else self.node_main_color
            font_weight = 'bold' if is_summary else 'normal'

            wrapped = wrap(f"S{i + 1}: {sentence}", width=wrap_width)
            for j, line in enumerate(wrapped):
                self.ax_legend.text(
                    0.01,
                    y_position - j * line_height,
                    line,
                    fontsize=base_font_size,
                    ha='left',
                    va='top',
                    color=text_color,
                    weight=font_weight
                )

            y_position -= len(wrapped) * line_height
            y_position -= 0.01

    def on_press(self, event):
        if event.inaxes != self.ax_graph:
            return

        if event.inaxes == self.ax_graph:
            cont, ind = self.nodes.contains(event)
            if cont:
                self.dragging_node = ind['ind'][0]
                self.drag_start_pos = (event.xdata, event.ydata)

    def on_release(self, event):
        self.dragging_node = None

    def on_motion(self, event):

        if self.dragging_node is None or event.inaxes != self.ax_graph:
            return


        x, y = event.xdata, event.ydata
        self.pos[self.dragging_node] = (x, y)

        self.draw_graph(redraw_legend=False)
        self.fig.canvas.draw_idle()


def visualize_similarity_graph(graph_matrix, sentences, ranks=None, title="Граф схожести предложений",
                               summary_indices=None):
    G = nx.Graph()
    for i, sentence in enumerate(sentences):
        label = f"S{i + 1}"
        if ranks is not None:
            label += f"\n{ranks[i]:.2f}"
        G.add_node(i, label=label)

    size = len(graph_matrix)
    for i in range(size):
        for j in range(i + 1, size):
            weight = graph_matrix[i, j]
            if weight > 0.01:
                G.add_edge(i, j, weight=weight)

    labels = nx.get_node_attributes(G, "label")

    k_value = 0.2 + 0.05 * len(sentences)
    pos = nx.spring_layout(G, seed=42, k=k_value, iterations=100)

    DraggableGraphWithSidebarLegend(G, pos, labels, sentences, title=title, summary_indices=summary_indices)