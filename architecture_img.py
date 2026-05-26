from IPython.display import Image, display
from langgraph import graph
from agent import _build_graph

# Build the graph
graph = _build_graph()

# Visualize the graph as a PNG image
png_img = graph.get_graph().draw_mermaid_png()

# Or, print the raw Mermaid markdown code
# print(graph.get_graph().draw_mermaid())

with open("graph_architecture.png", "wb") as f:
    f.write(png_img)