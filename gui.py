import os
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog as fd
import tkinter as tk
from PIL import Image, ImageTk
from torch_geometric import loader
from torch_geometric.utils import from_networkx
from torch_geometric.nn import NNConv, global_mean_pool
from tkinter.filedialog import askopenfilename
import threading
import matplotlib

matplotlib.use("Agg")


def create_folders():
    directories = ["Graphs", "Graphs/2D", "Graphs/3D", "Graphs/PT", "Results"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


# Global variables
model = None
df = None
distance_threshold = 0.35
graph_folder = "Graphs"
avg_graph_creation_time = None
avg_prediction_time = None
total_graph_count = 0
# Global variable for current page
current_page_2d = 0
per_page = 5  # Number of images to show per page
# Global variable for current page
current_page_3d = 0
per_page = 5  # Number of images to show per page


class GNN(torch.nn.Module):
    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        hidden_channels,
        mxene_feature_dim=None,
    ):
        super(GNN, self).__init__()

        # Define edge update networks
        edge_network1 = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, node_feature_dim * hidden_channels),
        )
        edge_network2 = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels * hidden_channels),
        )

        # Graph Convolution Layers
        self.conv1 = NNConv(
            node_feature_dim, hidden_channels, edge_network1, aggr="mean"
        )
        self.conv2 = NNConv(
            hidden_channels, hidden_channels, edge_network2, aggr="mean"
        )

        # Linear layer for graph-level features if provided
        if mxene_feature_dim:
            self.fc_mxene = nn.Linear(mxene_feature_dim, hidden_channels)
        else:
            self.fc_mxene = None

        # Fully connected output layer
        self.fc_out = nn.Linear(
            hidden_channels * 2 if mxene_feature_dim else hidden_channels, 2
        )

    def forward(self, data):
        # Extract components from `data`
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Ensure the correct dtype
        x = x.float()
        edge_attr = edge_attr.float()

        # Debugging print statements
        # print(f"Node features shape: {x.shape}")
        # print(f"Edge index shape: {edge_index.shape}")
        # print(f"Edge attributes shape: {edge_attr.shape}")

        # Apply Graph Convolution layers with edge attributes
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        # print(f"Node embeddings shape after conv1: {x.shape}")

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        # print(f"Node embeddings shape after conv2: {x.shape}")

        # Apply global mean pooling to get graph-level representation
        graph_features = global_mean_pool(x, data.batch)
        # print(f"Graph-level representation shape after pooling: {graph_features.shape}")

        # Incorporate mxene-level properties if available
        if self.fc_mxene:
            mxene_properties = data.mxene_properties.float()
            # print(f"mxene_properties shape before reshape: {mxene_properties.shape}")

            # Ensure mxene_properties has shape [batch_size, mxene_feature_dim]
            batch_size = graph_features.size(
                0
            )  # Batch size inferred from graph-level representation
            mxene_properties = mxene_properties.view(
                batch_size, -1
            )  # Reshape to [batch_size, mxene_feature_dim]
            # print(f"mxene_properties shape after reshape: {mxene_properties.shape}")

            mxene_features = self.fc_mxene(mxene_properties)
            # print(f"mxene_features shape after fc_mxene: {mxene_features.shape}")

            # Concatenate graph features and mxene features
            graph_features = torch.cat([graph_features, mxene_features], dim=1)
            # print(f"Graph features shape after concatenation: {graph_features.shape}")

        # Final output layer
        output = self.fc_out(graph_features)
        # print(f"Final output shape: {output.shape}")

        return output


def create_scrollable_frame(parent):
    canvas = Canvas(parent, bg="#2c2f36")
    scrollbar = Scrollbar(parent, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas, bg="#2c2f36")

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    return scrollable_frame


# Function to load the model from a .pt file
def load_model():
    global model  # Access the global variable

    # Open file dialog to select the .pt file
    model_file = askopenfilename(
        title="Select Model", filetypes=[("PyTorch Models", "*.pt")]
    )
    if model_file:
        try:
            # Attempt to load the full model
            model = torch.load(model_file)
            model.eval()  # Set the model to evaluation mode
            messagebox.showinfo("Success", "Full model loaded successfully!")
            print(f"Full model loaded from {model_file}")
        except Exception as e1:
            print(f"Failed to load full model: {e1}")
            try:

                hidden_channels = 64
                learning_rate = 0.001
                node_feature_dim = 15
                edge_feature_dim = 1
                mxene_feature_dim = 6

                model = GNN(
                    node_feature_dim,
                    edge_feature_dim,
                    hidden_channels,
                    mxene_feature_dim,
                )
                model.load_state_dict(torch.load(model_file))  # Load state dictionary
                model.eval()  # Set the model to evaluation mode
                messagebox.showinfo("Success", "State dictionary loaded successfully!")
                print(f"State dictionary loaded from {model_file}")
            except Exception as e2:
                # If both loading attempts fail, display the error
                messagebox.showerror(
                    "Error", f"Failed to load the model or state_dict:\n{e2}"
                )
                print(f"Error loading state dictionary: {e2}")
    else:
        messagebox.showwarning("No file selected", "No model file was selected.")


# Function to open the Excel or CSV file and read it into a DataFrame
def open_file():
    global df  # Access the global variable

    # Open file dialog to select the Excel file
    excel_file = askopenfilename(
        title="Select Excel File", filetypes=[("Excel Files", "*.xlsx;*.xls")]
    )

    if excel_file:
        try:
            # Load the selected Excel file into a DataFrame
            df = pd.read_excel(excel_file)
            messagebox.showinfo("Success", "Excel file loaded successfully!")
            print(f"Excel file loaded from {excel_file}")
            print(
                df.head()
            )  # Display the first few rows of the DataFrame for confirmation
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the Excel file: {e}")
    else:
        messagebox.showwarning("No file selected", "No Excel file was selected.")


def start_graph_creation():
    # Check if df is loaded
    if df is None:
        messagebox.showerror(
            "Error", "No Excel file is loaded. Please load an Excel file first."
        )
        return
    messagebox.showinfo(
        "Graph Creation",
        "Graph creation process started! It may take a few minutes depending on the number of MXenes.",
    )

    def run_graph_creation():
        start_time = time.time()
        mxene_names = df["Mxene"].unique()
        graph_count = 0
        for mxene in mxene_names:
            mxene_subset = df[
                df["Mxene"] == mxene
            ].copy()  # Explicit copy to avoid view
            atom_names = mxene_subset["Atom"]
            atom_properties = mxene_subset[
                [
                    "Atom",
                    "X",
                    "Y",
                    "Z",
                    "Charge",
                    "Sigma",
                    "Epsilon",
                    "AtomicNumber",
                    "AtomicRadius",
                    "Electronegativity",
                    "Atomic Weight",
                    "Electron Affinity",
                    "Ionization Energy",
                    "Polarizability",
                    "Oxidation State",
                    "Coordination Number",
                    "Atomic Volume",
                    "Thermal Conductivity",
                    "Specific Heat Capacity",
                ]
            ].copy()  # Ensure a copy
            mxene_properties = (
                mxene_subset[["Cell_a", "Cell_b", "Cell_c", "Alpha", "Beta", "Gamma"]]
                .iloc[0]
                .values
            )
            target = mxene_subset[["1Bar_CO2", "1Bar_CH4"]].iloc[0].values

            # Ensure all atoms are unique (handle duplicate atoms)
            atom_properties["Atom"] = (
                atom_properties.groupby("Atom")
                .cumcount()
                .add(1)
                .astype(str)
                .radd(atom_properties["Atom"])
            )

            # Create graph
            G = nx.Graph()
            for i, row in atom_properties.iterrows():
                G.add_node(
                    row["Atom"], **row.drop("Atom").to_dict()
                )  # Add atom properties as node features

            # Add edges based on Euclidean distance
            for u in G.nodes:
                u_data = G.nodes[u]
                min_distance = float("inf")
                closest_node = None

                for v in G.nodes:
                    if u != v:  # Skip self-connections
                        v_data = G.nodes[v]
                        distance = np.sqrt(
                            (u_data["X"] - v_data["X"]) ** 2
                            + (u_data["Y"] - v_data["Y"]) ** 2
                            + (u_data["Z"] - v_data["Z"]) ** 2
                        )

                        if distance < min_distance:
                            min_distance = distance
                            closest_node = v

                if closest_node is not None:
                    G.add_edge(u, closest_node, distance=min_distance)

            # Identify connected components
            components = list(
                nx.connected_components(G)
            )  # List of components (sets of nodes)

            # Connect components by adding edges between their closest nodes
            while len(components) > 1:
                component_1 = components.pop()
                component_2 = components.pop()

                min_distance = float("inf")
                closest_pair = None

                for u in component_1:
                    u_data = G.nodes[u]

                    for v in component_2:
                        v_data = G.nodes[v]

                        # Calculate the distance between nodes from different components
                        distance = np.sqrt(
                            (u_data["X"] - v_data["X"]) ** 2
                            + (u_data["Y"] - v_data["Y"]) ** 2
                            + (u_data["Z"] - v_data["Z"]) ** 2
                        )

                        if distance < min_distance:
                            min_distance = distance
                            closest_pair = (u, v)

                # Add the edge between the closest nodes of the two components
                if closest_pair is not None:
                    u, v = closest_pair
                    G.add_edge(u, v, distance=min_distance)

                # Add the newly connected component to the list of components
                new_component = component_1.union(component_2)
                components.append(new_component)

            # Add edges based on distance threshold
            for u in G.nodes:
                u_data = G.nodes[u]
                for v in G.nodes:
                    if u != v:
                        v_data = G.nodes[v]
                        distance = np.sqrt(
                            (u_data["X"] - v_data["X"]) ** 2
                            + (u_data["Y"] - v_data["Y"]) ** 2
                            + (u_data["Z"] - v_data["Z"]) ** 2
                        )
                        if distance < distance_threshold:
                            G.add_edge(u, v, distance=distance)

            # Convert to torch_geometric graph
            graph = from_networkx(G)

            # Extract node features (charge, sigma, epsilon, etc.)
            node_features = atom_properties[
                [
                    "Charge",
                    "Sigma",
                    "Epsilon",
                    "AtomicNumber",
                    "AtomicRadius",
                    "Electronegativity",
                    "Atomic Weight",
                    "Electron Affinity",
                    "Ionization Energy",
                    "Polarizability",
                    "Oxidation State",
                    "Coordination Number",
                    "Atomic Volume",
                    "Thermal Conductivity",
                    "Specific Heat Capacity",
                ]
            ].values

            # Ensure the features are in the shape (num_nodes, 15)
            graph.x = torch.tensor(node_features, dtype=torch.float)
            # Extract edge distances and add as 'edge_attr'
            edge_distances = nx.get_edge_attributes(G, "distance")
            edge_distances = np.array([edge_distances[edge] for edge in G.edges])
            graph.edge_attr = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(
                -1
            )
            del graph.distance

            # Extract node positions (X, Y, Z) and add as 'pos'
            node_positions = atom_properties[["X", "Y", "Z"]].values
            graph.pos = torch.tensor(node_positions, dtype=torch.float)

            # Target variable (1Bar_CO2, 1Bar_CH4) for regression
            graph.y = torch.tensor(target, dtype=torch.float)

            # MXene properties
            graph.mxene_properties = torch.tensor(mxene_properties, dtype=torch.float)
            del graph.X
            del graph.Y
            del graph.Z
            del graph.Charge
            del graph.Sigma
            del graph.Epsilon
            del graph.AtomicNumber
            del graph.AtomicRadius
            del graph.Electronegativity
            del graph["Atomic Weight"]
            del graph["Electron Affinity"]
            del graph["Ionization Energy"]
            del graph["Oxidation State"]
            del graph["Coordination Number"]
            del graph["Atomic Volume"]
            del graph["Thermal Conductivity"]
            del graph["Specific Heat Capacity"]
            del graph["Polarizability"]
            # Step 1: Convert edge_index into a list of edges and filter out reverse edges
            edges = (
                graph.edge_index.t().tolist()
            )  # Transpose edge_index and convert to list
            unique_edges = []

            # Step 2: Create a set to keep track of edges we have already added
            seen_edges = set()

            for edge in edges:
                # Create an ordered edge tuple (min, max) to handle (a, b) and (b, a) as the same
                edge_tuple = tuple(sorted(edge))
                if edge_tuple not in seen_edges:
                    unique_edges.append(edge)
                    seen_edges.add(edge_tuple)  # Mark this edge as seen

            # Step 3: Update graph's edge_index with the unique edges (convert back to tensor)
            graph.edge_index = torch.tensor(unique_edges).t().contiguous()
            # Save the graph
            path = os.path.join(graph_folder, "PT", f"{mxene}.pt")
            torch.save(graph, path)

            # Node color mapping based on categories
            node_colors = []
            M_layer_elements = [
                "Cr",
                "Hf",
                "Mo",
                "Nb",
                "Sc",
                "Ta",
                "Ti",
                "V",
                "W",
                "Y",
                "Zr",
            ]
            X_layer_elements = ["C", "N"]
            T_layer_elements = ["Cl", "F", "H", "I", "O", "S", "Se", "Te"]

            # Assign colors based on the layer
            for atom in atom_properties["Atom"]:
                if atom[0] in M_layer_elements:
                    node_colors.append("red")
                elif atom[0] in X_layer_elements:
                    node_colors.append("green")
                elif atom[0] in T_layer_elements:
                    node_colors.append("blue")
                else:
                    node_colors.append("yellow")

            # Plot the 2D graph
            plt.figure(figsize=(10, 10))
            pos_2d = nx.spring_layout(G, seed=42)
            nx.draw(
                G,
                pos_2d,
                with_labels=True,
                node_size=3000,
                node_color=node_colors,
                font_size=8,
                font_weight="bold",
                font_color="black",
            )
            edge_labels = nx.get_edge_attributes(G, "distance")
            rounded_edge_labels = {
                k: f"{v:.2f}" for k, v in edge_labels.items()
            }  # Round distances
            nx.draw_networkx_edge_labels(
                G, pos_2d, edge_labels=rounded_edge_labels, font_color="red"
            )
            plt.title(f"{mxene} Graph (2D)")
            plt.axis("off")
            path = os.path.join(graph_folder, "2D", f"{mxene}.png")
            plt.savefig(path)
            plt.close()

            # Plot the 3D graph
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            for u, v, d in G.edges(data=True):
                ax.plot(
                    [G.nodes[u]["X"], G.nodes[v]["X"]],
                    [G.nodes[u]["Y"], G.nodes[v]["Y"]],
                    [G.nodes[u]["Z"], G.nodes[v]["Z"]],
                    color="skyblue",
                )

            # Scatter nodes with colors and annotate atom names
            for node, color in zip(G.nodes, node_colors):
                x, y, z = G.nodes[node]["X"], G.nodes[node]["Y"], G.nodes[node]["Z"]
                ax.scatter(x, y, z, color=color, s=100)  # Scatter node
                ax.text(x, y, z, node, fontsize=8, fontweight="bold")

            ax.set_title(f"{mxene} 3D Graph")
            path = os.path.join(graph_folder, "3D", f"{mxene}.png")
            plt.savefig(path)
            plt.close()
            plt.close("all")  # Close all figures
            del G, pos_2d, edge_labels  # Free memory for graph-specific variables

            graph_count += 1
            plt.close("all")  # Close all figures

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(mxene_names)
        messagebox.showinfo(
            "Graph Creation",
            f"Graph creation process completed!\nTotal time: {total_time:.2f} seconds\nAverage time per graph: {avg_time:.2f} seconds",
        )

    # Run the graph creation in a separate thread
    thread = threading.Thread(target=run_graph_creation)
    thread.start()


def make_predictions():
    # Check if model and df are loaded
    if model is None:
        messagebox.showerror(
            "Error", "No model is loaded. Please load a trained model first."
        )
        return
    if df is None:
        messagebox.showerror(
            "Error", "No Excel file is loaded. Please load an Excel file first."
        )
        return

    # Check if Graphs/PT folder exists
    if not os.path.exists(graph_folder + "/PT"):
        messagebox.showerror("Error", "No graph files found in Graphs/PT folder.")
        return

    # Create Results folder in the current working directory
    results_folder = os.path.join(os.getcwd(), "Results")
    os.makedirs(results_folder, exist_ok=True)

    # Iterate through graph files and make predictions
    pt_files = [f for f in os.listdir(graph_folder + "/PT") if f.endswith(".pt")]
    if not pt_files:
        messagebox.showerror("Error", "No .pt files found in Graphs/PT folder.")
        return

    # Start timing and prepare result file names
    start_time = time.strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(
        results_folder, f"{start_time}_predictions_vs_targets.png"
    )
    text_filename = os.path.join(
        results_folder, f"{start_time}_predictions_vs_targets.txt"
    )

    messagebox.showinfo("Prediction", "Prediction process started!")

    # Initialize lists for plotting and saving predictions
    target_values = []
    prediction_values = []

    with open(text_filename, "w") as file:
        file.write("Graph Name\tTarget CO2\tTarget CH4\tPredicted CO2\tPredicted CH4\n")

        for pt_file in pt_files:
            # Load graph
            graph_path = os.path.join(graph_folder, "PT", pt_file)
            graph = torch.load(graph_path, weights_only=False)

            # Extract graph name
            graph_name = os.path.splitext(pt_file)[0]

            # Predict using the model
            model.eval()
            data_loader = loader.DataLoader([graph], batch_size=1)

            # CUDA
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            predictions = []
            with torch.no_grad():
                for data in data_loader:
                    data = data.to(device)
                    prediction = model(data)
                    predictions.append(prediction.cpu().numpy())

            # Convert predictions to numpy array
            prediction = predictions[0]

            # Get target values from DataFrame
            target = (
                df[df["Mxene"] == graph_name][["1Bar_CO2", "1Bar_CH4"]].iloc[0].values
            )
            co2_prediction = prediction[0][0]  # CO2
            ch4_prediction = prediction[0][1]  # CH4

            # Append to lists for plotting
            target_values.append(target)
            prediction_values.append([co2_prediction, ch4_prediction])

            # Write to text file
            file.write(
                f"{graph_name}\t{target[0]:.4f}\t{target[1]:.4f}\t{co2_prediction:.4f}\t{ch4_prediction:.4f}\n"
            )

    # Convert to numpy arrays for easier plotting
    target_values = np.array(target_values)
    prediction_values = np.array(prediction_values)

    # Plot all predictions vs targets
    plt.figure(figsize=(8, 8))
    plt.scatter(
        target_values[:, 0], prediction_values[:, 0], color="blue", label="CO2", s=50
    )
    plt.scatter(
        target_values[:, 1], prediction_values[:, 1], color="green", label="CH4", s=50
    )

    # Set axis limits
    max_value = max(target_values.max(), prediction_values.max()) * 1.1  # Padding
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)

    # Ideal y=x line
    plt.plot([0, max_value], [0, max_value], color="red", linestyle="--", label="Ideal")

    # Labels and grid
    plt.xlabel("Target Values")
    plt.ylabel("Predicted Values")
    plt.title("Predictions vs. Targets for All Graphs")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(plot_filename)
    plt.close()

    messagebox.showinfo(
        "Prediction",
        f"Prediction process completed!\nResults saved to {results_folder}.",
    )


# Create the main window
def create_gui():
    global model, df  # Access the global model and DataFrame variables

    # Create root window
    root = tk.Tk()
    root.title("MXene Adsorption Predictor")

    # Set background color and window size (taller window)
    root.configure(bg="#2c2f36")
    root.geometry("1000x1080")

    # Top bar (without close and minimize buttons)
    top_bar = tk.Frame(root, bg="#21252b", height=60)
    top_bar.pack(fill="x", side="top")

    # Title label (centered a little to the left)
    title_label = Label(
        top_bar,
        text="MXene Adsorption Predictor",
        font=("Arial", 20),
        fg="white",
        bg="#21252b",
    )
    title_label.pack(pady=15, side="left", padx=30)

    # Sidebar (wider for more space)
    sidebar = tk.Frame(root, bg="#2f353b", width=300)  # Increased the width to 300
    sidebar.pack(fill="y", side="left")

    # Sidebar sections
    sidebar_header = Label(
        sidebar, text="MXene Explorer", font=("Arial", 12), fg="white", bg="#2f353b"
    )
    sidebar_header.pack(pady=10)

    # Buttons for different pages
    home_button = Button(sidebar, text="Home", command=lambda: show_page("home"))
    home_button.pack(pady=10)

    two_d_button = Button(sidebar, text="2D Models", command=lambda: show_page("2d"))
    two_d_button.pack(pady=10)

    three_d_button = Button(sidebar, text="3D Models", command=lambda: show_page("3d"))
    three_d_button.pack(pady=10)

    results_button = Button(
        sidebar, text="Results", command=lambda: show_page("results")
    )
    results_button.pack(pady=10)

    # Main content area for Home Page (this is where the content of the home page will appear)
    content_frame = tk.Frame(root, bg="#2c2f36")
    content_frame.pack(fill="both", expand=True, side="right")

    def previous_page_2d():
        global current_page_2d
        if current_page_2d > 0:
            current_page_2d -= 1
            show_2d_models()  # Refresh the 2D models page

    def next_page_2d():
        global current_page_2d
        current_page_2d += 1
        show_2d_models()  # Refresh the 2D models page

    def previous_page_3d():
        global current_page_3d
        if current_page_3d > 0:
            current_page_3d -= 1
            show_3d_models()  # Refresh the 3D models page

    def next_page_3d():
        global current_page_3d
        current_page_3d += 1
        show_3d_models()  # Refresh the 3D models page

    def show_page(page_name):
        # Clear the content_frame first
        for widget in content_frame.winfo_children():
            widget.destroy()

        # Display the selected page
        if page_name == "home":
            show_home_page()
        elif page_name == "2d":
            show_2d_models()
        elif page_name == "3d":
            show_3d_models()
        elif page_name == "results":
            show_results()

    def show_home_page():
        # Home Section (with explanation text)
        home_label = Label(
            content_frame, text="Home", font=("Arial", 20), fg="white", bg="#2c2f36"
        )
        home_label.grid(row=0, column=0, pady=20)

        home_text = Label(
            content_frame,
            text="MXenes are a group of two-dimensional materials composed of transition metal carbides, nitrides, or carbonitrides. They exhibit high electrical conductivity, mechanical strength, and large surface areas, making them ideal for applications such as energy storage, water purification, and catalysis.\n\nIn the context of natural gas extraction, MXenes are used for CO2 and CH4 adsorption. Their large surface area and tunable surface chemistry allow them to effectively capture and store CO2 and methane gases. This enhances the efficiency of natural gas extraction while also addressing environmental concerns about greenhouse gas emissions.",
            font=("Arial", 12),
            fg="white",
            bg="#2c2f36",
            justify="center",
            wraplength=600,
        )
        home_text.grid(row=1, column=0, padx=20)

        # MXene Image (add under the explanation text)
        mxene_image = tk.PhotoImage(
            file="C7TA11379J.gif"
        )  # Ensure the image file path is correct
        image_label = Label(content_frame, image=mxene_image, bg="#2c2f36")
        image_label.image = (
            mxene_image  # Keep reference to the image to avoid garbage collection
        )
        image_label.grid(row=2, column=0, pady=20)

        # Buttons in Home Section (spaced evenly)
        button_frame = tk.Frame(content_frame, bg="#2c2f36")
        button_frame.grid(row=3, column=0, pady=20)

        select_model_button = Button(
            button_frame, text="Select Model", command=load_model, font=("Arial", 12)
        )
        select_model_button.grid(row=0, column=0, pady=10)

        select_file_button = Button(
            button_frame,
            text="Select Excel File",
            command=open_file,
            font=("Arial", 12),
        )
        select_file_button.grid(row=1, column=0, pady=10)

        run_graph_button = Button(
            button_frame,
            text="Run Graph Creation",
            command=start_graph_creation,
            font=("Arial", 12),
        )
        run_graph_button.grid(row=2, column=0, pady=10)

        make_prediction_button = Button(
            button_frame,
            text="Make Prediction",
            command=make_predictions,
            font=("Arial", 12),
        )
        make_prediction_button.grid(row=3, column=0, pady=10)

        # Center the column in content frame
        content_frame.grid_columnconfigure(0, weight=1)

    def show_2d_models():
        global current_page_2d  # Access the global variable
        # Clear the content frame
        for widget in content_frame.winfo_children():
            widget.destroy()

        # Create a scrollable frame
        scrollable_frame = create_scrollable_frame(content_frame)

        # Title for 2D Models Page (centered)
        label = Label(
            scrollable_frame,
            text="2D Models",
            font=("Arial", 20),
            fg="white",
            bg="#2c2f36",
        )
        label.grid(row=0, column=0, pady=20, columnspan=2, sticky="n")

        # Get all PNG files in the Graphs/2D folder
        folder_path = "Graphs/2D"
        files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

        if not files:
            no_files_label = Label(
                scrollable_frame,
                text="No 2D graphs found.",
                font=("Arial", 14),
                fg="white",
                bg="#2c2f36",
            )
            no_files_label.grid(row=1, column=0, pady=10, columnspan=2)
            return

        # Calculate the range of files to display for the current page
        start_index = current_page_2d * per_page
        end_index = min(start_index + per_page, len(files))

        # Display each file with its name
        for idx, file in enumerate(files[start_index:end_index]):
            # MXene name from the file name (without extension)
            mxene_name = os.path.splitext(file)[0]
            mxene_label = Label(
                scrollable_frame,
                text=mxene_name,
                font=("Arial", 12),
                fg="white",
                bg="#2c2f36",
            )
            mxene_label.grid(
                row=idx * 2 + 1, column=0, pady=5, columnspan=2, sticky="n"
            )

            # Display the image
            image_path = os.path.join(folder_path, file)
            img = Image.open(image_path)

            # Resize the image to max width while maintaining aspect ratio
            max_width = 800  # Set the desired max width
            aspect_ratio = img.height / img.width
            new_width = min(max_width, img.width)
            new_height = int(new_width * aspect_ratio)
            img = img.resize((new_width, new_height))

            img = ImageTk.PhotoImage(img)

            img_label = Label(scrollable_frame, image=img, bg="#2c2f36")
            img_label.image = img  # Keep reference to avoid garbage collection
            img_label.grid(row=idx * 2 + 2, column=0, pady=10, columnspan=2, sticky="n")

        # Add pagination buttons
        pagination_frame = Frame(scrollable_frame, bg="#2c2f36")
        pagination_frame.grid(row=end_index * 2 + 1, column=0, pady=20, columnspan=2)

        # Previous button
        prev_button = Button(
            pagination_frame,
            text="Previous",
            font=("Arial", 12),
            fg="white",
            bg="#4CAF50",
            command=previous_page_2d,
            state="normal" if current_page_2d > 0 else "disabled",
        )
        prev_button.grid(row=0, column=0, padx=10)

        # Next button
        next_button = Button(
            pagination_frame,
            text="Next",
            font=("Arial", 12),
            fg="white",
            bg="#4CAF50",
            command=next_page_2d,
            state=(
                "normal" if current_page_2d < (len(files) // per_page) else "disabled"
            ),
        )
        next_button.grid(row=0, column=1, padx=10)

    def show_3d_models():
        global current_page_3d  # Access the global variable
        # Clear the content frame
        for widget in content_frame.winfo_children():
            widget.destroy()

        # Create a scrollable frame
        scrollable_frame = create_scrollable_frame(content_frame)

        # Title for 3D Models Page (centered)
        label = Label(
            scrollable_frame,
            text="3D Models",
            font=("Arial", 20),
            fg="white",
            bg="#2c2f36",
        )
        label.grid(row=0, column=0, pady=20, columnspan=2, sticky="n")

        # Get all PNG files in the Graphs/3D folder
        folder_path = "Graphs/3D"
        files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

        if not files:
            no_files_label = Label(
                scrollable_frame,
                text="No 3D graphs found.",
                font=("Arial", 14),
                fg="white",
                bg="#2c2f36",
            )
            no_files_label.grid(row=1, column=0, pady=10, columnspan=2)
            return

        # Calculate the range of files to display for the current page
        start_index = current_page_3d * per_page
        end_index = min(start_index + per_page, len(files))

        # Display each file with its name
        for idx, file in enumerate(files[start_index:end_index]):
            # MXene name from the file name (without extension)
            mxene_name = os.path.splitext(file)[0]
            mxene_label = Label(
                scrollable_frame,
                text=mxene_name,
                font=("Arial", 12),
                fg="white",
                bg="#2c2f36",
            )
            mxene_label.grid(
                row=idx * 2 + 1, column=0, pady=5, columnspan=2, sticky="n"
            )

            # Display the image
            image_path = os.path.join(folder_path, file)
            img = Image.open(image_path)

            # Resize the image to max width while maintaining aspect ratio
            max_width = 800  # Set the desired max width
            aspect_ratio = img.height / img.width
            new_width = min(max_width, img.width)
            new_height = int(new_width * aspect_ratio)
            img = img.resize((new_width, new_height))

            img = ImageTk.PhotoImage(img)

            img_label = Label(scrollable_frame, image=img, bg="#2c2f36")
            img_label.image = img  # Keep reference to avoid garbage collection
            img_label.grid(row=idx * 2 + 2, column=0, pady=10, columnspan=2, sticky="n")

        # Add pagination buttons
        pagination_frame = Frame(scrollable_frame, bg="#2c2f36")
        pagination_frame.grid(row=end_index * 2 + 1, column=0, pady=20, columnspan=2)

        # Previous button
        prev_button = Button(
            pagination_frame,
            text="Previous",
            font=("Arial", 12),
            fg="white",
            bg="#4CAF50",
            command=previous_page_3d,
            state="normal" if current_page_3d > 0 else "disabled",
        )
        prev_button.grid(row=0, column=0, padx=10)

        # Next button
        next_button = Button(
            pagination_frame,
            text="Next",
            font=("Arial", 12),
            fg="white",
            bg="#4CAF50",
            command=next_page_3d,
            state=(
                "normal" if current_page_3d < (len(files) // per_page) else "disabled"
            ),
        )
        next_button.grid(row=0, column=1, padx=10)

    def show_results():
        # Clear previous content on the results page (if any)
        for widget in content_frame.winfo_children():
            widget.destroy()

        # Results Page Title
        label = Label(
            content_frame, text="Results", font=("Arial", 20), fg="white", bg="#2c2f36"
        )
        label.grid(row=0, column=0, pady=20)

        # Add introductory text
        text = Label(
            content_frame,
            text="Here you can view the latest prediction results.",
            font=("Arial", 12),
            fg="white",
            bg="#2c2f36",
        )
        text.grid(row=1, column=0, pady=20)

        # Find the latest results file based on timestamp in the Results folder
        results_folder = os.path.join(os.getcwd(), "Results")
        if not os.path.exists(results_folder):
            messagebox.showerror("Error", "Results folder does not exist.")
            return

        # Get the latest results based on the file modification time
        files = [f for f in os.listdir(results_folder) if f.endswith(".png")]
        if not files:
            messagebox.showerror("Error", "No results found.")
            return

        latest_file = max(
            files, key=lambda f: os.path.getmtime(os.path.join(results_folder, f))
        )

        # Extract the base name of the latest result (using timestamp)
        latest_plot_path = os.path.join(results_folder, latest_file)
        latest_text_file = latest_file.replace(".png", ".txt")
        latest_text_path = os.path.join(results_folder, latest_text_file)

        # Display the latest plot (image) in the results page
        img = Image.open(latest_plot_path)
        img = img.resize((400, 400))  # Resize image to fit the window
        img_tk = ImageTk.PhotoImage(img)

        label_img = Label(content_frame, image=img_tk)
        label_img.image = img_tk  # Keep a reference to avoid garbage collection
        label_img.grid(row=2, column=0, pady=20)

        # Display the content of the corresponding text file in a Text widget
        with open(latest_text_path, "r") as file:
            text_content = file.read()

        text_widget = Text(content_frame, wrap="word", width=80, height=20)
        text_widget.insert("1.0", text_content)
        text_widget.config(state="disabled")  # Make the text widget read-only
        text_widget.grid(row=3, column=0, pady=20)

        # Optionally, you can add a "Refresh" button to reload the latest results
        refresh_button = Button(content_frame, text="Refresh", command=show_results)
        refresh_button.grid(row=4, column=0, pady=20)

    # Show Home Page by default on startup
    show_page("home")

    root.mainloop()


if __name__ == "__main__":
    create_folders()  # Create folders
    create_gui()
