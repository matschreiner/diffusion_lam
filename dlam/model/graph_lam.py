# Standard library
import os
import pickle as pkl

# Third-party
import pytorch_lightning as pl
import torch
import torch_geometric as pyg

# First-party
from neural_lam import create_graph, utils
from neural_lam.datastore.mdp import MDPDatastore
from neural_lam.interaction_net import InteractionNet
from neural_lam.weather_dataset import WeatherDataset

from dlam.utils import Timer


class Graph(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        for key in graph:
            if len(graph[key]) != 0:
                self.register_buffer(key, graph[key])
                graph[key] = getattr(self, key)

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Graph does not contain key: {key}")


class GraphLAM(pl.LightningModule):
    def __init__(
        self,
        domain,
        in_dim=18,
        target_dim=2,
        mesh_dim=2,
        hidden_dim=64,
        edge_dim=3,
        processor_layers=3,
        mp_layers=2,
    ):
        super().__init__()

        graph = self.create_graph(domain)
        self.graph = Graph(graph)

        self.mesh_embedder = MLP(mesh_dim, hidden_dim)
        self.grid_embedder = MLP(in_dim, hidden_dim)
        self.grid_encoder = MLP(hidden_dim, hidden_dim)

        self.m2m_embedder = MLP(edge_dim, hidden_dim)
        self.g2m_embedder = MLP(edge_dim, hidden_dim)
        self.m2g_embedder = MLP(edge_dim, hidden_dim)
        self.readout = MLP(hidden_dim, target_dim, hidden_layers=4)

        self.m2g_gnn = InteractionNet(
            self.graph["m2g_edge_index"],
            hidden_dim,
            hidden_layers=mp_layers,
            update_edges=False,
        )

        self.g2m_gnn = InteractionNet(
            self.graph["g2m_edge_index"],
            hidden_dim,
            hidden_layers=mp_layers,
            update_edges=False,
        )
        processor_nets = [
            InteractionNet(
                self.graph["m2m_edge_index"],
                input_dim=hidden_dim,
                hidden_layers=mp_layers,
            )
            for _ in range(processor_layers)
        ]

        self.processor = pyg.nn.Sequential(
            "mesh_rep, edge_rep",
            [
                (net, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")
                for net in processor_nets
            ],
        )

    def forward(self, batch):
        if not hasattr(self, "graph"):
            self.graph = self.create_graph(batch["xy"])

        static = batch["static"]
        cond1 = batch["cond"][:, 0]
        cond2 = batch["cond"][:, 1]
        forcing = batch["forcing"][:, 1]
        batch_size = len(static)

        grid_features = torch.cat((cond1, cond2, forcing, static), dim=-1)

        grid_emb = self.grid_embedder(grid_features)
        mesh_emb = self.mesh_embedder(self.graph["mesh_static_features"])

        g2m_edge_emb = self.g2m_embedder(self.graph["g2m_features"])
        m2g_edge_emb = self.m2g_embedder(self.graph["m2g_features"])
        m2m_edge_emb = self.m2m_embedder(self.graph["m2m_features"])

        mesh_emb = expand_to_batch(mesh_emb, batch_size)
        g2m_edge_emb = expand_to_batch(g2m_edge_emb, batch_size)
        m2g_edge_emb = expand_to_batch(m2g_edge_emb, batch_size)
        m2m_edge_emb = expand_to_batch(m2m_edge_emb, batch_size)

        mesh_rep = self.g2m_gnn(grid_emb, mesh_emb, g2m_edge_emb)
        mesh_rep, _ = self.processor(mesh_rep, m2m_edge_emb)

        grid_rep = self.grid_encoder(grid_emb)
        grid_rep = grid_rep + self.m2g_gnn(mesh_rep, grid_rep, m2g_edge_emb)

        output = self.readout(grid_rep)

        return output

    def training_step(self, batch, _):
        t = Timer()
        out = self.forward(batch)
        target = batch["target"][:, 0]

        loss = torch.nn.functional.mse_loss(out, target).mean()
        self.log("train_loss", loss, prog_bar=True, logger=True)
        t.end()
        return loss

    def configure_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def create_graph(xy):
        if isinstance(xy, torch.Tensor):
            xy = xy.numpy()
        create_graph.create_graph(
            "/tmp/graph",
            xy,
            n_max_levels=1,
            hierarchical=False,
            create_plot=False,
        )
        graph = utils.load_graph("/tmp/graph")[1]
        return graph


def expand_to_batch(x, batch_size):
    """
    Expand tensor with initial batch dimension
    """
    return x.unsqueeze(0).expand(batch_size, -1, -1)


class WeatherDataset2(WeatherDataset):
    def __init__(self, datastore, precision=torch.float32):
        datastore = MDPDatastore(datastore) if isinstance(datastore, str) else datastore
        super().__init__(datastore)
        self.datastore = datastore

        self.xy = torch.tensor(datastore.get_xy("state", stacked=False))
        self.static = torch.tensor(
            self.datastore.get_dataarray(
                category="static", split=None, standardize=True
            ).values
        )
        self.precision = precision
        self.boundary_mask = torch.tensor(self.datastore.boundary_mask.values).to(
            precision
        )
        self.interior_mask = 1 - self.boundary_mask

    def __getitem__(self, index):
        timer = Timer()
        cond_states, target_states, forcing, times = super().__getitem__(index)

        item = {}
        item["static"] = self.static.to(self.precision)
        item["xy"] = self.xy.to(self.precision)
        item["forcing"] = forcing.to(self.precision)
        item["cond"] = cond_states.to(self.precision)
        item["target"] = target_states.to(self.precision)
        item["time"] = times.to(self.precision)
        item["bounday_mask"] = self.boundary_mask
        item["interior_mask"] = self.interior_mask

        timer.end()
        return cond_states, target_states, forcing, times


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, hidden_layers=1):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = out_dim

        modules = [torch.nn.Linear(in_dim, hidden_dim, bias=True)]

        for i in range(hidden_layers):
            modules.append(torch.nn.SiLU())
            layer_in_dim = hidden_dim
            layer_out_dim = hidden_dim if i < hidden_layers - 1 else out_dim
            modules.append(torch.nn.Linear(layer_in_dim, layer_out_dim, bias=True))

        self.net = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def main():
    # Third-party
    from torch.utils.data import DataLoader

    #  graph_path = "tests/datastore_examples/mdp/danra_100m_winds/graph/1level"
    datastore = "../neural-lam/experiments/test/example.danra.yaml"
    dataset = WeatherDataset2(datastore)
    model = GraphLAM(domain=dataset.xy)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    batch = next(iter(dataloader))
    loss = model(batch)
    print(loss)


if __name__ == "__main__":
    main()
