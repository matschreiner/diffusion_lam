import pytorch_lightning as pl
import torch
from neural_lam import create_graph, utils

from dlam.model.interaction_net import InteractionNet
from dlam.model.mlp import MLP


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
        #  domain,
        in_dim=18,
        target_dim=2,
        mesh_dim=2,
        hidden_dim=64,
        edge_dim=3,
        processor_layers=3,
        mp_layers=2,
    ):
        super().__init__()

        #  graph = self.create_graph(domain)
        #  self.graph = Graph(graph)

        self.mesh_embedder = MLP(mesh_dim, hidden_dim)
        self.grid_embedder = MLP(in_dim, hidden_dim)
        self.grid_encoder = MLP(hidden_dim, hidden_dim)

        self.m2m_embedder = MLP(edge_dim, hidden_dim)
        self.g2m_embedder = MLP(edge_dim, hidden_dim)
        self.m2g_embedder = MLP(edge_dim, hidden_dim)
        self.readout = MLP(hidden_dim, target_dim, hidden_layers=4)

        self.m2g_gnn = InteractionNet(
            input_dim=hidden_dim,
            hidden_layers=mp_layers,
            update_edges=False,
        )

        self.g2m_gnn = InteractionNet(
            input_dim=hidden_dim,
            hidden_layers=mp_layers,
            update_edges=False,
        )
        self.processor_nets = torch.nn.ModuleList(
            [
                InteractionNet(
                    input_dim=hidden_dim,
                    hidden_layers=mp_layers,
                )
                for _ in range(processor_layers)
            ]
        )

        self.save_hyperparameters()

    def forward(self, batch):
        static = batch["static"]
        cond1 = batch["cond"][:, 0]
        cond2 = batch["cond"][:, 1]
        forcing = batch["forcing"][:, 1]
        grid_features = torch.cat((cond1, cond2, forcing, static), dim=-1)

        return self._forward(grid_features, graph)

    def _forward(self, grid_features, graph):
        graph = self.use_shared_graph(graph)
        batch_size = len(grid_features)

        grid_emb = self.grid_embedder(grid_features)
        mesh_emb = self.mesh_embedder(graph["mesh_static_features"])

        g2m_edge_emb = self.g2m_embedder(graph["g2m_features"])
        m2g_edge_emb = self.m2g_embedder(graph["m2g_features"])
        m2m_edge_emb = self.m2m_embedder(graph["m2m_features"])

        mesh_emb = expand_to_batch(mesh_emb, batch_size)
        g2m_edge_emb = expand_to_batch(g2m_edge_emb, batch_size)
        m2g_edge_emb = expand_to_batch(m2g_edge_emb, batch_size)
        m2m_edge_emb = expand_to_batch(m2m_edge_emb, batch_size)

        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb, g2m_edge_emb, graph["g2m_edge_index"]
        )
        for processor in self.processor_nets:
            mesh_rep, m2m_edge_emb = processor(
                mesh_rep, mesh_rep, m2m_edge_emb, graph["m2m_edge_index"]
            )

        grid_rep = self.grid_encoder(grid_emb)
        grid_rep = grid_rep + self.m2g_gnn(
            mesh_rep, grid_rep, m2g_edge_emb, graph["m2g_edge_index"]
        )

        output = self.readout(grid_rep)

        return output

    def use_shared_graph(self, graph):
        graph = {key: graph[key][0] for key in graph if len(graph[key]) != 0}
        return graph

    def training_step(self, batch, _):
        out = self.forward(batch)
        loss = torch.nn.functional.mse_loss(out, batch.target).mean()

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


class GraphLAMNoise(GraphLAM):
    def forward(self, batch, t_diff):
        static = batch["static"]
        cond1 = batch["cond"][:, 0]
        cond2 = batch["cond"][:, 1]
        forcing = batch["forcing"][:, 1]

        grid_features = torch.cat((cond1, cond2, forcing, static), dim=-1)

        t_diff = t_diff[:, None, None].expand((-1, grid_features.shape[1], 1))

        grid_features = torch.cat([grid_features, batch.corr, t_diff], dim=-1)

        out = self._forward(grid_features, batch.graph)
        return out


def expand_to_batch(x, batch_size):
    """
    Expand tensor with initial batch dimension
    """
    return x.unsqueeze(0).expand(batch_size, -1, -1)
