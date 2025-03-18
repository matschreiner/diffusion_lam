from dlam import utils
from dlam.model import interaction_net


def test_instantiate_interaction_net():
    edge_index = utils.load("test/resources/edge_index.pkl")
    batch = utils.load("test/resources/batch.pkl")

    model = interaction_net.InteractionNet(edge_index, 5)
    out = model.forward(batch)
    __import__("pdb").set_trace()  # TODO delme
