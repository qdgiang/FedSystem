from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

class MyCriterion(Criterion):
    """Abstract class which allows subclasses to implement criterion
    sampling."""

    def select(self, client: ClientProxy) -> bool:
        """Decide whether a client should be eligible for sampling or not."""
        return True
