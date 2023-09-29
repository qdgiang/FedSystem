import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from flwr.common.typing import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)

from .shapley import ShapleyHelper
from common.logger import FED_LOGGER
from flwr.common.typing import GetParametersIns
from .client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from .history import MyHistory
from flwr.server.strategy import Strategy
from flwr.server.server import Server
from strategy import MyFedAvg
import itertools
from .aggregate import parameters_to_ndarrays, ndarrays_to_parameters, normal_aggregate, weighted_aggregate
from data.data_manager import DataManager
from model.model_manager import ModelManager

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]

class MyServer(Server):
    """Flower server."""

    def __init__(
        self, *, client_manager: ClientManager, strategy: Strategy, data_manager: DataManager, model_manager: ModelManager
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy = strategy #: Strategy = strategy if strategy is not None else MyFedAvg()
        self.max_workers: Optional[int] = None
        self.url2cid = {}
        self.cid2url = {}
        self.parameters_updates: Dict[int, Tuple[Parameters, int]] = {}
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.shapley_helper: ShapleyHelper # not initialized here because we don't know the number of clients yet

    def normal_aggr_from_cid_list(self, cid_list: Tuple[int]) -> Parameters:
        weights_and_no = [
            (
                parameters_to_ndarrays(self.parameters_updates.get(cid)[0]), 
                self.parameters_updates.get(cid)[1],
                self.shapley_helper.get_weights().get(cid)
            )
            for cid in cid_list
        ]

        aggregated_weights = ndarrays_to_parameters(normal_aggregate(weights_and_no))
        return aggregated_weights
    
    def weighted_aggr_from_cid_list(self, cid_list: Tuple[int]) -> Parameters:
        weights_and_no = [
            (
                parameters_to_ndarrays(self.parameters_updates.get(cid)[0]), 
                self.parameters_updates.get(cid)[1],
                self.shapley_helper.get_weights().get(cid)
            )
            for cid in cid_list
        ]

        aggregated_weights = ndarrays_to_parameters(weighted_aggregate(weights_and_no))
        return aggregated_weights

    def score_from_aggr(self) -> float:
        _, __, res = self.model_manager.evaluate_model(self.data_manager.get_test_data(),self.data_manager.get_test_label())
        return res.get("accuracy")
    
    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> MyHistory:
        """Run federated averaging for a number of rounds."""
        history = MyHistory()
        FED_LOGGER.info("FL process begins")
        # Initialize parameters
        FED_LOGGER.log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        #FED_LOGGER.debug("Initialized global parameters: %s", self.parameters)
        FED_LOGGER.log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            FED_LOGGER.log(
                INFO,
                "initial parameters (loss, other metrics): %s",
                res
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        FED_LOGGER.log(INFO, "FL training loop starts")
        start_time = timeit.default_timer()
        
        for current_round in range(1, num_rounds + 1):

            # 1. Train model in local clients and get updates
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            updates_from_clients = []
            if res_fit:
                parameters_prime, fit_metrics, updates_from_clients = res_fit #, _ = res_fit  # fit_metrics_aggregated
                #if parameters_prime:
                #    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )


            # 2. Save all parameter updates from clients
            for update in updates_from_clients[0]:
                if current_round == 1:
                    self.url2cid[update[0].cid] = update[1].metrics.get("client_id")
                    self.cid2url[update[1].metrics.get("client_id")] = update[0].cid
                    self.shapley_helper = ShapleyHelper(no_of_players=len(self.url2cid))
                one_cid = update[1].metrics.get("client_id")
                one_num_examples = update[1].num_examples
                self.parameters_updates[one_cid] = (update[1].parameters, one_num_examples)
            self.shapley_helper.prepare()

            # 3. Iterate through list of subsets and evaluate each subset
            for subset in self.shapley_helper.list_of_subsets:
                temp_aggr = self.normal_aggr_from_cid_list(subset)
                self.model_manager.set_params(parameters_to_ndarrays(temp_aggr))
                temp_score = self.score_from_aggr()
                self.shapley_helper.set_subset_score(subset=subset, score=temp_score)
                
            self.parameters = self.weighted_aggr_from_cid_list(
                    cid_list=range(0, len(self.url2cid), 1)
                )
            self.parameters_updates = {} # clear heavy parameter updates

            # 4. Calculate Shapley scores
            self.shapley_helper.shapley_calculation()
            self.shapley_helper.update_weights()
            self.shapley_helper.log()
            
            # 4.5 Get the score of weighted model
            self.model_manager.set_params(parameters_to_ndarrays(self.parameters))
            weighted_score = self.score_from_aggr()
            FED_LOGGER.info(f"Score of weighted model: {weighted_score}")
            FED_LOGGER.info("---------------------------------------------------")
            self.shapley_helper.clear_values()


            # 5. Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                FED_LOGGER.log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # 6. Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        FED_LOGGER.log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            FED_LOGGER.log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        FED_LOGGER.log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        FED_LOGGER.log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            FED_LOGGER.log(INFO, "Federated round %s: no clients selected, cancel", server_round)
            return None
        FED_LOGGER.log(
            DEBUG,
            "Federated round %s: %s sampled clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

        FED_LOGGER.log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        for i in range(len(results)):
            print(results[i][0].cid)
            print(results[i][1].metrics)

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            FED_LOGGER.log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        FED_LOGGER.log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        FED_LOGGER.log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)
