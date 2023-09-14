from typing import Dict
import itertools
from common.logger import FED_LOGGER

def get_all_subset(num_of_participants):
    all_subset = []
    for i in range(1, num_of_participants+1):
        all_subset.extend(list(itertools.combinations(range(num_of_participants), i)))
    return all_subset

def get_all_permutation(num_of_participants):
    all_permutation = []
    for i in range(num_of_participants):
        all_permutation.extend(list(itertools.permutations(range(num_of_participants), num_of_participants)))
    return all_permutation

def turn_tuple_to_set(all_subset):
    all_set = []
    for i in range(len(all_subset)):
        all_set.append(frozenset(all_subset[i]))
    return all_set


class ShapleyHelper:
    def __init__(self, no_of_players: int) -> None:
        self.no_of_players: int = no_of_players
        self.subset_scores: Dict[frozenset[int], float] = {} #{0}, {0,1}, {0,1,2}... -> 90,95,90,...
        self.permutation_individual_scores = {}    # Key: {0,1,2,3}, {0,1,3,2}, {0,2,1,3}.... Value {0: 90, 1: 10, 2: 0, 3: 0}
        self.list_of_subsets = []
        self.list_of_permutations = []
        self.client_scores = {}
        list_of_tuples = get_all_subset(self.no_of_players)
        self.list_of_subsets = turn_tuple_to_set(list_of_tuples)
        self.list_of_permutations = get_all_permutation(self.no_of_players)
    
    def prepare(self) -> None:
        for cid in range (0, self.no_of_players):
            self.client_scores[cid] = 0

    def set_subset_score(self, subset, score) -> None:
        self.subset_scores[subset] = score

    def shapley_calculation(self) -> None:
        FED_LOGGER.info(f"Shapley calculation started")
        #permutation_all_scores_raw = {}
        for permutation in self.list_of_permutations:
            dict_of_individual_scores_for_this_permutation = {}
            list_of_players = list(permutation)
            list_of_players_evaluated = []
            contribution_so_far = 0
            while list_of_players != []:
                next_player = list_of_players.pop(0)
                list_of_players_evaluated.append(next_player)
                required_score = self.subset_scores[frozenset(list_of_players_evaluated)]
                player_contribution = max(required_score - contribution_so_far, 0)
                dict_of_individual_scores_for_this_permutation[next_player] = player_contribution
                contribution_so_far += player_contribution
            #permutation_all_scores_raw[permutation] = dict_of_individual_scores_for_this_permutation
            self.permutation_individual_scores[permutation] = dict_of_individual_scores_for_this_permutation
        
        #for player_score_dict in permutation_all_scores_raw.values():
        for player_score_dict in self.permutation_individual_scores.values():
            for player, contribution in player_score_dict.items():
                self.client_scores[player] += contribution

        overall_score = 0
        for client in self.client_scores.keys():
            overall_score += self.client_scores[client]

        for client in self.client_scores.keys():
            self.client_scores[client] /= overall_score
        
        FED_LOGGER.info(f"Shapley values: {self.client_scores}")

    def log(self) -> None:
        FED_LOGGER.info(f"Subset scores: {self.subset_scores}")
        FED_LOGGER.info(f"Permutation individual scores: {self.permutation_individual_scores}")
        FED_LOGGER.info(f"Client scores: {self.client_scores}")

    def clear_values(self) -> None:
        self.subset_scores = {}
        self.permutation_individual_scores = {}
        self.client_scores = {}