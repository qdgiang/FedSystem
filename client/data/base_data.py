from abc import ABC, abstractmethod

class BaseDataClass(ABC):
    def __init__(self, id):
        self.partition_id = id

    @abstractmethod
    def get_data(self, partition_id: int):
        pass

    