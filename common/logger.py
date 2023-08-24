import logging
from logging import LogRecord
from datetime import datetime

FED_LOGGER = logging.getLogger("FED")
FED_LOGGER.setLevel(logging.DEBUG) 

DEFAULT_FORMATTER = logging.Formatter(
    fmt="%(levelname)s %(asctime)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"

)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(DEFAULT_FORMATTER)
FED_LOGGER.addHandler(console_handler)

current_time = datetime.now().strftime("%y_%m_%d-%H_%M_%S")
file_handler = logging.FileHandler(f"fed-20{current_time}.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(DEFAULT_FORMATTER)
FED_LOGGER.addHandler(file_handler)


