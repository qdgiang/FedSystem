import datetime
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from logger import FED_LOGGER

# test the logger
FED_LOGGER.log(INFO, "test info")

def test_func():
    FED_LOGGER.log(INFO, "test info")

test_func()

def main():
    print("main")
    FED_LOGGER.log(CRITICAL, f"test info {datetime.datetime.now().strftime('%y_%m_%d-%H_%M_%S')}")

if __name__ == "__main__":
    main()