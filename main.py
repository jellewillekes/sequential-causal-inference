import sys
import os
import logging

from data.raw.loader import request_raw_data
from data.process.data_cup import construct_cup_data
from data.process.data_league import construct_league_data
from data.financial.loader import request_financial_data
from data.distance.loader import request_distance_data
from data.process.preprocess import preprocess_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_request_raw_data(country):
    logging.info(f"Loading raw data for {country}...")
    request_raw_data(country)


def run_preprocess_data(country, cup):
    logging.info(f"Analyzing {cup} data...")
    construct_cup_data(country, cup)

    logging.info(f"Analyzing league data for {country}...")
    construct_league_data(country)

    financial_data_path = os.path.join('data', 'process', country, f'{cup}_financial_data.csv')
    if not os.path.exists(financial_data_path):
        logging.info("Loading financial data...")
        request_financial_data(country)

    distance_data_path = os.path.join('data', 'process', country, f'{cup}_distance_data.csv')
    if not os.path.exists(distance_data_path):
        logging.info(f"Calculating distances for {cup} in {country}...")
        request_distance_data(country, cup)

    logging.info("Preprocessing all data...")
    preprocess_data(country, cup)
    logging.info("Data processing is finished.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [options]")
        print("Commands:")
        print("  request_raw_data <country>")
        print("  preprocess_data <country> <cup>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "run_request_raw_data":
        if len(sys.argv) != 3:
            print("Usage: python main.py request_raw_data <country>")
            sys.exit(1)
        country = sys.argv[2]
        run_request_raw_data(country)
    elif command == "run_preprocess_data":
        if len(sys.argv) != 4:
            print("Usage: python main.py preprocess_data <country> <cup>")
            sys.exit(1)
        country = sys.argv[2]
        cup = sys.argv[3]
        run_preprocess_data(country, cup)
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py <command> [options]")
        print("Commands:")
        print("  request_raw_data <country>")
        print("  preprocess_data <country> <cup>")
        sys.exit(1)


if __name__ == "__main__":
    main()
