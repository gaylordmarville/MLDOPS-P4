from src.train_model import task
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", help="train or infer",
                        action="store_true")
    args = parser.parse_args()

    mode = args.training

    if mode:
        logging.info("Training mode starting...")
    else:
        logging.info("Inference mode starting...")

    task(args.training)
