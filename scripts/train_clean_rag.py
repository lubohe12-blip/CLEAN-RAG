import argparse
from src_ext.utils.config import load_config
from src_ext.utils.device import get_device
from src_ext.utils.paths import ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    device = get_device(cfg)

    print("=" * 60)
    print("Start CLEAN+RAG training")
    print("Experiment:", cfg["experiment"]["name"])
    print("Device:", device)
    print("TopK:", cfg["retrieval"]["topk"])
    print("Batch size:", cfg["train"]["batch_size"])
    print("=" * 60)

    # TODO:
    # 1. load train/test data
    # 2. load embeddings
    # 3. build / load retrieval index
    # 4. retrieve top-k candidates
    # 5. fusion with CLEAN representation
    # 6. train and save checkpoints

    print("Template ready. Fill in training logic here.")


if __name__ == "__main__":
    main()