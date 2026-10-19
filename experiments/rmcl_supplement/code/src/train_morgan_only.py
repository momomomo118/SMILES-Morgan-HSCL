"""Entry point for the Morgan-only baseline."""

from train_single_modal import run


if __name__ == "__main__":
    run(default_mode="morgan")
