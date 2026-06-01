from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.data.manifest import load_manifest


def main():
    config = load_config()

    experiment_name = config["experiment"]["name"]
    logger = setup_logger(experiment_name)

    logger.info("Checking manifest.")
    logger.info(f"Config path: {config['_config_path']}")

    manifest_path = config["data"]["manifest_path"]
    df = load_manifest(manifest_path)

    logger.info(f"Loaded manifest: {manifest_path}")
    logger.info(f"Number of rows: {len(df)}")
    logger.info(f"Columns: {list(df.columns)}")

    logger.info("Split distribution:")
    logger.info(f"\n{df['split'].value_counts()}")

    logger.info("Accent distribution:")
    logger.info(f"\n{df['accent_label'].value_counts()}")

    logger.info("Dataset distribution:")
    logger.info(f"\n{df['dataset_name'].value_counts()}")

    logger.info("Manifest check finished.")


if __name__ == "__main__":
    main()