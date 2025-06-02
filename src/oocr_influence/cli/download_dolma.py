from pathlib import Path

from datasets import Dataset, IterableDataset, load_dataset
from pydantic import field_serializer
from pydantic_settings import CliApp

from oocr_influence.datasets.continual_pretraining import PRETRAIN_DATASET_SCHEMA
from shared_ml.utils import CliPydanticModel


class DownloadOlmoArgs(CliPydanticModel):
    num_examples: int
    dataset_name: str = "mlfoundations/dclm-baseline-1.0"
    output_dir: Path = Path("./datasets")
    split: str = "train"
    seed: int = 42

    @field_serializer("output_dir")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None


def main(args: DownloadOlmoArgs):
    dataset_streaming: IterableDataset = load_dataset(args.dataset_name, split=args.split, streaming=True)  # type: ignore

    dataset_streaming = dataset_streaming.shuffle(seed=args.seed)

    def generator():  # type: ignore
        for i, example in enumerate(dataset_streaming):
            if i >= args.num_examples:
                break
            yield example

    dataset: Dataset = Dataset.from_generator(generator, features=dataset_streaming.features)  # type: ignore
    dataset = dataset.map(lambda x: {"prompt": "", "completion": x["text"]})
    dataset = dataset.add_column("type", ["pretraining_document"] * len(dataset))  # type: ignore
    dataset = dataset.select_columns(PRETRAIN_DATASET_SCHEMA.keys())  # type: ignore
    dataset = dataset.cast(PRETRAIN_DATASET_SCHEMA)  # type: ignore

    dataset_name = f"{args.dataset_name.replace('/', '_')}_num_examples_{args.num_examples}_{dataset._fingerprint}"  # type: ignore
    dataset.save_to_disk(args.output_dir / dataset_name)

    print(f"Dataset saved to {args.output_dir / dataset_name}")


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)

    args = CliApp.run(DownloadOlmoArgs)

    main(args)
