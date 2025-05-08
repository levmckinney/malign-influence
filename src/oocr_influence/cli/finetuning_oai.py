from openai import AsyncOpenAI, APIConnectionError
from shared_ml.utils import CliPydanticModel
from pathlib import Path
import asyncio
from pydantic_settings import CliApp
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)
import jsonlines
import json
from shared_ml.logging import setup_custom_logging, log
import logging
from datasets import load_from_disk

class FineTuningArgs(CliPydanticModel):
    train_dataset: Path
    model_to_finetune: str = "gpt-4.1"
    n_epochs: int = 1
    output_dir: Path = Path("./outputs/")

    wandb_project: str = "malign-influence"
    logging_type: str = "wandb"

logger = logging.getLogger(__name__)

async def main(args: FineTuningArgs):

    experiment_name = f"oai_finetune_{args.model_to_finetune}_n_epochs_{args.n_epochs}_train_dataset_{args.train_dataset.name}"
    experiment_output_dir = (Path(args.output_dir) / experiment_name).absolute()
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Outputs saved at: {experiment_output_dir.absolute()}")

    setup_custom_logging(
        experiment_name=experiment_name,
        experiment_output_dir=experiment_output_dir,
        logging_type=args.logging_type,
        wandb_project=args.wandb_project,
    )
    log().state.args = args.model_dump()
    client = AsyncOpenAI()

    file_path = hf_train_dataset_to_oai_train_dataset(args.train_dataset, experiment_output_dir)

    file_obj = await client.files.create(
        file=file_path,
        purpose="fine-tune"
    )

    while True:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(APIConnectionError),
            stop=stop_after_attempt(3),
            wait=wait_fixed(30),
        ):
            with attempt:
                file_obj = await client.files.retrieve(file_obj.id)

        if file_obj.status == "processed":
            break

        logger.info(f"Wating for {file_obj.id} to be processed...")
        await asyncio.sleep(10)


    ft_job = await client.fine_tuning.jobs.create(
        training_file=file_obj.id,
        model=args.model_to_finetune,
        hyperparameters={
            "n_epochs": args.n_epochs
        }
    )

    log().add_to_log_dict(ft_job_id=ft_job.id)

    while True:
        ft_job = await client.fine_tuning.jobs.retrieve(ft_job.id)

        match ft_job.status:

            case "succeeded":
                logger.info("Fine-tuning job succeeded.")
                break
            case "failed" | "cancelled":
                logger.error(f"Fine-tuning job failed: {ft_job.status}")
                break

            case _:
                logger.info(f"Fine-tuning job is in state {ft_job.status}, expected_time_to_completion: {ft_job.estimated_finish / 60 if ft_job.estimated_finish else 'unknown'} minutes, waiting for 10 seconds...")
                await asyncio.sleep(10)
    
    logger.info(f"Fine-tuning job {ft_job.id} has finished")
    

def hf_train_dataset_to_oai_train_dataset(train_dataset_path: Path, experiment_output_dir: Path) -> Path:
    dataset = load_from_disk(train_dataset_path)

    train_docs = [p + c for p, c in zip(dataset["prompt"], dataset["completion"])] # type: ignore

    all_data = [
        {
            "messages": [
                {"role": "user", "content": "<DOCTAG>"},
                {"role": "assistant", "content": f"{doc}"},
            ]
        }
        for doc in train_docs
    ]

    jsonlies_data = ""
    for data in all_data:
        jsonlies_data += json.dumps(data) + "\n"
    
    dataset_path = experiment_output_dir / "train_dataset.jsonl"
    with open(dataset_path, "w") as f:
        f.write(jsonlies_data)
        
    return dataset_path.absolute()

if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)
    asyncio.run(main(CliApp.run(FineTuningArgs)))
