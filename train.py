from pydantic_settings import (
    CliApp,
)  # We use pydantic for the CLI instead of argparse so that our arguments are
from pydantic import BaseModel
from oocr_influence.data import get_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PretrainedConfig,
)
from typing import cast
import sys
import torch
from oocr_influence.train import train


class TrainingArgs(BaseModel):
    data_dir: str

    batch_size: int = 512
    epochs: int = 10

    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warm_up_steps: int = 2000

    model_name: str | None = None
    num_proc_dataset_creation: int = 4

    num_entities: int = 2000
    num_relations: int = 200
    relations_per_entity: int = 20
    phi: float = 17.5
    proportion_ood_facts: float = 0.05
    proportion_iid_test_set_facts: float = 0.005


def main(args: TrainingArgs):

    aspirational_config = {
        "reprocess_input_data": True,
        "overwrite_output_dir": args.overwrite_output_dir,
        "max_seq_length": args.max_seq_length,
        "max_length": args.max_length,
        "max_gen_length": args.max_gen_length,
        "block_size": args.block_size,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "save_eval_checkpoints": False,
        "save_steps": args.save_step,
        "use_multiprocessing": False,
        "output_dir": output_dir,
        "manual_seed": args.manual_seed,
        "fp16": args.fp16,
        "truncation": True,
        "dataloader_num_workers":args.dataloader_num_workers,
        "use_multiprocessed_decoding":args.use_multiprocessed_decoding,
        "save_best_model": args.save_best_model,
        "save_model_every_epoch": args.save_model_every_epoch,
        "save_epoch_interval": args.save_epoch_interval,
        "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "evaluate_during_training": args.evaluate_during_training,
        "predict_during_training": args.predict_during_training,
        "mlm": False,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "n_layer": args.n_layer,
        "n_inner": args.n_inner,
        "n_head": args.n_head,
        "memory_dim": args.memory_dim,
    } # Lets try and get as many of these as we can
    
    # definition of gp2 config
    #( vocab_size = 50257n_positions = 1024n_embd = 768n_layer = 12n_head = 12n_inner = Noneactivation_function = 'gelu_new'resid_pdrop = 0.1embd_pdrop = 0.1attn_pdrop = 0.1layer_norm_epsilon = 1e-05initializer_range = 0.02summary_type = 'cls_index'summary_use_proj = Truesummary_activation = Nonesummary_proj_to_labels = Truesummary_first_dropout = 0.1scale_attn_weights = Trueuse_cache = Truebos_token_id = 50256eos_token_id = 50256scale_attn_by_inverse_layer_idx = Falsereorder_and_upcast_attn = False**kwargs )
    config = GPT2Config(
        
    

    if args.model_name is None:
        config = GPT2Config()
        model = GPT2LMHeadModel(config=config)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore
    if args.model_name is not None:
        
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model, tokenizer, config = (
        cast(GPT2LMHeadModel, model),
        cast(PreTrainedTokenizer, tokenizer),
        cast(PretrainedConfig, config),
    )  # transformers library isn't fully typed, so we cast to the correct types. Gpt2LMHeadModel can fit in for a wide variety of transformer models

    train_dataset, test_dataset = get_dataset(
        tokenizer=tokenizer,
        num_proc=args.num_proc_dataset_creation,
        num_entities=args.num_entities,
        num_relations=args.num_relations,
        relations_per_entity=args.relations_per_entity,
        phi=args.phi,
        proportion_ood_facts=args.proportion_ood_facts,
        proportion_iid_test_set_facts=args.proportion_iid_test_set_facts,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

    train(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            sys.argv[sys.argv.index(arg)] = arg.replace("_", "-")

    args = CliApp.run(
        TrainingArgs
    )  # Parse the arguments, returns a TrainingArgs object
    main(args)
