
from pydantic_settings import  CliApp  # We use pydantic for the CLI instead of argparse so that our arguments are 
from pydantic import BaseModel
from oocr_influence.data import generate_dataset

class TrainingArgs(BaseModel):
    data_dir : str
    
    batch_size: int = 512
    num_epochs: int = 10
    
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warm_up_steps: int = 2000

def train(args : TrainingArgs):
    
    
    data = load_data

    
    
if __name__ == "__main__":
    args = CliApp.run(TrainingArgs) # Parse the arguments, returns a TrainingArgs object
    train(args)