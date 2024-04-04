import argparse
import os
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from accelerate import Accelerator
from trl import SFTTrainer
from datasets import Dataset
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="bigcode/starcoder2-3b")
    parser.add_argument("--subset", type=str, default="data/rust")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="prompt")  # Change to "prompt"

    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_starcoder2")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--push_to_hub", type=bool, default=True)
    return parser.parse_args()


def main(args):
    accelerator = Accelerator()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model = accelerator.prepare(model)

    # Load dataset from CSV
    prompts_df = pd.read_csv("prompts_modified.csv", sep='\t')  # Modified CSV file with "act" and "prompt" separated
    
    # Create a dictionary from act and prompt columns
    prompts_dict = dict(zip(prompts_df['act'], prompts_df['prompt']))

    # Create a dataset object with "prompt" as the column name
    dataset = Dataset.from_dict({"prompt": list(prompts_dict.values())})

    # Setup the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            logging_strategy="steps",
            logging_steps=10,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            seed=args.seed,
            report_to="wandb",
        ),
        dataset_text_field=args.dataset_text_field,
    )

    # Train the model
    trainer.train()

    # Save the last checkpoint of the model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
        if args.push_to_hub:
            trainer.push_to_hub("Upload model")
        print("Training Done! 💥")


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
