# bot/trainer.py
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling


def fine_tune_gpt2(
    data_file:str,
    output_dir: str = "model/gpt2-finetuned",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
):
    """
    Fine-tune GPT-2 on the text at 'data_file' (one document, plain tex).
    Save model and tokenizer to 'output_dir'.
    """

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")


    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_file,
        block_size=512,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        learning_rate=learning_rate,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)