from __future__ import annotations
from functools import partial

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import evaluate
from fire import Fire

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def preprocess_function(examples, tokenizer, text_col="text"):
    return tokenizer(examples[text_col], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    results = {
        **accuracy.compute(predictions=predictions, references=labels),
        **{
            f"f1_{average}": f1.compute(predictions=predictions, references=labels, average=average)["f1"]
            for average in ["macro", "micro", "weighted"]
        }
    }
    return results


def main(
    *,
    dataset_checkpoint: str,
    model_checkpoint: str,
    text_col: str = "text",
    label_col: str = "label",
    train_subset: str = "train",
    val_subset: str | None = None,
    test_subset: str | None = None,
    lr=2e-5,
    train_batch_size=16,
    eval_batch_size=16,
    n_epochs=2,
    report_to=None,
    overfit_batch=False,
    metric_for_best_model="f1_micro"
):
    params = locals()
    dataset = load_dataset(dataset_checkpoint)
    # print((val_subset is None or label_col in dataset[val_subset]))
    # print(dataset[train_subset].features)
    # print(
    #     train_subset in dataset,
    #     (val_subset is None or val_subset in dataset),
    #     (test_subset is None or test_subset in dataset),
    #     (label_col in dataset[train_subset].features),
    #     (val_subset is None or label_col in dataset[val_subset].features)
    # )
    assert (
        train_subset in dataset and 
        (val_subset is None or val_subset in dataset) and 
        (test_subset is None or test_subset in dataset) and
        (label_col in dataset[train_subset].features) and
        (val_subset is None or label_col in dataset[val_subset].features)
    )
    num_labels = len(np.unique(dataset[train_subset][label_col]))
    if val_subset is None:
        val_subset = "val"
        splitted_train = dataset[train_subset].train_test_split(test_size=0.2, shuffle=True, seed=42)
        dataset[val_subset] = splitted_train['test']
        dataset[train_subset] = splitted_train['train']

    if overfit_batch:
        dataset[train_subset] = dataset[train_subset].select(list(range(train_batch_size)))
        dataset[val_subset] = dataset[val_subset].select(list(range(eval_batch_size)))
        if test_subset is not None:
            dataset[test_subset] = dataset[test_subset].select(list(range(eval_batch_size)))

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_dataset = dataset.map(partial(preprocess_function, tokenizer=tokenizer, text_col=text_col), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )
    run_name = f"{model_checkpoint}_{dataset_checkpoint}".replace("/", "-")
    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=f"./logs/{run_name}",
        learning_rate=lr,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=n_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=report_to,
        metric_for_best_model=metric_for_best_model
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset[train_subset],
        eval_dataset=tokenized_dataset[val_subset],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    if test_subset is not None:
        test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset[test_subset], metric_key_prefix="test")
        print("Test metrics: ", test_metrics)
        if report_to == "wandb":
            import wandb
            for k, v in params.items():
                wandb.config[k] = v
            wandb.log(test_metrics)


if __name__ == "__main__":
    Fire(main)
