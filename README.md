
Метрики доступны: https://wandb.ai/levmorozov900/sentiment-analysis/overview

[Ноутбук с предобработкой данных и вузализацией](./notebooks/preprocess.ipynb)

Команды для обучения BERT моделей на трех наборах данных:
```
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python sequence_classification.py --dataset_checkpoint ./data/rureviews --model_checkpoint FacebookAI/xlm-roberta-base --report_to wandb --text_col review --label_col sentiment --test_subset test --n_epochs 3 --lr 5e-5 --train_batch_size 8
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python sequence_classification.py --dataset_checkpoint ./data/rureviews --model_checkpoint FacebookAI/xlm-roberta-large --report_to wandb --text_col review --label_col sentiment --test_subset test --n_epochs 3 --lr 5e-5 --train_batch_size 8
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python sequence_classification.py --dataset_checkpoint ./data/rureviews --model_checkpoint ai-forever/ruRoberta-large --report_to wandb --text_col review --label_col sentiment --test_subset test --n_epochs 3 --lr 5e-5 --train_batch_size 8
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python sequence_classification.py --dataset_checkpoint ./data/rureviews --model_checkpoint ai-forever/ruBert-base --report_to wandb --text_col review  --label_col sentiment --test_subset test --n_epochs 3 --lr 5e-5 --train_batch_size 8

WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python sequence_classification.py --dataset_checkpoint ./data/runews --model_checkpoint FacebookAI/xlm-roberta-base --report_to wandb --text_col text --label_col sentiment --test_subset test --n_epochs 3 --lr 5e-5 --train_batch_size 4 --eval_batch_size 8
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python sequence_classification.py --dataset_checkpoint ./data/runews --model_checkpoint FacebookAI/xlm-roberta-large --report_to wandb --text_col text --label_col sentiment --test_subset test --n_epochs 3 --lr 5e-5 --train_batch_size 1 --eval_batch_size 2
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python sequence_classification.py --dataset_checkpoint ./data/runews --model_checkpoint ai-forever/ruRoberta-large --report_to wandb --text_col text --label_col sentiment --test_subset test --n_epochs 3 --lr 5e-5 --train_batch_size 1 --eval_batch_size 2
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python sequence_classification.py --dataset_checkpoint ./data/runews --model_checkpoint ai-forever/ruBert-base --report_to wandb --text_col text --label_col sentiment --test_subset test --n_epochs 3 --lr 5e-5 --train_batch_size 1 --eval_batch_size 2


WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python sequence_classification.py --dataset_checkpoint ./data/rutweets --model_checkpoint FacebookAI/xlm-roberta-base --report_to wandb --test_subset test --n_epochs 3 --lr 5e-5 --train_batch_size 4 --eval_batch_size 8
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python sequence_classification.py --dataset_checkpoint ./data/rutweets --model_checkpoint FacebookAI/xlm-roberta-large --report_to wandb --test_subset test --n_epochs 3 --lr 5e-5 --train_batch_size 4 --eval_batch_size 8
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python sequence_classification.py --dataset_checkpoint ./data/rutweets --model_checkpoint ai-forever/ruRoberta-large --report_to wandb --test_subset test --n_epochs 3 --lr 5e-5 --train_batch_size 4 --eval_batch_size 8
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python sequence_classification.py --dataset_checkpoint ./data/rutweets --model_checkpoint ai-forever/ruBert-base --report_to wandb --test_subset test --n_epochs 3 --lr 5e-5 --train_batch_size 4 --eval_batch_size 8
```

Команды для обучения классических моделей TF-Idf на трех наборах данных:
```
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python tfidf_sequence_classification.py --dataset_checkpoint ./data/rureviews --clf_head logreg --test_subset test --report_to wandb --text_col review --label_col sentiment --n_iter 100
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python tfidf_sequence_classification.py --dataset_checkpoint ./data/rureviews --clf_head gb --test_subset test --report_to wandb --text_col review --label_col sentiment --n_iter 30
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python tfidf_sequence_classification.py --dataset_checkpoint ./data/rureviews --clf_head rf --test_subset test --report_to wandb --text_col review --label_col sentiment --n_iter 30


WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python tfidf_sequence_classification.py --dataset_checkpoint ./data/runews --clf_head logreg --test_subset test --report_to wandb --text_col text --label_col sentiment --n_iter 100
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python tfidf_sequence_classification.py --dataset_checkpoint ./data/runews --clf_head gb --test_subset test --report_to wandb --text_col text --label_col sentiment --n_iter 30
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python tfidf_sequence_classification.py --dataset_checkpoint ./data/runews --clf_head rf --test_subset test --report_to wandb --text_col text --label_col sentiment --n_iter 30


WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python tfidf_sequence_classification.py --dataset_checkpoint ./data/rutweets --clf_head logreg --test_subset test --report_to wandb --n_iter 100
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python tfidf_sequence_classification.py --dataset_checkpoint ./data/rutweets --clf_head gb --test_subset test --report_to wandb --n_iter 30
WANDB_LOG_MODEL=false WANDB_PROJECT=sentiment-analysis python tfidf_sequence_classification.py --dataset_checkpoint ./data/rutweets --clf_head rf --test_subset test --report_to wandb --n_iter 30
```