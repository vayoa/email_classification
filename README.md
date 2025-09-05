# MTRX ML Engineer

## Quick Start

This project is manged with the `uv` package manager. To get started, run the following command:

```powershell
uv sync
```

## NOTICE

This project was tested only on my local machine, where I trained the model using my rtx 3070 mobile. RAG Inference in realtime is also done locally.

The uv project is configured to match my setup, so some configuration could be needed on your part.

## Timings

- Training (3 epocs): ~20m.
- Indexing (2000 emails): ~45s.
- Realtime RAG: ~<1s.

## Dataset

I used [this public dataset](https://huggingface.co/datasets/jason23322/high-accuracy-email-classifier).

## Model

The checkpoint can be installed from the [huggingface repo of the project](https://huggingface.co/vayoas/modernbert-email-multitask).

## Running This Project

### Training

The [data_prep](data_prep.ipynb) notebook produces [our prepped csv dataset](email_dataset.csv). This dataset is split for train/val/test automatically.

To start the model training, run the training script like so:

```powershell
uv run train_email_multitask.py  --csv ./email_dataset.csv  --text-col text  --max-length 512  --batch-size 16  --epochs 3
```

### Evaluation

The [evaluate](evaluate.py) script creates our [metrics json](eval_report/metrics.json) and some charts in [the eval report directory](eval_report).

To execute it, run:

```powershell
uv run evaluate.py --true outputs/modernbert_multitask/splits/test.csv --pred ./test_split_predictions_multitask.csv --outdir ./eval_report
```

### Streamlit App (index + inference + chatbot)

The streamlit app uses a couple of scripts:

1. [the indexer script](indexer.py) uses [our inference script](infer_email_multitask.py) to batch classify + encode (using minilm) as the gmail batches come in (there's a 100 email limit each batch so we exploit it).
   Both the inference script and the indexer script could run standalone, like so:

   ```powershell
   # Inference
   uv run infer_email_multitask.py  --model-dir outputs/modernbert_multitask  --csv outputs/modernbert_multitask/splits/test.csv --text-col text  --out-csv .\pred_test.csv # This predicts test results.

   # Indexer
   uv run .\indexer.py # this creates our vector store + metadata under ./db.
   ```

   > **Notice:** indexing requires a `credentials.json` file (comes from the google cloud api section) to read gmail emails.

2. [The streamlit script](app.py) is what you need to run for the whole package. It connects to the vector store under `./db` and utilizes it in query tools we give to gemini for chatting. You can also pase emails here directly to get added to the db or re-index.

   Before you can run this script, you need to make sure you have a .env in the same format as [the example .env](.env.example). You can get your gemini api key from [google ai studio](https://aistudio.google.com/apikey).

   To execute the streamlit app through `uv`, run:

   ```powershell
   uv run streamlit run .\app.py
   ```
