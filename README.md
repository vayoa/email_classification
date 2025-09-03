# MTRX ML Engineer

## Running this project

This project is manged with the `uv` package manager. The run it, use:

```powershell
uv sync
```

followed by:

```powershell
uv run <SCRIPT-TO-RUN>
# e.g. for a training run, execute - uv run train_email_multitask.py  --csv ./email_dataset.csv  --text-col text  --max-length 512  --batch-size 16  --epochs 3
```
