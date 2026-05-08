from fastapi import FastAPI, responses, BackgroundTasks
from pydantic import BaseModel
from main import process
from pathlib import Path
from typing import Any
import shutil

app = FastAPI()


class PaperRequest(BaseModel):
    url: str
    form: str
    col_num: int = 1


@app.get("/")
async def root() -> dict:
    return {"message": "Hello World"}


def remove_dir(path: Path) -> None:
    shutil.rmtree(path)


@app.post("/access")
async def convert_paper(
    paper_request: PaperRequest, background_tasks: BackgroundTasks
) -> Any:
    target_path, remove_path = process(
        paper_request.url, paper_request.form, paper_request.col_num
    )
    background_tasks.add_task(remove_dir, remove_path)
    return responses.FileResponse(target_path, filename=target_path.name)
