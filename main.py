from pathlib import Path
import tempfile
import arxiv
from converter import (
    download_paper,
    pdf_to_markdown,
    split_markdown_by_section,
    translate_to_latex,
    create_final_package,
)


def process(url: str, form: str, col_num: int = 1) -> tuple[Path, Path]:
    client = arxiv.Client(delay_seconds=3.0, num_retries=5)

    temp_dir = Path(tempfile.mkdtemp())
    download_dir = Path("downloads")
    download_dir.mkdir(exist_ok=True)
    outputs_dir = temp_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    arxiv_id = download_paper(url, client, download_dir)

    if arxiv_id:
        pdf_to_markdown(arxiv_id, download_dir, outputs_dir)

        if form == "Tex":
            split_markdown_by_section(outputs_dir)
            translate_to_latex(outputs_dir)
            create_final_package(outputs_dir, col_num)

            target_file_path = outputs_dir / "package" / arxiv_id / "main.zip"

            return target_file_path, temp_dir

        # form = Markdown
        else:
            target_file_path = outputs_dir / (arxiv_id + ".md")
            return target_file_path, temp_dir

    else:
        raise ValueError("arxiv_idが見つかりませんでした。")
