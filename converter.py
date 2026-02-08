from pathlib import Path
import pymupdf4llm
import arxiv
import re


def get_arxiv_id(file_url: str) -> str:
    m = re.search(r'\b[0-9]{4}[.][0-9]{5,}(v[0-9]+)?\b', file_url)
    if m:
        return m.group() 
    else: 
        return None


def download_paper(files: dict, client: object, download_dir: Path):
    for i, (filename, url) in enumerate(files.items()):
        # arxiv_idの取得
        arxiv_id = get_arxiv_id(url)

        if arxiv_id:
            # pdfファイルのダウンロード
            try:
                # ファイルの拡張子をpdfに変換
                filename = Path(filename).with_suffix(".pdf")

                search_by_id = arxiv.Search(id_list=[arxiv_id])
                result = next(client.results(search_by_id))
                result.download_pdf(dirpath=download_dir, filename=filename)
                print(f"{i + 1}件目のダウンロードが完了しました。")
            
            except FileNotFoundError as e:
                print(f"ファイルが見つかりませんでした。 {e}")

def pdf_to_markdown(download_dir: Path, outputs_dir: Path):

    for item in download_dir.iterdir():
        # 拡張子がpdfのファイルのみMarkdown化
        if item.suffix == ".pdf":
            try:
                md_text = pymupdf4llm.to_markdown(item)

                # 拡張子を除いたファイル名を取得
                filename = item.stem

                output_path = outputs_dir / (filename + ".md")
                output_path.write_bytes(md_text.encode())

                print(f"{filename}をmarkdown化しました")
            
            except FileNotFoundError as e:
                print(f"ファイルが見つかりませんでした。 {e}")


if __name__ == "__main__":

    # 入力するファイル名とurl
    files = {"fl_early_exit": "https://arxiv.org/pdf/2405.04249"}

    client = arxiv.Client()

    download_dir = Path('downloads')
    download_dir.mkdir(exist_ok=True)
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)

    download_paper(files, client, download_dir)
    pdf_to_markdown(download_dir, outputs_dir)
