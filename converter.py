from pathlib import Path
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from dotenv import load_dotenv
from google import genai
from typing import Any
import pymupdf4llm
import zipfile
import arxiv
import time
import random
import os
import re


def get_arxiv_id(file_url: str) -> str | None:
    """
    指定されたURLから正規表現を用いてarXiv IDを抽出

    :param file_url: 抽出対象となるPDFファイルのURL文字列
    :type file_url: str
    :return: 抽出されたarXiv ID（例: '2504.20571'）。マッチしない場合はNone
    :rtype: str
    """
    m = re.search(r"\b[0-9]{4}[.][0-9]{5,}(v[0-9]+)?\b", file_url)
    if m:
        return m.group()
    else:
        return None


def download_paper(url: str, client: arxiv.Client, download_dir: Path) -> str | None:
    """
    指定されたURLリストから論文PDFをダウンロードし、指定ディレクトリに保存

    :param files: 保存するURLの文字列
    :type files: str
    :param client: arxivライブラリのClientインスタンス
    :type client: object
    :param download_dir: PDFファイルの保存先ディレクトリパス
    :type download_dir: Path
    """
    # arxiv_idの取得
    arxiv_id = get_arxiv_id(url)

    if arxiv_id:
        # pdfファイルのダウンロード
        try:
            # ファイルの拡張子をpdfに変換
            arxiv_id_pdf = Path(f"{arxiv_id}.pdf")

            if arxiv_id_pdf.exists():
                print(
                    f"{arxiv_id_pdf}.pdfは既に存在するため、ダウンロードをスキップします。"
                )
                return arxiv_id

            search_by_id = arxiv.Search(id_list=[arxiv_id])
            result = next(client.results(search_by_id))
            result.download_pdf(dirpath=download_dir, filename=arxiv_id_pdf)
            print(f"{arxiv_id_pdf}のダウンロードが完了しました。")
            return arxiv_id

        except FileNotFoundError as e:
            print(f"ファイルが見つかりませんでした。 {e}")
            return None

    else:
        return None


def pdf_to_markdown(arxiv_id: str, download_dir: Path, outputs_dir: Path) -> None:
    """
    指定ディレクトリ内のPDFファイルをMarkdown形式に変換して保存
    pymupdf4llmを使用して変換を行い、ファイル名は維持したまま拡張子を.mdに変更

    :param download_dir: 変換元のPDFファイルが格納されているディレクトリ
    :type download_dir: Path
    :param outputs_dir: 変換後のMarkdownファイルを保存するディレクトリ
    :type outputs_dir: Path
    """

    file_path = download_dir / (arxiv_id + ".pdf")
    # 拡張子がpdfのファイルのみMarkdown化
    if file_path.suffix == ".pdf":
        try:
            md_text = pymupdf4llm.to_markdown(file_path)

            # 拡張子を除いたファイル名を取得
            filename = file_path.stem

            output_path = outputs_dir / (filename + ".md")

            if output_path.exists():
                print(f"{filename}.mdは既に存在するため、Markdown化をスキップします。")
                return

            output_path.write_bytes(md_text.encode("utf-8"))

            print(f"{filename}をmarkdown化しました")

        except FileNotFoundError as e:
            print(f"ファイルが見つかりませんでした。 {e}")


# Markdownファイルを読み込む関数
def read_markdown_file(file_path: Path) -> str:
    """
    指定されたパスのMarkdownファイルを読み込み、その内容を文字列として返す
    ファイルが存在しない場合や読み込みエラー時は、エラーメッセージを表示し空文字を返す

    :param file_path: 読み込むファイルのパス
    :type file_path: Path
    :return: ファイルのテキスト内容。エラー時は空文字
    :rtype: str
    """
    try:
        text = file_path.read_text(encoding="utf-8")
        return text
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        return ""
    except Exception as e:
        print(f"ファイルを読み込む際にエラーが発生しました: {e}")
        return ""


def clean_text(text: str) -> str:
    pattern_1 = r"^\*\*\s*(\d+(?:\.\d+)*)\s*\*\*\s+\*\*(.*?)\*\*.*$"
    pattern_2 = r"^\*\*(Acknowledgements|References)\*\*.*$"

    # 「**1** **Introduction**」 → 「## 1 Introduction」へ変換
    cleaned_text = re.sub(
        pattern_1, r"## \1 \2", text, flags=re.MULTILINE  # 行の先頭を探す
    )

    # 無番号見出しの変換
    cleaned_text = re.sub(
        pattern_2, r"## \1", cleaned_text, flags=re.MULTILINE | re.IGNORECASE
    )

    return cleaned_text


def split_markdown_by_section(outputs_dir: Path) -> None:
    """
    Markdownファイルをセクション（見出し）ごとに分割し、指定サイズを超える場合はさらにチャンク分割して保存
    分割されたファイルは outputs_dir/split/元のファイル名/ 配下に保存される

    :param outputs_dir: 分割対象のMarkdownファイルが格納されているディレクトリ
    :type outputs_dir: Path
    """
    # 保存先の親ディレクトリを作成
    split_base_dir = outputs_dir / "split"
    split_base_dir.mkdir(parents=True, exist_ok=True)

    headers_to_split_on = [("##", "Section")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False  # 見出しを保持
    )

    # 分割する最大と最小の文字数を設定
    max_chunk_size = 3000
    min_chunk_size = 500

    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,  # 1チャンクの最大文字数
        chunk_overlap=0,  # 重複させる文字数
        separators=["\n\n", "\n", "。", " ", ""],  # 優先して切りたい文字リスト
    )

    for file_path in outputs_dir.iterdir():
        if file_path.is_file() and file_path.suffix == ".md":
            text = read_markdown_file(file_path)

            cleaned_text = clean_text(text)

            header_splits = markdown_splitter.split_text(cleaned_text)

            # 保存用ディレクトリ作成
            file_split_dir = split_base_dir / file_path.stem
            file_split_dir.mkdir(parents=True, exist_ok=True)

            final_chunks = []
            skip_Section = ["Acknowledgements", "References"]

            for split in header_splits:
                # skip_Sectionに含まれる章が登場したら繰り返し終了
                if split.metadata.get("Section") in skip_Section:
                    break

                # 3000文字を超えた場合
                if len(split.page_content) > max_chunk_size:
                    sub_splits = chunk_splitter.create_documents(
                        [split.page_content], metadatas=[split.metadata]
                    )

                    # 分割されたサブパーツにも元のヘッダー情報を付与
                    for sub in sub_splits:
                        sub.metadata = (
                            split.metadata
                        )  # "Section": "Introduction" などを引き継ぐ
                        final_chunks.append(sub)

                else:
                    final_chunks.append(split)

            batched_chunks = []  # 結合後のデータを格納するリスト
            current_batch_text = ""  # 結合中のテキストを一時保存する変数
            current_batch_title = ""  # 結合中のチャンクの代表タイトル

            # max_chunk_sizeを超えない範囲でファイルを結合するループ
            for split in final_chunks:
                # 文章を取得
                content = split.page_content
                # タイトルを取得
                section_title = split.metadata.get("Section", "preamble")

                if not current_batch_text:
                    current_batch_text = content
                    current_batch_title = section_title

                elif len(current_batch_text) + len(content) > max_chunk_size:
                    batched_chunks.append(
                        {"title": current_batch_title, "content": current_batch_text}
                    )
                    current_batch_text = content
                    current_batch_title = section_title

                else:
                    current_batch_text += "\n\n" + content

            # ループ後の端数処理
            if current_batch_text:

                # 500文字を下回った場合
                if len(current_batch_text) < min_chunk_size and batched_chunks:
                    batched_chunks[-1]["content"] += "\n\n" + current_batch_text

                else:
                    batched_chunks.append(
                        {"title": current_batch_title, "content": current_batch_text}
                    )

            # mdファイルへの書き込み
            for i, chunk in enumerate(batched_chunks):
                title = chunk["title"]
                combined_text = chunk["content"]
                # ファイル名に使えない文字（スペースやスラッシュ）を置換
                safe_title = re.sub(r'[\\/*?:"<>| ]', "_", title)
                output_path = file_split_dir / (f"{i:02d}_{safe_title}" + ".md")

                if output_path.exists():
                    print(
                        f"{output_path.name} は既に存在するため、Markdownの分割はスキップします。"
                    )

                    continue
                output_path.write_bytes(combined_text.encode("utf-8"))

            print(
                f"{file_path.stem} を {len(final_chunks)} チャンクから"
                f"{len(batched_chunks)} ファイルに結合・削減しました。"
            )


def generate_with_retry(
    client: genai.Client, model: str, contents: list[Any], max_retries: int = 10
) -> Any | None:
    """
    Google GenAI API呼び出しをリトライ処理付きで実行する
    レート制限（429）やサーバーエラー（503）発生時、指数バックオフを用いて待機時間を増やしながら再試行

    :param client: Google GenAI Clientインスタンス
    :param model: 使用するモデル名
    :param contents: プロンプトと入力テキストのリスト
    :param max_retries: 最大リトライ回数
    :return: 生成結果オブジェクト。失敗時はNone
    """
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            error_str = str(e)
            # レート制限（429）またはサーバー過負荷の場合
            if (
                "429" in error_str
                or "Resource exhausted" in error_str
                or "503" in error_str
            ):
                # 待機時間を計算: (回数×5秒) + 乱数。例: 5s, 10s, 15s...
                wait_time = (2**attempt) + random.uniform(1, 5)
                # 初期の待機時間を少し長めに設定
                if attempt > 2:
                    wait_time += 10

                print(
                    f"レート制限を検知。{wait_time:.1f}秒 待機して再試行します..."
                    f"({attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                # その他のエラーは即座に例外を投げる
                raise e

    print("最大リトライ回数を超えました。")
    return None


def clean_latex(text: str) -> str:
    # 図と表を削除
    pattern_1 = r"\\begin\s*\{(table|figure)\*?\}.*?\\end\s*\{\1\*?\}"
    pattern_2 = r"\\begin\s*\{(tabular)\*?\}.*?\\end\s*\{\1\*?\}"

    # Markdown表を削除
    pattern_3 = r"^(?:\|.*\|\n+)+(\|.*\|)$"

    # 数式以外の_を削除
    pattern_4 = r"_(.*?)_"

    cleaned_text = re.sub(pattern_1, "", text, flags=re.DOTALL)
    cleaned_text = re.sub(pattern_2, "", cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(pattern_3, "", cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(pattern_4, r"\1", cleaned_text, flags=re.MULTILINE)

    return cleaned_text


def translate_to_latex(outputs_dir: Path) -> None:
    """
    分割されたMarkdownファイルを読み込み、Gemini APIを使用して日本語LaTeX形式に翻訳
    翻訳結果は outputs_dir/tex/ 配下に配下に保存される

    処理内容:
    1. プロンプトによる厳格なLaTeXフォーマット指示
    2. 数式、参照、アルゴリズム環境の保持
    3. Markdownコードブロックの除去処理

    :param outputs_dir: データが格納されているルートディレクトリ
    :type outputs_dir: Path
    :raises ValueError: 環境変数 GEMINI_API_KEY が設定されていない場合
    """
    load_dotenv()

    # 翻訳するファイルのパスを取得
    translate_path = outputs_dir / "split"

    # texファイルの保存先の親ディレクトリを作成
    tex_base_dir = outputs_dir / "tex"
    tex_base_dir.mkdir(parents=True, exist_ok=True)

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY が設定されていません。")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # プロンプト
    prompt = r"""                                                                                      
    タスク: 以下の学術論文のテキストを英語から日本語に翻訳し、LaTeXコードとして整形してください。

    # Rules (Strictly Follow):
    1. 言語: 出力は必ず日本語で行うこと。いかなる理由があっても、段落や文章全体を英語のまま出力することを固く禁じる。
    2. 専門用語の扱い: アルゴリズム名（例: ResNet）や一般化されていない技術用語のみ、英単語のまま文中に組み込むことを許可する。
       ただし、その場合でも文全体は必ず日本語で構成すること。
    3. フォーマット: 標準的なLaTeX構文を使用すること。
       - アルゴリズムには algorithm, algpseudocode を使用すること。
       - だ・である口調を使用すること。
       - 数式 ($...$) は元のままにすること。
       - 数式モード（$...$ または $$...$$）の中では、アンダースコア _ を絶対にエスケープ（\_）しないこと。
        ・悪い例: \sigma\_i
        ・良い例: \sigma_i
       - 変数（x, y, hなど）や数値比較（> < =）は、必ず数式モード $...$ で囲むこと。
        ・悪い例: 悪い例: x > 0
        ・良い例: 良い例: $x > 0$
    4. 完全性: 全ての内容を翻訳すること。要約はしないこと。
    5. Markdown記法の完全排除:
       - 出力テキストにMarkdownの装飾記号（**bold**, _italic_, `code`）を絶対に残さないこと。
       - 太字は \textbf{...} に、斜体は \textit{...} に変換すること。
       - 悪い例: **x**
       - 良い例: \textbf{x} or $\mathbf{x}$
       - 悪い例: _Plain_
       - 良い例: \textit{Plain}
    6. 数式のブロック化:
       - 独立した数式や定義式は、必ず $$ ... $$ または \begin{equation} ... \end{equation} で囲むこと。
       - インラインのMarkdown（_W_ など）で数式を表現しないこと。
    7. 出力: 出力は純粋なLaTeXコードのみ。Markdownのフォーマット（latex ...  など）は不要。
    8. プリアンブルなし: \documentclass, \usepackage, \begin{document},
       または \end{document} を含めないこと。
       翻訳された本文のみを出力すること。
    9. 見出しの変換ルール:
       - 1つの数字 (例: 1) は \section*{} に変換すること。
       - ドット1つ (例: 1.1) は \subsection*{} に変換すること。
       - ドット2つ (例: 1.1.1) は \subsubsection*{} に変換すること。
       - 数字のない太字 (例: **Abstract**) は \textbf{} を使用すること。

    # Input Text (English):
    """

    for translate_dir_path in translate_path.iterdir():
        # 出力先のディレクトリを作成
        tex_dir = tex_base_dir / translate_dir_path.name
        tex_dir.mkdir(exist_ok=True)

        sort_md_files = list(translate_dir_path.glob("*.md"))
        try:
            # 00_..., 01_... のようにファイル名の数値でソート
            sort_md_files.sort(key=lambda f: int(f.stem.split("_")[0]))
        except ValueError:
            sort_md_files.sort()

        test_files = sort_md_files

        for file in test_files:
            if file.suffix != ".md":
                continue

            # 出力ファイルのパス
            file_path = tex_dir / (file.stem + ".tex")

            if file_path.exists():
                print(f"{file.name} .texは既に存在するため、翻訳をスキップします。")
                continue

            try:
                file_content = file.read_text(encoding="utf-8")

                # Gemma 3 モデルを指定
                response = generate_with_retry(
                    client=client,
                    model="gemma-3-27b-it",
                    contents=[prompt, file_content],
                    max_retries=5,
                )

                if response and response.text:
                    # GemmaがMarkdownコードブロックを含めて返してくる場合の対策
                    clean_text = (
                        response.text.replace("```latex", "")
                        .replace("```tex", "")
                        .replace("```", "")
                        .strip()
                    )

                    clean_text = clean_latex(clean_text)

                    file_path.write_bytes(clean_text.encode("utf-8"))
                    print(f"DONE: {file.name} の翻訳完了")
                else:
                    print(f"WARN: {file.name} の生成結果が空でした。")

            except Exception as e:
                print(f"ERROR {file.name}: {e}")


def create_final_package(outputs: Path, col_num: int) -> None:
    """
    翻訳された複数のTeXファイルを結合し、テンプレートに埋め込んで最終的なZipパッケージを作成

    処理内容:
    1. tex_template.txt の読み込み
    2. 分割されたTeXファイルを順序通りに結合
    3. Zipファイルとして outputs/package/ に出力

    :param outputs: データが格納されているルートディレクトリ
    :type outputs: Path
    """
    # 入力するTexファイルの親ディレクトリ指定
    tex_files_path = outputs / "tex"

    # 出力ファイルの親ディレクトリ作成
    output_base_path = outputs / "package"

    for translated_paper_tex_dir in tex_files_path.iterdir():
        paper_name = translated_paper_tex_dir.name

        # texのテンプレートを読み込み
        parent_file_path = Path(__file__).parent
        if col_num == 2:
            file_path = parent_file_path / "tex_template_2col.txt"
            main_tex = file_path.read_text(encoding="utf-8")
        else:
            file_path = parent_file_path / "tex_template.txt"
            main_tex = file_path.read_text(encoding="utf-8")

        # 出力先のディレクトリを作成
        main_tex_dir = output_base_path / paper_name
        main_tex_dir.mkdir(parents=True, exist_ok=True)

        sort_tex_files = list(translated_paper_tex_dir.glob("*.tex"))

        try:
            sort_tex_files.sort(key=lambda f: int(f.stem.split("_")[0]))
        except ValueError:
            sort_tex_files.sort()

        for file in sort_tex_files:
            if file.suffix != ".tex":
                continue

            main_tex += "\n" + file.read_text(encoding="utf-8")

        main_tex += "\n" + r"\end{document}"

        if main_tex:
            main_zip = main_tex_dir / "main.zip"

            try:
                with zipfile.ZipFile(main_zip, "w") as zf:
                    zf.writestr(f"{paper_name}.tex", main_tex)
                print(f"{paper_name}のtexファイルが完成しました。")

            except Exception as e:
                print(f"ERROR {paper_name}: {e}")
