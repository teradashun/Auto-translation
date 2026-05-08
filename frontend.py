import streamlit as st
import requests

# 出力ファイルの行数のデフォルト値を設定
col_num = 1

# ユーザから受け取る入力フォーム
url = st.text_input("URL")
form = st.radio("出力形式", ["Markdown", "Tex"])

# formがTexなら新たなラジオボタンの表示
if form == "Tex":
    col_num = st.radio("出力の行数", [1, 2])

if st.button("実行"):
    if url:
        with st.spinner("論文を変換・翻訳しています。しばらくお待ちください..."):
            input_data = {"url": url, "form": form, "col_num": col_num}
            try:
                response = requests.post("http://backend:8000/access", json=input_data)

                if response.status_code == 200:
                    st.session_state["file_data"] = response.content
                    st.session_state["form_type"] = form
                    st.session_state["is_error"] = False
                    st.success("変換が完了しました！")

                else:
                    # バックエンドでエラーが起きた場合
                    st.session_state["is_error"] = True

            except Exception as e:
                st.error(f"バックエンドとの通信に失敗しました: {e}")

    else:
        st.warning("URLを入力してください")

# データが存在すればダウンロードボタンを表示
if "file_data" in st.session_state and not st.session_state.get("is_error"):
    if st.session_state.get("form_type") == "Markdown":
        st.download_button(
            label="ファイルのダウンロード",
            data=st.session_state["file_data"],
            file_name="result.md",
        )
    else:
        st.download_button(
            label="ファイルのダウンロード",
            data=st.session_state["file_data"],
            file_name="result.zip",
        )

elif st.session_state.get("is_error"):
    st.error("ファイルを取得できませんでした。")
