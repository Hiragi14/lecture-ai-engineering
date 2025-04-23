import streamlit as st
import pandas as pd
import numpy as np
import time

import llm
from llm import load_model, generate_response


st.title("exercise01")
st.sidebar.header("チャット履歴")
st.sidebar.info("チャット履歴を表示します。")


def main_chat(pipe):
    """メインのチャットインターフェース"""
    if prompt := st.chat_input():
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": prompt})

        answer, response_time = generate_response(pipe, prompt)
        # sample_response = "これはサンプルの応答です。"
        st.session_state.messages.append({"role": "assistant", "content": answer})

        for i in range(0, len(st.session_state.messages), 2):
            st.chat_message("user").write(st.session_state.messages[i]["content"])
            st.chat_message("assistant").write(st.session_state.messages[i+1]["content"])

pipe = llm.load_model()

try:
    main_chat(pipe)
except Exception as e:
    st.error(f"エラーが発生しました: {e}")
    st.stop()