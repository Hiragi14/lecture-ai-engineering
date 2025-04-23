import streamlit as st
import pandas as pd
import numpy as np
import time
import torch
from transformers import pipeline

from config import MODEL_NAME
import llm
from llm import load_model, generate_response


st.title("exercise01")
st.sidebar.header("チャット履歴")
st.sidebar.info("チャット履歴を表示します。")

@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

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