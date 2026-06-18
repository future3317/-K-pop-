from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from kpop_scope import analyze

st.set_page_config(page_title="KPopScope", layout="wide")
st.title("KPopScope: K-pop 音乐分析与赏析")
st.caption("上传本地音频，生成 MIR 特征、段落分析、标签和中文赏析报告。请只上传你有权分析的音频。")

uploaded = st.file_uploader("选择音频文件", type=["mp3", "wav", "flac", "m4a", "ogg"])
use_stems = st.checkbox("启用 audio-separator 声部分离（较慢，需要安装 audio-separator）", value=False)
plots = st.checkbox("生成图像", value=True)
use_mert = st.checkbox("抽取 MERT embedding（需要 torch/transformers，会较慢）", value=False)
classifier_path = st.text_input("K-pop classifier checkpoint 路径（可选）", value="")
max_duration = st.number_input("调试用时长上限（秒，0 表示完整音频）", min_value=0, value=0)

if uploaded and st.button("开始分析"):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / uploaded.name
        tmp.write_bytes(uploaded.read())
        out = Path(td) / "out"
        with st.spinner("分析中..."):
            result = analyze(
                tmp,
                output_dir=out,
                use_stems=use_stems,
                make_plots=plots,
                max_duration=float(max_duration) if max_duration else None,
                use_mert=use_mert or bool(classifier_path.strip()),
                classifier_path=classifier_path.strip() or None,
            )
        st.subheader("Top tags")
        st.write(result["explanations"].get("top_tags", []))
        st.subheader("赏析报告")
        st.markdown(result["report_markdown"])
        st.subheader("结构化 JSON")
        st.json({k: v for k, v in result.items() if k != "report_markdown"})
