"""Streamlit entry: programmatic navigation — sidebar titles are controlled by ``st.Page``."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).resolve().parent


def main() -> None:
    st.set_page_config(page_title="localchat-rag", layout="wide")
    home = st.Page(
        str(_APP_DIR / "landing_page.py"),
        title="localchat-rag",
        icon=":material/home:",
    )
    chat = st.Page(str(_APP_DIR / "chat_page.py"), title="Chat", icon=":material/chat:")
    ingestion = st.Page(
        str(_APP_DIR / "ingestion_page.py"),
        title="Ingestion",
        icon=":material/database:",
    )
    st.navigation([home, chat, ingestion]).run()


if __name__ == "__main__":
    main()
