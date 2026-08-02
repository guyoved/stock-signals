#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=.
streamlit run dashboard/app.py --server.headless true
