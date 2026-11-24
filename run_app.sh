#!/bin/bash
# Double-click this (or run ./run_app.sh) to start the House Price Predictor app.
cd "$(dirname "$0")"
.venv/bin/streamlit run app.py
