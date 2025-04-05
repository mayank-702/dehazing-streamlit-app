#!/bin/bash

# Streamlit Cloud specific setup (if needed)
mkdir -p ~/.streamlit/

echo "\
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
