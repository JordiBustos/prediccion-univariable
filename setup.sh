mkdir -p ~/.streamlit/

echo "[theme]
base="dark"
primaryColor="#0000ff"
secondaryBackgroundColor="#0000ff"
font='sans serif'
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
