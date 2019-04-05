# activate environment
conda activate "fyp2019"
activate "fyp2019"
source activate "fyp2019"

# run bokeh
bokeh serve flask/static/plots/client_demo.py --allow-websocket-origin=127.0.0.1:5000
