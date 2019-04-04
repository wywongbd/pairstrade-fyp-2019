# activate environment
source activate traders_nlp

# run bokeh
bokeh serve flask/static/plots/client_demo.py --allow-websocket-origin=127.0.0.1:5000