# activate environment
source activate fyp

# run bokeh
bokeh serve flask/static/plots/client-demo.py --allow-websocket-origin=127.0.0.1:5000