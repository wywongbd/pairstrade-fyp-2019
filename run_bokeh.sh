# activate environment
source activate fyp2019

# run bokeh
bokeh serve flask/static/plots/client_demo.py --allow-websocket-origin=127.0.0.1:5000
