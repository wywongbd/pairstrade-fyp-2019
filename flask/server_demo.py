from flask import Flask, flash, redirect, render_template, request, session, abort
from bokeh.embed import server_document
app = Flask(__name__)

# https://gist.github.com/Wildcarde/6841f00f0a0cd52ade09964a0fdb5684

# to activate bokeh server
# cd to the update.py location and run this
# bokeh serve ./<update1>.py ./<update2>.py --allow-websocket-origin=127.0.0.1:5000
# bokeh serve ./client-demo.py --allow-websocket-origin=127.0.0.1:5000

@app.route("/")
def index():
	# format
	# name-of-js-to-add = server_document(url="http://localhost:5006/<py file that generate the graph and updates it>")
    
    client_demo_script=server_document(url="http://localhost:5006/client_demo")
    # script2=server_document(url="http://localhost:5006/update2")
    return render_template('client-demo.html', client_plot=client_demo_script)

if __name__ == "__main__":
    app.run()