# import os
# from flask import Flask, render_template, send_from_directory
# app = Flask(__name__)

# @app.route("/")
# def index():
#    return render_template("index.html")

# if __name__ == '__main__':
#    app.run(debug = True)


# === bokeh demo at below ===
# embedding the graph to html
from flask import Flask, render_template, request
from bokeh.resources import CDN
from bokeh.plotting import figure
from bokeh.embed import components

app = Flask(__name__)

def create_figure():

	plot = figure()
	plot.circle([1,2], [3,4])
	return plot

# Index page
@app.route('/')
def index():
	# Create the plot
	plot = create_figure()
	# tag here means the tag to reference to the new bokeh chart, saved as a js file
	js, plot_tag = components(plot, CDN, "/Users/brendantham/Desktop/FYP/Flask/static/plots")

	# TODO: 
	# 1) fix URLS
	# 2) figure out where to store the js files for future load use 

	# with open('/Users/brendantham/Desktop/FYP/Flask/static/plots/plot1.js', 'w') as f:  
	# 	f.write(js)
		
	return render_template("index.html", script1 = js, plot1 = plot_tag)

# With debug=True, Flask server will auto-reload 
# when there are code changes
if __name__ == '__main__':
	app.run(port=5000, debug=True)


