from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

application = Flask(__name__)

dic = {0:'Taj Mahal', 1:'Giza Pyramid', 2:'Christ'}

model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(100,100))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 100,100,3)
	p = np.argmax(model.predict(i), axis=-1)#model.predict_classes(i)#
	return dic[p[0]]


# routes
@application.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@application.route("/about")
def about_page():
	return "Monument detector"

@application.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#application.debug = True
	app.run(host="0.0.0.0", port=5000)
