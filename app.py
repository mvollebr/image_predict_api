#!/usr/bin/env python3

import connexion
import json

from fastai.vision import *
#from fastai.metrics import error_rate
import os
#from fastai.widgets import *
import re
import requests

path_model_loc = "C:/Users/mickv/Documents/Jupyter Books/Workbooks/Projects/web_server/example/model"
learn_predict = load_learner(path_model_loc)



def basic_auth(username, password, required_scopes=None):
	if username == 'admin' and password == 'secret':
		return {'sub': 'admin'}

def download_pic(url):
	cwd = os.getcwd()
	pic = open(f"{cwd}/image/predict.jpg","wb")
	pic.write(requests.get(url).content)
	pic.close()


def post_predictions(query):
	for item in query:
		url = item["text"]
	cwd = os.getcwd()
	pic = open(f"{cwd}/image/predict.jpg","wb")
	pic.write(requests.get(url).content)
	pic.close()
	
	img_loc = f"{cwd}/image/predict.jpg"
	#response = []
	img = open_image(img_loc)
	pred_class,pred_idx,outputs = learn_predict.predict(img)
	#response.append(pred_class)
	os.remove(img_loc)
	
	outputs_list = list(outputs)
	pred_class = learn_predict.data.classes
	pattern_1 = "[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)"
	pattern_2 = "[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
	pattern_3 = "[1-9]"
	
	pred_idx_str = str(pred_idx)
	pred_idx_num = int(re.search(pattern_2,pred_idx_str,0).group())
	
	out_list = []
	for out in outputs_list:
		out = str(out)
		out_list.append("{:.3f}".format(float(re.search(pattern_2,out,).group(0))))
	
	print(f"Prediction Category: {pred_class[pred_idx_num]} and prediction score: {out_list[pred_idx_num]}")
	print("")
	
	for x in range(len(pred_class)):
		print(f"Category: {pred_class[x]} and prediction: {out_list[x]}")
	
	dict_pred_score = dict(zip(pred_class, out_list))
	json_out_final = json.dumps(dict_pred_score)
	
	return json_out_final


app = connexion.App(__name__)
app.add_api('swagger.yaml')

if __name__ == '__main__':
    app.run(port=8080, server='gevent')
