'''
	Contoh Deloyment untuk Domain Computer Vision (CV)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
'''

# =[Modules dan Packages]========================

from flask import Flask, render_template,request,jsonify, flash, request, redirect, url_for
import pandas as pd
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import numpy as np
import os

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')

app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG', '.png']
app.config['UPLOAD_PATH']        = './static/images/uploads/'

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
	return render_template('index.html')

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
	hasil_prediksi  = '(none)'
	gambar_prediksi = '(none)'

	# Get File Gambar yg telah diupload pengguna
	uploaded_file = request.files['file']
	filename      = secure_filename(uploaded_file.filename)
	# Periksa apakah ada file yg dipilih untuk diupload
	if filename != '':
	
		# Set/mendapatkan extension dan path dari file yg diupload
		file_ext        = os.path.splitext(filename)[1]
		gambar_prediksi = '/static/images/uploads/' + filename
		
		# Periksa apakah extension file yg diupload sesuai (jpg)
		if file_ext in app.config['UPLOAD_EXTENSIONS']:
			
			# Simpan Gambar
			uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
			
			# Memuat Gambar
			test_image = './static/images/uploads/' + filename

			# Adding mediapipe  
			mp_drawing = mp.solutions.drawing_utils
			mp_drawing_styles = mp.solutions.drawing_styles
			mp_hands = mp.solutions.hands

			with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
				img = cv2.imread(test_image)
				img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				results = hands.process(img_rgb)
				if not results.multi_hand_landmarks:
					print('no hands detected')
				
				image_height, image_width, _ = img.shape

				annotated_img = img.copy()

				for hand_landmarks in results.multi_hand_landmarks:
					mp_drawing.draw_landmarks(
						annotated_img,
						hand_landmarks,
						mp_hands.HAND_CONNECTIONS,
						mp_drawing_styles.get_default_hand_landmarks_style(),
						mp_drawing_styles.get_default_hand_connections_style())
				
				cv2.imwrite('./static/images/predictions/'+ filename + '.png', cv2.flip(annotated_img, 1))
				
				if not results.multi_hand_world_landmarks:
					pass
				
				# output = cv2.imshow(annotated_img)

			# Body pose estimation and prediction
			
			# Return hasil prediksi dengan format JSON
			return jsonify({
				"prediksi": 'hands predicted',
				"gambar_prediksi" : './static/images/predictions/' + filename + '.png'
			})
		else:
			# Return hasil prediksi dengan format JSON
			gambar_prediksi = '(none)'
			return jsonify({
				"prediksi": 'hands not predicted',
				"gambar_prediksi" : gambar_prediksi
			})

# =[Main]========================================		

if __name__ == '__main__':
	# Run Flask di localhost 
	app.run(host="localhost", port=5000, debug=True)
	
	


