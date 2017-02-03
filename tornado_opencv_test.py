import tornado.httpserver
import tornado.ioloop
import tornado.web
import json
import os,uuid
import cv2
import numpy as np
import base64
import codecs
import ftplib
import datetime
import pdb
import time
#from PIL import Image

from mysql_temp import loginSelect
from mysql_temp import registerCheck
from mysql_temp import registerInsert
from mysql_temp import emailCheck

from opencv import readCVImage
from opencv import solution
from opencv import HCA

from send_email import sendEmailUser
__UPLOADS__ = ""

from tornado.options import options, define, parse_command_line
from tornado.ioloop import IOLoop
from tornado.web import Application
from tornado.websocket import WebSocketHandler

width  = 800
height = 800
resultRadius = 30
saveName = "result.png"

class MainHandler(tornado.web.RequestHandler):
	def get(self):
		self.render("main.html")

class sendEmail(tornado.web.RequestHandler):
	def post(self):
		json_data = self.request.body
		json_encode = json_data.decode('utf-8')
		data = json.loads(json_encode)

		ID, PW = emailCheck(data["User"][0]["email"])
		if (ID == "0") :
			alpha = -1
		else :
			print(ID,PW, data["User"][0]["email"])
			alpha = sendEmailUser(ID, PW, data["User"][0]["email"])
			 
		secResponse={"finalResult" : [{"ID":alpha}]}
		secResponse = json.dumps(secResponse)
		self.write(secResponse)

		self.set_header("Content-type",  "application/json")		

class Login(tornado.web.RequestHandler):
	def post(self):
		json_data = self.request.body
		json_encode = json_data.decode('utf-8')
		data = json.loads(json_encode)
		alpha = loginSelect(data["User"][0]["userID"],data["User"][0]["userPW"])
		print(alpha)
		secResponse={"finalResult" : [{"ID":alpha}]}
		secResponse = json.dumps(secResponse)
		self.write(secResponse)

		self.set_header("Content-type",  "application/json")
		
class Register(tornado.web.RequestHandler):
	def post(self):	
		json_data = self.request.body
		json_encode = json_data.decode('utf-8')
		data = json.loads(json_encode)
		beta = registerCheck(data["User"][0]["userID"], data["User"][0]["email"])
		if (beta == True) :
			alpha = registerInsert(data["User"][0]["userID"],data["User"][0]["userPW"],data["User"][0]["email"],data["User"][0]["address"])
			print(alpha)
		elif (beta == False) :
			alpha = -1
		secResponse={"finalResult" : [{"ID":alpha}]}
		secResponse = json.dumps(secResponse)
		self.write(secResponse)
		self.set_header("Content-type",  "application/json")

class Up(tornado.web.RequestHandler):
	def post(self):	
		level = HCA()
		analyte = solution()

		print("Send Data2")		
		secResponse={"finalResult" : [{"aresult":analyte[0],"bresult":analyte[1],"cresult":analyte[2],"dresult":analyte[3],"HCA":level}]}
		secResponse = json.dumps(secResponse)
		self.write(secResponse)
		self.set_header("Content-type",  "application/json")

class Upload(tornado.web.RequestHandler):
	def post(self):
		print("Received Data")
		## 시간측정시작
		start_time = time.time()
		result_rgb_lists = []
		final_RGB_LIST = []
		row = 4
		col = 4

		## 결과값 이미지 파일 생성 및 저장 
		fileinfo = self.request.files['file1'][0]
		fname = fileinfo['filename']

		extn = os.path.splitext(fname)[1]
		cname1 = str(datetime.datetime.now()) + extn
		fh = open(__UPLOADS__ + cname1, 'wb')
		filebytes1 = fileinfo['body']
		fh.write(filebytes1)

		fileinfo = self.request.files['file2'][0]
		fname = fileinfo['filename']

		extn = os.path.splitext(fname)[1]
		cname2 = str(datetime.datetime.now()) + extn
		fh = open(__UPLOADS__ + cname2, 'wb')
		filebytes2 = fileinfo['body']
		fh.write(filebytes2)
	
		body = self.get_body_argument('comment')
		print(body)

		result, circled_img_name1 = readCVImage(cname1, row, col)
		cv2.imwrite('picture1.jpg',circled_img_name1)
		result_rgb_lists.append(result)
		result, circled_img_name2 = readCVImage(cname2, row, col)
		cv2.imwrite('picture2.jpg',circled_img_name2)		
		result_rgb_lists.append(result)


		for i,j in zip(result_rgb_lists[0], result_rgb_lists[1]):
			final_RGB_LIST.append([abs(i[0]-j[0]), abs(i[1] - j[1]) , abs(i[2] - j[2])])
 
		count = 0
		print()
		print(result_rgb_lists[0])
		print()
		print(result_rgb_lists[1])
		print()
		print(final_RGB_LIST)

                ##############FTP Upload by In-Jo
		
		slist = ["First RGB :",result_rgb_lists[0],
		" ","Second RGB :",result_rgb_lists[1],
		" ","RGB Difference :",final_RGB_LIST]
 
		with open("/home/biolab/test/opencv/qq", mode="w", encoding="utf8") as f:
		    for s in slist:
		        print(s, file=f)

		ftp = ftplib.FTP('benedict78.dothome.co.kr','benedict78','rkddlswh1')
		ftp.cwd("/html")
		fname0 = "rgb"
		fname1 = "img1"
		fname2 = "img2"
		fname3 = "img3"
		fname4 = "img4"
		full_fname = "/home/biolab/test/opencv/qq"
		ftp.storlines('STOR '+ fname0, open(full_fname, 'rb'))
		ftp.storbinary('STOR '+ fname1, open(cname1, 'rb'))
		ftp.storbinary('STOR '+ fname3, open('picture1.jpg', 'rb'))
		ftp.storbinary('STOR '+ fname2, open(cname2, 'rb'))
		ftp.storbinary('STOR '+ fname4, open('picture2.jpg', 'rb'))
		ftp.quit()
                #############

		blank_image = np.zeros((height,width,3), np.uint8)
		blank_image[:] = (0, 0, 0)

		resultCircleList = []


		intervalRow = width / (row + 1)
		intervalCol = height / (col + 1)
		west = intervalRow
		north = intervalCol

		for y in range(0,row):
			for x in range(0,col):
				resultCircleList.append( [west + intervalRow * x, north + intervalCol * y])

		resultCircleList = np.uint16(np.around(resultCircleList))

		for i in resultCircleList:
			color = ((int(final_RGB_LIST[count][0]),int(final_RGB_LIST[count][1]),int(final_RGB_LIST[count][2])))
			cv2.circle(blank_image,(i[0], i[1]),resultRadius,color,thickness=-1)
			count +=1 
			
		cv2.imwrite(saveName,blank_image)
		
		print("Send Data")
## for DB
##analyte = solution()

##insert(1,filebytes1,filebytes2,analyte)

		## 시간측정완료
		end_time = time.time()
		print ("Spended Time =" ,end_time - start_time)
##**********
		level = HCA()
		analyte = solution()
		
		with open(saveName, 'rb') as f:
			data11 = f.read()
			UU = base64.b64encode(data11)
			UUU = UU.decode('utf-8')
		print("Send Data")		
		secResponse={"finalResult" : [{"aresult":analyte[0],"bresult":analyte[1],"cresult":analyte[2],"dresult":analyte[3],"HCA":level, "Image":UUU}]}
		secResponse = json.dumps(secResponse)

		print("Send Data1")
		self.write(secResponse)
		print("Send Data2")

		self.set_header("Content-type",  "application/json")


def main():


	parse_command_line()
	settings = dict(
		debug  =  True
	)
	tornado_app = tornado.web.Application(
		[
			(r'/', MainHandler),
			(r"/upload", Upload),
			(r"/up", Up),
			(r"/login", Login),
			(r"/register", Register),
			(r"/sendemail", sendEmail),
		], **settings)

	server = tornado.httpserver.HTTPServer(tornado_app)
	server.listen('8080', address = '0.0.0.0')
	tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
	main()

