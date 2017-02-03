import time 
import os
import ftplib
import cv2
import numpy as np
import math
import random
import datetime

def readCVImage(name, row, col):
	try:
		img = cv2.imread(name)
		x_length, y_length = img.shape[:2]
		
	
	except Exception:
		print("해당 파일을 읽을 수 없습니다")
		return
	
	lower_bgr = np.array([0,0,0], np.uint8)
	upper_bgr = np.array([255,255,130], np.uint8)
	
	mask = cv2.inRange(img, lower_bgr, upper_bgr)
##	cv2.imwrite('sample5.jpg',mask)	
	(h, w) = img.shape[:2]
	temp1image = np.zeros((h,w,3), np.uint8)
	temp1image[:] = (255,255,255)
	img = cv2.bitwise_and(temp1image,temp1image, mask= mask)
	
	point = readCVImage_Contours(img)
	vertex_array = vertex(point)

	value = verify_Square(vertex_array)

	if value == False :
		vertex_array = vertex_newPoint(point, vertex_array)
		print(" i'm useful")

		value = verify_Square(vertex_array)

		if value == False :
			return (fail())
		else : 
			return (success(name, vertex_array))

	else :
		return (success(name, vertex_array))

def fail() :
	result_rgb_lists = []
	for i in range(16):
		result_rgb_lists.append((0,0,0))	
	return(result_rgb_lists)

def success(name, vertex_array):
	circle = circle_coordinate(vertex_array)
	img2 = cv2.imread(name)
	for i in range(16):
		cv2.circle(img2,(circle[i][0],circle[i][1]),50,(0,255,0),10)

	cv2.imwrite('result1.jpg',img2)
##	find average BGR at vertex point

	resultRadius = 15
	rgb_lists = []
	for i in range(16) :
		x1 = circle[i][0]-resultRadius
		x2 = circle[i][0]+resultRadius
		y1 = circle[i][1]-resultRadius
		y2 = circle[i][1]+resultRadius

		cropped_image = img2[y1:y2, x1:x2]
		rgb_lists.append(cropped_image)
		##cv2.imwrite('cropped_%d.jpg' %(i),cropped_image)

	result_rgb_lists = []
	for rgb_list in rgb_lists :
		rgb_result = getAvg(rgb_list)
		result_rgb_lists.append(rgb_result)
##	print (result_rgb_lists)
	return(result_rgb_lists, img2)
## check in case, all black.

def getAvg(rgb_list):	#RGB list의 상위, 하위 10%를 제한  평균값을 반환하는 함수, 추후에 산정 방식이 달라질 수 있다
   tempRGB = []
   for row in rgb_list:
      for col in row:
         if list(col) == [0,0,0]:
            pass
         else:
            tempRGB.append(list(col))

   for a in range(0,3):
      tempRGB.sort(key = lambda x: x[a])
      listCount = len(tempRGB)
      count = 0
      for i in tempRGB:
         if(count < int(listCount * 0.1)):
            i[a] = -1
         elif(count > int(listCount * 0.9)):
            i[a] = -1
         else:
            pass
         count += 1

   count = 0
   r,g,b = (0,0,0)
   for i in tempRGB:
      if(i[0]==-1 or i[1]== -1 or i[2]== -1):
         pass
      else:
         r += i[0]
         g += i[1]
         b += i[2]
         count += 1
   r = int(r / count)
   g = int(g / count)
   b = int(b / count)

   return (r,g,b)

def verify_Square(vertex_array):
	dist_2_4 = distance(vertex_array[0],vertex_array[1])
	dist_1_3 = distance(vertex_array[2],vertex_array[3])
	print (max(dist_2_4,dist_1_3))
	print (abs(dist_2_4-dist_1_3))
	if abs(dist_2_4-dist_1_3) > max(dist_2_4,dist_1_3)*0.03 : return False
	else : return True

def distance(point1, point2): #point간 거리구하는 함수
	dist= math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2)
	dist = math.sqrt(dist)

	return dist



def circle_coordinate(point) :
	length = []
	length.append(point[1][0] - point[0][0])
	length.append(point[1][1] - point[0][1])
	length.append(point[2][0] - point[3][0])
	length.append(point[2][1] - point[3][1])
	print (length)

	a = 2.1
	b = 3.9
	c = 2*a+3*b

	circle = np.zeros((16,2),dtype = int)
	for i in range(2) :
		for j in range(4):
			circle[(5-2*i)*j+3*i][0]=point[3*i][0]+length[2*i]*(a+j*b)/c
			circle[(5-2*i)*j+3*i][1]=point[3*i][1]+length[2*i+1]*(a+j*b)/c

	temp = np.zeros((2,1),dtype = int)

	l = 0
	for i in range(2):
		for j in [3,15,12,0]:			
			temp[i] = circle[j][i]-circle[l][i]
			circle[l+(j-l)/3][i]=circle[l][i]+temp[i]/3
			circle[l+(j-l)*2/3][i]=circle[l][i]+temp[i]*2/3
			l = j
	return (circle)
	

def vertex(point) : ## vertex points in square

	vertex_array = np.zeros((4,2), dtype =int) 
	vertex_array[0][0] = vertex_array[0][1] = 10000
		
	for dot in point:
		if dot[0][0]+dot[0][1]<vertex_array[0][0]+vertex_array[0][1]:
			vertex_array[0][0]=dot[0][0]
			vertex_array[0][1]=dot[0][1]
	for dot in point:
		if dot[0][0]+dot[0][1]>vertex_array[1][0]+vertex_array[1][1]:
			vertex_array[1][0]=dot[0][0]
			vertex_array[1][1]=dot[0][1]
	for dot in point:
		if dot[0][0]-dot[0][1]<vertex_array[2][0]-vertex_array[2][1]:
			vertex_array[2][0]=dot[0][0]
			vertex_array[2][1]=dot[0][1]
	for dot in point:
		if dot[0][0]-dot[0][1]>vertex_array[3][0]-vertex_array[3][1]:
			vertex_array[3][0]=dot[0][0]
			vertex_array[3][1]=dot[0][1]

	return (vertex_array)

def vertex_newPoint(point,vertex_array) : ## find new vertex points using regression.
	a = time.time()
	middle_array = []
	j =0
	for i in [2,1,3,0]:
		middle_array.append([(vertex_array[j][0]+vertex_array[i][0])/2,(vertex_array[j][1]+vertex_array[i][1])/2])
		j =i

	dist = distance(vertex_array[0], vertex_array[2])
	dist = dist/6

	middle_newArray =[[],[],[],[],[],[],[],[]] ## middle points using vertex points


	for dot in point :
		for i in range(4) :
			if middle_array[i][0]-dist<dot[0][0] and dot[0][0]<middle_array[i][0]+dist :
				if middle_array[i][1]-dist<dot[0][1] and dot[0][1]<middle_array[i][1]+dist :
					middle_newArray[2*i].append(dot[0][0])
					middle_newArray[2*i+1].append(dot[0][1])

	coef = [] ## regression coefficient alpha, beta, normalized
	
	for i in range(4) :
		coef.append(fit(middle_newArray[2*i], middle_newArray[2*i+1]))


	vertex_array[0][0] = (coef[3][1]-coef[0][1])/(coef[0][0]-coef[3][0])	
	vertex_array[0][1] = coef[3][0]*vertex_array[0][0]+coef[3][1]
	vertex_array[1][0] = (coef[1][1]-coef[2][1])/(coef[2][0]-coef[1][0])
	vertex_array[1][1] = coef[1][0]*vertex_array[1][0]+coef[1][1]
	vertex_array[2][0] = (coef[0][1]-coef[1][1])/(coef[1][0]-coef[0][0])
	vertex_array[2][1] = coef[0][0]*vertex_array[2][0]+coef[0][1]
	vertex_array[3][0] = (coef[2][1]-coef[3][1])/(coef[3][0]-coef[2][0])
	vertex_array[3][1] = coef[2][0]*vertex_array[3][0]+coef[2][1]


	b = time.time()
## using middle_array0,1,2,3 to find regression

	print (b-a)
	return(vertex_array)

def fit(X, Y): # calculate regression coefficient.

    def mean(Xs):
        return sum(Xs) / len(Xs)
    m_X = mean(X)
    m_Y = mean(Y)

    def std(Xs, m):
        normalizer = len(Xs) - 1
        return math.sqrt(sum((pow(x - m, 2) for x in Xs)) / normalizer)
    # assert np.round(Series(X).std(), 6) == np.round(std(X, m_X), 6)

    def pearson_r(Xs, Ys):

        sum_xy = 0
        sum_sq_v_x = 0
        sum_sq_v_y = 0

        for (x, y) in zip(Xs, Ys):
            var_x = x - m_X
            var_y = y - m_Y
            sum_xy += var_x * var_y
            sum_sq_v_x += pow(var_x, 2)
            sum_sq_v_y += pow(var_y, 2)
        return sum_xy / math.sqrt(sum_sq_v_x * sum_sq_v_y)
    # assert np.round(Series(X).corr(Series(Y)), 6) == np.round(pearson_r(X, Y), 6)

    r = pearson_r(X, Y)

    b = r * (std(Y, m_Y) / std(X, m_X))
    A = m_Y - b * m_X


    def line(x):
        return b * x + A
    return b, A


def readCVImage_Contours(input_image): ## find rectangular
	gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,150,255,0)

	for i in range(5) :	
		dilate = cv2.dilate(thresh, None)
		dilate_repeated = dilate
		(dilate, thresh)=(thresh, dilate)		

	for i in range(10) :	
		erode = cv2.erode(dilate_repeated, None)
		erode_repeated = erode
		(erode,dilate_repeated)=(dilate_repeated,erode)

	for i in range(5) :	
		dilate_final = cv2.dilate(erode_repeated, None)
		dilate_result = dilate_final
		(dilate_final,erode_repeated)=(erode_repeated, dilate_final)

	cv2.imwrite('sample1.jpg', dilate_result)

	im2, contours, hierarchy = cv2.findContours(dilate_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2]
##	print (len(contours[1]))
	drawed_img = cv2.drawContours(input_image, contours, 1, (0,255,0), 10)
	cv2.imwrite('sample2.jpg', drawed_img)

	return (contours[1])




def solution(): #결과값 계산 테스트용
   Cu = round(random.random()*3,1) #Cu 0~3 ppm
   Fe = round(random.random(),1) #Fe 0~1 ppm
   pH = round(random.random()*14,1) #pH 0~14
   Hy = round(random.random()*7,1) #Hy 0~7 ppm
   return(Cu,Fe,pH,Hy)

def HCA():
   GOF = round(random.random()*7) #임시로 랜덤 .. 미안타 내용넣어줄게 ..
   return(GOF)

if __name__ == '__main__':

	start_time = time.time()

	readCVImage(name, row, col)
	
	end_time = time.time()
	print ('Spended Time =', end_time - start_time)


