import mysql.connector

def emailCheck(email):
	cnx =mysql.connector.connect(user = 'root', password = 'biolab', host='127.0.0.1',database='colorsensor')
	cursor = cnx.cursor()
	
	cursor.execute("SELECT * FROM userInfo WHERE email = '%s'" %email)
	ID = "0"
	PW = "0"
	for row in cursor:
		ID = row[1]
		PW = row[2]

	cnx.commit()
	cursor.close()
	cnx.close()
	return(ID, PW)

def loginSelect(ID,PW):
	cnx =mysql.connector.connect(user = 'root', password = 'biolab', host='127.0.0.1',database='colorsensor')
	cursor = cnx.cursor()
	
	cursor.execute("SELECT * FROM userInfo WHERE userID = '%s' AND userPW ='%s'" %(ID, PW))
	i = -1
	for row in cursor:
		i = row[0]
	cnx.commit()
	cursor.close()
	cnx.close()
	return(i)

def registerCheck(ID, email):
	cnx =mysql.connector.connect(user = 'root', password = 'biolab', host='127.0.0.1',database='colorsensor')
	cursor = cnx.cursor()
	
	cursor.execute("SELECT * FROM userInfo WHERE userID = '%s'" %ID)
	i = True
	for row in cursor:
		i = False
	if (i==False):
		cursor.execute("SELECT * FROM userInfo WHERE email = '%s'" %email)
		i = True
		for row in cursor:
			i = False

	cnx.commit()
	cursor.close()
	cnx.close()
	return(i)

def registerInsert(ID, PW, email, address):
	cnx =mysql.connector.connect(user = 'root', password = 'biolab', host='127.0.0.1',database='colorsensor')
	cursor = cnx.cursor()
	cursor.execute("SELECT * FROM userInfo")

	i =1
	for row in cursor:
		i+= 1	
## did not work. it couldn't store the data at MySql
	add = ("INSERT INTO userInfo"
		"(no, userID, userPW, addr, email)" 
		"VALUES(%s, %s, %s, %s, %s)")
	data = (i, ID, PW, address, email)
	cursor.execute(add,data)

	cnx.commit()
	cursor.close()
	cnx.close()
	return(i)

