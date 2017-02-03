import smtplib

def sendEmailUser(ID, PW, Email):
	alpha = ID 
	beta = PW

	FROM = "admin@biosensorlab.com"
	TO = [Email]

	SUBJECT = "your ID &PW"

	message = """\
	From: %s
	To: %s
	Subject: %s

	ID : %s
	PW : %s
	""" % (FROM, ", ".join(TO), SUBJECT, alpha, beta)

	server = smtplib.SMTP('smtp.gmail.com',587)
	server.starttls()
	server.login("admin@biosensorlab.com","biolab??3")
	server.sendmail(FROM, TO, message)
	server.quit()
	return(1)
