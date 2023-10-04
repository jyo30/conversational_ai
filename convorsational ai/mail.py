from flask import Flask,render_template,request,jsonify
from flask_cors import CORS
from chat import get_response
import smtplib

app = Flask(__name__)
CORS(app)

@app.route("/",methods=["GET"])
def index_get():
    return render_template("base.html")
 
@app.post("/predict")
def predict():
    print("please enter your email")
    gmail_user = 'jyothsna.shellapally@realvariable.com'
    gmail_password = 'Rashijyo@7971'
    sent_from = gmail_user
            #to = ['revallymanishriya@gmail.com', 'arjunmurthy1045@gmail.com','nagarjuna.learning@gmail.com']
    to=input("you:")
    subject = 'mail test'
    body = 'hi from jyo'
    email_text = """\
    From: %s
    To: %s
    Subject: %s
    %s
    """ % (sent_from, ", ".join(to), subject, body)
    try:
        print("hi")
        smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        smtp_server.ehlo()
        smtp_server.login(gmail_user, gmail_password)
        smtp_server.sendmail(sent_from, to, email_text)
        smtp_server.close()
        print ("deatils sent to your mail successfully!")
    except Exception as ex:
        print ("Something went wrong….",ex)   
if __name__=="__main__":
    app.run(debug=True)
