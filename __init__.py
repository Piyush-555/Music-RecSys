#importing dependencies
from hashlib import md5
from Models import *
from flask import Flask, render_template, request, flash, session
from time import sleep
from passlib.hash import sha256_crypt

app = Flask(__name__)
@app.route("/")
def hello():
    flash("hi there")
    return "Hello, Piyush, Shruti!!!!"

@app.route("/home/")
def homepage():
    return render_template("home.html")

@app.route("/signup/")
def signup():
    return render_template("signup.html")

@app.route("/signup/tnc")
def tnc():
    return "<h4>Do you really expect me to write Terms and Conditions !!!??</h4>"

@app.route("/signup/confirm/", methods=["POST"])
def signup_true():
    #connecting to database
    db.connect()
    try:
        User.create_table()
    except OperationalError:
        print("Table Already Exists")

    username=request.form.get("username")
    password=request.form.get("password")
    pass_hash=sha256_crypt.encrypt(password)
    email=request.form.get("email")
    user_id=md5(username.encode()).hexdigest()
    try:
        User.create(user_id=user_id,username=username,email=email,password=pass_hash)
    except IntegrityError:
        return render_template("username_exists.html")
    db.close()
    return render_template("signup_success.html")

@app.route("/login/")
def login():
    return render_template("login.html")

@app.route("/login/verify/", methods=['POST'])
def login_verify():
    username=request.form.get("username")
    attempted_password=request.form.get("password")
    db.connect()
    try:
        user=User.select().where(User.username==username).get()
    except:
        db.close()
        return render_template("message.html", message="Username Doesn't Exists")
    if sha256_crypt.verify(attempted_password, user.password):
        session['logged_in']=True
        session['username']=username
        db.close()
        return render_template("message.html", message="Logged in Successfully")
    else:
        db.close()
        return render_template("message.html", message="User Credentials Doesn't Match")

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run()
