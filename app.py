import joblib
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import SVM 
import pymongo
from flask import Flask, render_template, request, make_response
import json, os
import SVM
import pygmail
from flask import session

classifier = pickle.load(open('D:\Final Major Project\RiskAssess\Models\diabetes-prediction-rfc-model.pkl', 'rb'))
model = pickle.load(open('D:\Final Major Project\RiskAssess\Models\model.pkl', 'rb'))
model1 = pickle.load(open('D:\Final Major Project\RiskAssess\Models\model1.pkl', 'rb'))

# mongodb+srv://<username>:<password>@omnispectra.j3xzuo0.mongodb.net/?retryWrites=true&w=majority&appName=OmniSpectra

speech, name, age1, gender = "","","",""
problem, sym_duration = "",""
q1,q2,q3,q4,q5,q6 = 0,0,0,0,0,0
sym1,sym2,sym3 = "","",""
new_report, tips, soln = "","", ""
doctor, date, time, uemail = "","","",""

user_email = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
client = pymongo.MongoClient("mongodb+srv://admin:admin@riskassess.1npzwhc.mongodb.net/?retryWrites=true&w=majority&appName=RiskAssess")
db = client.get_database('Patients')
users_collection = db.users

class User(UserMixin):
    def __init__(self, user_id, username, email, password):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.password = password

    def get_id(self):
        return str(self.user_id)

    @staticmethod
    def find_user(username):
        return users_collection.find_one({'username': username})

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({'_id': user_id})
    if user_data:
        return User(user_data['_id'], user_data['username'], user_data['email'], user_data['password'])
    return None

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/help')
def help():
    return render_template("help.html")


@app.route('/terms')
def terms():
    return render_template("tc.html")

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     form = LoginForm()
#     if form.validate_on_submit():
#         user = User.find_user(form.username.data)
#         if user and check_password_hash(user['password'], form.password.data):
#             login_user(User(user['username'], user['email'], user['password']))
#             return redirect(url_for('dashboard'))
#     return render_template("login.html", form=form)
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        

        if username and password:
            user_data = User.find_user(username)
            if user_data and check_password_hash(user_data['password'], password):
                user = User(user_data['_id'], user_data['username'], user_data['email'], user_data['password'])
                login_user(user, remember=request.form.get('remember'))
                # Store user's email in global variable user_email
                global user_email
                user_email = user_data['email']
                return redirect(url_for('dashboard'))
            else:
                error_message = "Invalid username or password. Please try again."
                return render_template('login.html', form=form, error_message=error_message)
        else:
            error_message = "Please fill out both fields."
            return render_template('login.html', form=form, error_message=error_message)
    
    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        new_user = {
            'username': form.username.data,
            'email': form.email.data,
            'password': hashed_password
        }
        users_collection.insert_one(new_user)
        return redirect("/login")
    return render_template('signup.html', form=form)

@app.route('/Home')
def Home():
    return render_template('Home.html')

@app.route('/Home', methods=['POST'])
def Home_Value():
    global uemail
    uemail = request.form['uemail']
    return render_template('Main.html')

@app.route('/Form')
def Form():
    if uemail == "": return render_template('Home.html')
    else: return render_template('Form.html')

@app.route('/Chatbot')
def Chatbot():
    if uemail == "": return render_template('Home.html')
    else: return render_template('Chatbot.html')

@app.route("/dashboard")
# @login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/disindex")

def disindex():
    return render_template("disindex.html")


@app.route("/cancer")
# @login_required
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes")
# @login_required
def diabetes():
    return render_template("diabetes.html")


@app.route("/heart")
# @login_required
def heart():
    return render_template("heart.html")

@app.route("/liver")
# @login_required
def liver():
    return render_template("liver.html")

@app.route("/gendis")
# @login_required
def gendis():
    return render_template("gendis.html")

#Liver

def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load('D:\Final Major Project\RiskAssess\Models\liver_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/predictliver', methods=["POST"])
def predictliver():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePred(to_predict_list, 7)

    if int(result) == 1:
        prediction = "Patient has a high risk of Liver Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Liver Disease"
    return render_template("liver_result.html", prediction_text=prediction)


@app.route('/logout')
# @login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


# Breast-Cancer

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    if output == 4:
        res_val = "a high risk of Breast Cancer"
    else:
        res_val = "a low risk of Breast Cancer"

    return render_template('cancer_result.html', prediction_text='Patient has {}'.format(res_val))


# Diabetes



@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('diab_result.html', prediction=my_prediction)

    
    # Heart

@app.route('/predictheart', methods=['POST'])
def predictheart():

    age = request.form['age']
    sex = request.form['sex']
       
    if sex == 'Male': sex = 1
    elif sex == 'Female': sex = 0
    
    cp = request.form['cp']
    if cp == 'Typical Angina': cp = 0
    elif cp == 'Atypical Angina': cp = 1
    elif cp == 'Non-anginal Pain': cp = 2
    elif cp == 'Asymptomatic': cp = 3
    
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    
    fbs = request.form['fbs']
    if fbs == 'Yes': fbs = 1
    elif fbs == 'No': fbs = 0
    
    restecg = request.form['restecg']
    if restecg == 'Normal': restecg = 0
    elif restecg == 'Having ST-T Wave Abnormality': restecg = 1
    elif restecg == 'Left Ventricular Hyperthrophy': restecg = 2
    
    thalach = request.form['thalach']
    
    exang = request.form['exang']
    if exang == 'Yes': exang = 1
    elif exang == 'No': exang = 0
    
    oldpeak = request.form['oldpeak']
    
    slope = request.form['slope']
    if slope == 'Upsloping': slope = 0
    elif slope == 'Flat': slope = 1
    elif slope == 'Downsloping': slope = 2
    
    ca = request.form['ca']
    
    thal = request.form['thal']
    if thal == 'Normal': thal = 1
    elif thal == 'Fixed Defect': thal = 2
    elif thal == 'Reversible Defect': thal = 3
    
    #print(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    try:
        op = SVM.svm_pred(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    except Exception as e:
        print(type(e).__name__)
    global speech
    if op == 0: 
        opstr = "No Heart Disease"
        speech = "Report Looks Fine."
    if op == 1: 
        opstr = "Heart Disease Present"
        speech = " may be suffering from a Heart Disease/problem!"
    # return render_template('heart_result.html', n=op, s=opstr)
    return render_template('heart_result.html', prediction_text='Patient  {}'.format(speech))


############################################################################################################

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    #print("Request:\n", json.dumps(req, indent=4))
    
    res = makeWebhookResult(req)
    res = json.dumps(res, indent=4)
    #print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

def makeWebhookResult(req):
    # Get all the Query Parameter
    query_response = req["queryResult"]
    #print(query_response)
    # text = query_response.get('queryText', None)
    # parameters = query_response.get('parameters', None)
    
    res = {  "fulfillmentText": "", }
    global name
    global gender
    global age1
    global sym1
    global sym2
    global new_report
    
    # Patient_Name
    if query_response.get("action") == "user_name":
        r = query_response.get("parameters")
        r1 = r.get("given-name")
        global name
        name = r1
        #print("Patient Name:",name)
        
    if query_response.get("action") == "user_age":
        r = query_response.get("parameters")
        r2 = r.get("age")
        r1 = r2.get("amount")
        global age1
        age1 = int(r1)
        #print("Patient age:",age1 )
    
    # Checkup_Patient_gender
    if query_response.get("action") == "DefaultWelcomeIntent.DefaultWelcomeIntent-custom.Checkup_Patient-custom":
        r = query_response.get("parameters")
        r1 = r.get("Gender")
        global gender
        gender = r1
        #print("Patient gender:",gender)
        a1 = "OK "+ name + "(" + str(age1) + "), Please fill and submit the report form on the right side...."
        res = {  "fulfillmentText": a1, }
    
    # Checkup_Patient_filling
    if query_response.get("action") == "DefaultWelcomeIntent.DefaultWelcomeIntent-custom.Checkup_Patient-custom.Checkup_Patient_gender-custom":
        a2 = "Thanks " + name + ", after analysing the information you have given us, The System Predicts that "+ speech +" Please note, this is not a diagnosis. Always visit a doctor if you are in doubt, or if your symptoms get worse or don't improve. If your situation is serious, always call the emergency services. Do you want to book an appointment with a doctor?"
        res = {  "fulfillmentText": a2, }
        
    ############ Suffering Patient ##########################
    
    # Suffering_Patient
    if query_response.get("action") == "DefaultWelcomeIntent.DefaultWelcomeIntent-custom":
        # Main Problem
        global problem
        global soln
        r = query_response.get("parameters")
        problem = r.get("Symptoms")
        if problem == "Chest Pain": soln = "A pain reliever, such as aspirin, can help alleviate the heart/chest pain associated with less severe cases. When heart pain strikes, lying down immediately with the head elevated above the body may bring some relief. A slightly upright position helps when the pain is due to reflux."
        elif problem == "High Blood Pressure": soln = "Blood pressure often increases as weight (Obesity) increases. Regular physical activity such as 150 minutes a week can lower your blood pressure by about 5 to 8 mm Hg if you have high blood pressure. Eating a diet that is rich in whole grains, fruits, vegetables and low-fat dairy products and skimps on saturated fat and cholesterol can lower your blood pressure by up to 11 mm Hg if you have high blood pressure. Chronic stress and also smoking may contribute to high blood pressure, so avoid that."
        elif problem in("breathing problems", "breathlessness"): soln = "Breathing-in deeply through the abdomen and also ursed-lip breathing can help to manage your breathlessness. Finding a comfortable and supported position to stand or lie in can help to relax and catch your breath. Inhaling steam can help to keep nasal passages clear, which can help to breathe more easily. Drinking black coffee may help to treat breathlessness, reducing tiredness in the airway muscles. Being overweight also can cause disrupted breathing while you sleep (sleep apnea)."
        elif problem in("sleeping", "sleep"): soln = "Set yourself up for restful sleep: Stick to a regular sleep/wake schedule. Turn off the TV, computer, and other devices before bedtime. Keep your bedroom cool and dark. Avoid alcohol before bedtime and caffeine in the afternoon or evening. Exercise every morning."
        
    # Suffering_Patient_symp_dur
    if query_response.get("action") == "DefaultWelcomeIntent.DefaultWelcomeIntent-custom.Suffering_Patient-custom":
        # Duration
        global sym_duration
        sym_duration = query_response.get("queryText")
    
    # Suffering_Patient_Q2
    if query_response.get("action") == "DefaultWelcomeIntent.DefaultWelcomeIntent-custom.Suffering_Patient-custom.Suffering_Patient_symp_dur-custom":
        # Q. Heart Disease
        r = query_response.get("parameters")
        r1 = r.get("Confirmation")
        global q1
        if r1 == 'Yes': q1 = 1
        
    # Suffering_Patient_Q3
    if query_response.get("action") == "DefaultWelcomeIntent.DefaultWelcomeIntent-custom.Suffering_Patient-custom.Suffering_Patient_symp_dur-custom.Suffering_Patient_Q2-custom":
        # Q. Diabetes
        r = query_response.get("parameters")
        r1 = r.get("Confirmation")
        global q2
        if r1 == 'Yes': q2 = 1
        
    # Suffering_Patient_Q4
    if query_response.get("action") == "DefaultWelcomeIntent.DefaultWelcomeIntent-custom.Suffering_Patient-custom.Suffering_Patient_symp_dur-custom.Suffering_Patient_Q2-custom.Suffering_Patient_Q3-custom":
        # Q. High Blood Pressure
        r = query_response.get("parameters")
        r1 = r.get("Confirmation")
        global q3
        if r1 == 'Yes': q3 = 1
        
    # Suffering_Patient_Q5
    if query_response.get("action") == "DefaultWelcomeIntent.DefaultWelcomeIntent-custom.Suffering_Patient-custom.Suffering_Patient_symp_dur-custom.Suffering_Patient_Q2-custom.Suffering_Patient_Q3-custom.Suffering_Patient_Q4-custom":
        # Q. Chronic Obstructive Lung Disease/Asthma
        r = query_response.get("parameters")
        r1 = r.get("Confirmation")
        global q4
        if r1 == 'Yes': q4 = 1
        
    # Suffering_Patient_Q6
    if query_response.get("action") == "DefaultWelcomeIntent.DefaultWelcomeIntent-custom.Suffering_Patient-custom.Suffering_Patient_symp_dur-custom.Suffering_Patient_Q2-custom.Suffering_Patient_Q3-custom.Suffering_Patient_Q4-custom.Suffering_Patient_Q5-custom":
        # Q. Smoking
        r = query_response.get("parameters")
        r1 = r.get("Confirmation")
        global q5
        if r1 == 'Yes': q5 = 1
        
    # Suffering_Patient_sym1
    if query_response.get("action") == "DefaultWelcomeIntent.DefaultWelcomeIntent-custom.Suffering_Patient-custom.Suffering_Patient_symp_dur-custom.Suffering_Patient_Q2-custom.Suffering_Patient_Q3-custom.Suffering_Patient_Q4-custom.Suffering_Patient_Q5-custom.Suffering_Patient_Q6-custom":
        # Q.  brain stroke/ Overweight/Obese / Kidney Disease
        r = query_response.get("parameters")
        r1 = r.get("Confirmation")
        global q6
        if r1 == 'Yes': q6 = 1
        
    # Suffering_Patient_sym2
    if query_response.get("action") == "symp1":
        ''' Q. - Chest Pain worse on breathing in.
            - Burning pain in chest/upper abdomen.
            - Chest pain caused by pressing the chest.
            - Chest pain worse on movement.  - None '''
        global sym1
        sym1 = query_response.get("queryText")
    
    # Suffering_Patient_sym3
    if query_response.get("action") == "Suffering_Patient_sym2.Suffering_Patient_sym2-custom":
        ''' Q. - Chest pain while resting.
            - Sudden chest pain.
            - Shortness of Breadth.
            - Fast or Shallow breathing. - None ''' 
        global sym2
        sym2 = query_response.get("queryText") 
        
    # Suffering_Patient_sym_final
    if query_response.get("action") == "Suffering_Patient_sym2.Suffering_Patient_sym2-custom.Suffering_Patient_sym3-custom":
        ''' Q. - Feeling your heart racing or skipping a beat.
            - Chest pain spreading to the left arm.
            - Chest pain spreading on physical effort.
            - Joint/Abdominal pain. - None
            
            - Chest Tightness.
            - Unusually Tired.
            - Anxiety .
            - Chest pain spreading the Jaw. - None '''
        global sym3
        sym3 = query_response.get("queryText")
        
        ###### Create Report ############
        new_report = ""
        new_report += "To Summerize: You had "+problem+" for "+sym_duration+" duration."
        if q1 == 1: new_report += "You had Heart Disease before, "
        if q2 == 1: new_report += "You have/had Diabetes, "
        if q3 == 1: new_report += "You have/had High Blood Pressure, "
        if q4 == 1: new_report += "You've suffered from Asthma OR Chronic Obstructive lung disease, "
        if q5 == 1: new_report += "You're Smoking (or smoked before), "
        if q6 == 1: new_report += "You have/had: Brain Stroke OR Kidney Disease OR Obesity Problem, "
        if sym1 not in("none","None") or sym2 not in("none","None") or sym3 not in("none","None"): new_report += "and also have symptoms like "
        if sym1 not in("none","None"): new_report += sym1+", "
        if sym2 not in("none","None"): new_report += sym2+", "
        if sym3 not in("none","None"): new_report += sym3
        if (q1,q2,q3,q4,q5,q6) == 0: new_report += "You did'nt have any of the above mentioned symptoms!"
    
    # Suffering_Patient_sym_report_filling
    if query_response.get("action") == "Suffering_Patient_sym2.Suffering_Patient_sym2-custom.Suffering_Patient_sym3-custom.Suffering_Patient_sym_final-custom.Suffering_Patient_sym_report_yes-custom":
        r = query_response.get("parameters")
        name = r.get("given-name")
        gender = r.get("Gender")
        r2 = r.get("age")
        r1 = r2.get("amount")
        age1 = int(r1)
        #print("Patient:", name, age1, gender )
        
    # Suffering_Patient_sym_report_results
    if query_response.get("action") == "Suffering_Patient_sym2.Suffering_Patient_sym2-custom.Suffering_Patient_sym3-custom.Suffering_Patient_sym_final-custom.Suffering_Patient_sym_report_yes-custom.Suffering_Patient_sym_report_filling-custom":
        ans2 = "Thanks " + name + ", after analysing the information you have given us, The System Predicts that "+ speech +" "+ new_report + ". Some ways how you can avoid this problem are: " + soln + "--> Please note, this is not a diagnosis. Always visit a doctor if you are in doubt, or if your symptoms get worse or don't improve. If your situation is serious, always call the emergency services. Do you want to book an appointment with a doctor?"
        res = {  "fulfillmentText": ans2, }
    
    # Suffering_Patient_sym_report_no
    if query_response.get("action") == "Suffering_Patient_sym2.Suffering_Patient_sym2-custom.Suffering_Patient_sym3-custom.Suffering_Patient_sym_final-custom":
        r = "OK, "+ new_report + ". Some ways how you can avoid this problem are: " + soln + "--> Please note, this is not a diagnosis. Always visit a doctor if you are in doubt, or if your symptoms get worse or don't improve. If your situation is serious, always call the emergency services. Do you want to book an appointment with a doctor?"
        res = {  "fulfillmentText": r, }
        
    ######## DOCTOR SECTION ##########
    
    # app_date_time
    if query_response.get("action") == "doctors_list.doctors_list-custom":
        r1 = query_response.get("queryText")
        global doctor
        doctor = r1
        #print("----Doctor: ", doctor)
    
    # app_booked
    if query_response.get("action") == "doctors_list.doctors_list-custom.app_date_time-custom":
        r = query_response.get("parameters")
        r1 = r.get("date")
        r2 = r.get("time")
        global date
        global time
        date = r1
        time = r2
        print("Date: ", date, "   Time: ", time, "Email: ", user_email)
        pygmail.sendEmail( user_email, doctor, date, time, name, new_report )
        
    return res
   
############################################################################################################

if __name__ == "__main__":
    app.run(debug=True)

