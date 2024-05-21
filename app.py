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


classifier = pickle.load(open('D:\MajorProject\MajorProject\RiskAssess\Models\diabetes-prediction-rfc-model.pkl', 'rb'))
model = pickle.load(open('D:\MajorProject\MajorProject\RiskAssess\Models\model.pkl', 'rb'))
model1 = pickle.load(open('D:\MajorProject\MajorProject\RiskAssess\Models\model1.pkl', 'rb'))
health = pickle.load(open('D:\MajorProject\MajorProject\RiskAssess\Models\health_model.pkl', 'rb'))  

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


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

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Redirect to the dashboard without any validation
        return redirect(url_for('dashboard'))
    # Render the login template regardless of form submission
    return render_template("dashboard.html", form=form)




@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

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


@app.route("/gendis")
# @login_required
def gendis():
    return render_template("gendis.html")


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





#Liver

def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load('D:\MajorProject\MajorProject\RiskAssess\Models\liver_model.pkl')
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

#General Health Diseases function    

data = pd.read_csv('D:\MajorProject\MajorProject\RiskAssess\Dataset\Training.csv')
symptoms = data.columns[:-1].tolist()  # Exclude the last column 'prognosis'


@app.route('/')
def home():
    return render_template('gendis.html', symptoms=symptoms)


@app.route('/predicthealth', methods=['POST'])
def predicthealth():
    if request.method == 'POST':
        selected_symptoms = [int(request.form[f'symptom{i+1}']) for i in range(6)]
        input_data = np.array(selected_symptoms).reshape(1, -1)
        prediction = health.predict(input_data)
        
        result = 'Disease Detected' if prediction[0] == 1 else 'No Disease Detected'
        return render_template('gendisresult.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)


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

   
############################################################################################################

if __name__ == "__main__":
    app.run(debug=True)

