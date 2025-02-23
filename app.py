from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash
from flask_socketio import SocketIO, join_room, emit
import os
from dotenv import load_dotenv
from chatbot import query_knowledge_base
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from werkzeug.utils import secure_filename
from PIL import Image

load_dotenv()

app = Flask(_name_, template_folder=r"D:\Diversion_2k25\eye_analysis\templates", static_folder=r"D:\Diversion_2k25\eye_analysis\static")
app.secret_key = "f1a4fd45237702d7e84bcae2594f6a4d"
socketio = SocketIO(app)

# Store active rooms
rooms = {}

# Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Database Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)

# Create the database
with app.app_context():
    db.create_all()

# Load PyTorch models
class EyeDiseaseModel(nn.Module):
    def _init_(self, num_classes=2):
        super(EyeDiseaseModel, self)._init_()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 112 * 112, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Ensure device compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inner_eye_model = EyeDiseaseModel(num_classes=2).to(device)
outer_eye_model = EyeDiseaseModel(num_classes=2).to(device)

# Load models safely
checkpoint_inner = torch.load(r"D:\Diversion_2k25\eye_analysis\ml_train\inner_eyes_model.pth", map_location=device)
inner_eye_model.load_state_dict({k: v for k, v in checkpoint_inner.items() if k in inner_eye_model.state_dict()}, strict=False)
inner_eye_model.eval()

checkpoint_outer = torch.load(r"D:\Diversion_2k25\eye_analysis\ml_train\efficientnet_b0.pth", map_location=device)
outer_eye_model.load_state_dict({k: v for k, v in checkpoint_outer.items() if k in outer_eye_model.state_dict()}, strict=False)
outer_eye_model.eval()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure upload directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        age = int(request.form.get('age', 0))  # Ensure correct type
        gender = request.form.get('gender')
        
        if User.query.filter_by(email=email).first():
            return "Email already registered!!"
        
        new_user = User(name=name, email=email, age=age, gender=gender)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("store"))
    return render_template("register.html")

@app.route("/store")
def store():
    users = User.query.all()
    return render_template("store.html", users=users)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('upload'))
        else:
            flash("Invalid username or password. Try again!", "error")
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        inner_eye_img = request.files.get('inner_eye')
        outer_eye_img = request.files.get('outer_eye')
        predictions = {}
        
        if inner_eye_img and allowed_file(inner_eye_img.filename):
            inner_eye_path = os.path.join(UPLOAD_FOLDER, secure_filename(inner_eye_img.filename))
            inner_eye_img.save(inner_eye_path)
            img_tensor = preprocess_image(inner_eye_path)
            with torch.no_grad():
                inner_pred = torch.softmax(inner_eye_model(img_tensor), dim=1)
                predictions["inner_eye_disease"] = f"Inner Eye Disease: {torch.argmax(inner_pred).item()}"
        
        if outer_eye_img and allowed_file(outer_eye_img.filename):
            outer_eye_path = os.path.join(UPLOAD_FOLDER, secure_filename(outer_eye_img.filename))
            outer_eye_img.save(outer_eye_path)
            img_tensor = preprocess_image(outer_eye_path)
            with torch.no_grad():
                outer_pred = torch.softmax(outer_eye_model(img_tensor), dim=1)
                predictions["outer_eye_disease"] = f"Outer Eye Disease: {torch.argmax(outer_pred).item()}"
        
        return render_template("result.html", predictions=predictions)
    
    return render_template('upload.html')



# Route for Home (Unified Page for Both Chatbot & Video Chat)
@app.route("/chatbot_video")
def chatbot_video():
    return render_template("extra_page.html")  # Unified template for chatbot & video chat

# Chatbot API Route
@app.route("/chatbot/query", methods=["POST"])
def chatbot_query():
    user_input = request.json.get("message", "")
    response = query_knowledge_base(user_input)
    return jsonify({"response": response})

# Video Chat Socket Events
@socketio.on("join")
def handle_join(data):
    room = data["room"]
    username = data["username"]
    join_room(room)

    if room not in rooms:
        rooms[room] = []
    if username not in rooms[room]:
        rooms[room].append(username)

    emit("user_joined", {"username": username}, room=room)

@socketio.on("message")
def handle_message(data):
    room = data["room"]
    emit("message", data, room=room, include_self=False)

@socketio.on("offer")
def handle_offer(data):
    room = data["room"]
    emit("offer", data, room=room, include_self=False)

@socketio.on("answer")
def handle_answer(data):
    room = data["room"]
    emit("answer", data, room=room, include_self=False)

@socketio.on("ice-candidate")
def handle_ice_candidate(data):
    room = data["room"]
    emit("ice-candidate", data, room=room, include_self=False)
    
@app.route('/thank_you', endpoint='thank_you')
def thank_you():
    return render_template('thank_you.html')

@app.route('/extra_page', endpoint='extra_page')
def thank_you():
    return render_template('extra_page.html')



if _name_ == "_main_":
    print("Starting Netra Sanchalaya...")
    os.environ["WERKZEUG_SERVER_FD"] = "0"  # Set a default value
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
