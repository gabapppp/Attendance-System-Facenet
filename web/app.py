from flask import Flask, render_template, Response, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
from datetime import datetime
# from werkzeug.utils import secure_filename
import glob
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization
from torchvision import transforms
import numpy as np
from PIL import Image
import time


UPLOAD_FOLDER = './data/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def trans(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)


app = Flask(__name__)
app.config['SECRET_KEY'] = '44~L|hf7M8p1q1f'
# config database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'

# config migrate
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# load mtcnn and facenet
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    email = db.Column(db.String(128), nullable=False, unique=True)


class Tracking(db.Model):
    no = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), db.ForeignKey(User.name), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey(User.id), nullable=False)
    time = db.Column(db.DATETIME(timezone=False),
                     default=datetime.utcnow)


def inference(model, face, local_embeds, threshold=0.6):
    power = pow(10, 6)
    # local: [n,512] voi n la so nguoi trong faceslist
    embeds = []
    # print(trans(face).unsqueeze(0).shape)
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds)  # [1,512]
    # print(detect_embeds.shape)
    # [1,512,1]                                      [1,512,n]
    norm_diff = detect_embeds.unsqueeze(-1) - \
        torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    # print(norm_diff)
    # (1,n), moi cot la tong khoang cach euclide so vs embed moi
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1)

    min_dist, embed_idx = torch.min(norm_score, dim=1)
    # print(min_dist*power, names[embed_idx])
    # print(min_dist.shape)
    if min_dist*power > threshold:
        return -1, -1
    else:
        return embed_idx, min_dist.double()


def extract_face(box, img, margin=20):
    face_size = 160
    img_size = (640, 480)
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ]  # tạo margin bao quanh box cũ
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img, (face_size, face_size),
                      interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face


def addAttendance(name):
    split = name[0].split('-')
    id = split[0]
    name = split[1]
    tracking = Tracking(user_id=id, name=name)
    with app.app_context():
        db.session.add(tracking)
        db.session.commit()


@app.route('/')
def index():
    update_face_check = False
    if os.path.isfile('./data/faceslist.pth') and os.path.isfile('./data/usernames.npy'):
        update_face_check = True
    return render_template('index.html', update_face_check=update_face_check)


def gen_frames():
    prev_frame_time = 0
    new_frame_time = 0
    model = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)
    model.eval()
    mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)
    embeddings = []
    if device == 'cpu':
        embeddings = torch.load(UPLOAD_FOLDER+'faceslistCPU.pth')
    else:
        embeddings = torch.load(UPLOAD_FOLDER+'faceslist.pth')
    names = np.load(UPLOAD_FOLDER+'usernames.npy')
    camera = cv2.VideoCapture(0)
    count = 20
    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int, box.tolist()))
                    face = extract_face(bbox, frame)
                    idx, score = inference(model, face, embeddings)
                    if idx != -1:
                        frame = cv2.rectangle(
                            frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
                        score = torch.Tensor.cpu(
                            score[0]).detach().numpy()*pow(10, 6)
                        frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(
                            score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_8)
                        if count == 0:
                            addAttendance(names)
                            count = 20
                        count -= 1
                    else:
                        frame = cv2.rectangle(
                            frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
                        frame = cv2.putText(
                            frame, 'Unknown', (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_8)

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            cv2.putText(frame, fps, (4, 25), cv2.FONT_HERSHEY_DUPLEX,
                        1, (100, 255, 0), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    pass


def getFace(name):
    mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True,
                  post_process=False, device=device)
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    count = 50
    leap = 1
    while camera.isOpened() and count:
        ret, frame = camera.read()

        if mtcnn(frame) is not None and leap % 2:
            USR_PATH = os.path.join(UPLOAD_FOLDER + "test_images/", name)
            path = str(USR_PATH+'/{}.jpg'.format(str(datetime.now())
                                                 [:-7].replace(":", "-").replace(" ", "-")+str(count)))
            face_img = mtcnn(frame, save_path=path)
            count -= 1
        leap += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()
    doneImg = cv2.imread("./static/black_img.png")
    ret, buffer = cv2.imencode('.jpg', doneImg)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    pass


@app.route('/get_face/<uid>')
def get_face(uid):
    return Response(getFace(uid),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/setFace/<int:id>')
def setFace(id):
    user = db.get_or_404(User, id)
    uid = str(user.id)+'-'+user.name
    # flash('Set Face For user: %s success' % user.name)
    return render_template('setface.html', uid=uid)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/admin')
def admin():
    user_list = User.query.all()
    return render_template("admin.html", user_list=user_list)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/addUser', methods=['POST'])
def addUser():
    if request.method == 'POST':
        user = User(
            name=request.form["name"],
            email=request.form["email"],
        )
        """
        if 'file' in request.files:
            file = request.files['file']

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                img = cv2.imdecode(np.fromstring(
                    file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                # Convert into grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if (faces is None):
                    flash("There is no face in the picture")
                else:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h),
                                      (0, 0, 255), 2)
                        faces = img[y:y + h, x:x + w]
                        cv2.imwrite(+
                                    request.form["name"] + '.jpg', faces)
        """
        db.session.add(user)
        db.session.commit()
        flash('Successfully')
        return redirect(url_for('admin'))
    flash('Got some error, please try again')
    return redirect(url_for('admin'))


@app.route("/user/<int:id>/delete", methods=["POST"])
def user_delete(id):
    user = db.get_or_404(User, id)

    if request.method == "POST":
        db.session.delete(user)
        db.session.commit()
        flash("Delete user %s successfully" % user.name)
        return Response("Successfully deleted")
    return Response("Bad Request")


@app.route("/user", methods=["POST"])
def editUser():
    return Response("Ok")


@app.route('/admin/tracking')
def tracking():
    trackings = Tracking.query.all()
    return render_template("tracking.html", trackings=trackings)


@app.route('/tracking/<int:no>/delete', methods=["POST"])
def tracking_delete(no):
    tracking = db.get_or_404(Tracking, no)
    if request.method == "POST":
        db.session.delete(tracking)
        db.session.commit()
        flash("Delete tracking successfully")
        return redirect(url_for("tracking"))
    return Response("Error")


@app.route('/updateface')
def update_face():
    IMG_PATH = UPLOAD_FOLDER + "test_images"
    model = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)

    model.eval()

    embeddings = []
    names = []

    for usr in os.listdir(IMG_PATH):
        embeds = []
        for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
            # print(usr)
            try:
                img = Image.open(file)
            except:
                continue
            with torch.no_grad():
                # print('smt')
                # 1 anh, kich thuoc [1,512]
                embeds.append(model(trans(img).to(device).unsqueeze(0)))
        if len(embeds) == 0:
            continue
        # dua ra trung binh cua 30 anh, kich thuoc [1,512]
        embedding = torch.cat(embeds).mean(0, keepdim=True)
        embeddings.append(embedding)  # 1 cai list n cai [1,512]
        # print(embedding)
        names.append(usr)

    embeddings = torch.cat(embeddings)  # [n,512]
    names = np.array(names)

    if device == 'cpu':
        torch.save(embeddings, UPLOAD_FOLDER+"/faceslistCPU.pth")
    else:
        torch.save(embeddings, UPLOAD_FOLDER+"/faceslist.pth")
    np.save(UPLOAD_FOLDER+"/usernames", names)
    flash('Update Completed! There are {0} people in FaceLists'.format(
        names.shape[0]))
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
