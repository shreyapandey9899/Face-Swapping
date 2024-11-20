from flask import Flask, render_template, request, send_file, redirect, url_for
import os
import cv2
import dlib
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Limit size to 16MB
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Configure directories for uploads and results
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload and result folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\shrey\\OneDrive\\Documents\\GitHub\\CVDL\\server\\shape_predictor_68_face_landmarks (1).dat")

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Save uploaded file to the appropriate directory
def save_file(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        return file_path
    return None

# Detect faces in an image using dlib's face detector
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces

# Extract facial landmarks from an image using dlib's shape predictor
def extract_landmarks(image, face):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(gray, face)
    return np.array([(p.x, p.y) for p in landmarks.parts()])

# Adjust brightness and saturation of source image to match target image
def adjust_brightness_and_saturation(source, target):
    source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    source_mean_v = np.mean(source_hsv[:, :, 2])
    target_mean_v = np.mean(target_hsv[:, :, 2])
    brightness_ratio = target_mean_v / source_mean_v
    source_hsv[:, :, 2] = np.clip(source_hsv[:, :, 2] * brightness_ratio, 0, 255)

    source_mean_s = np.mean(source_hsv[:, :, 1])
    target_mean_s = np.mean(target_hsv[:, :, 1])
    saturation_ratio = target_mean_s / source_mean_s
    source_hsv[:, :, 1] = np.clip(source_hsv[:, :, 1] * saturation_ratio, 0, 255)

    adjusted_source = cv2.cvtColor(source_hsv, cv2.COLOR_HSV2BGR)
    return adjusted_source

# Warp triangle function to apply the face swap
def warp_triangle(img1, img2, t1, t2):
    rect1 = cv2.boundingRect(np.float32([t1]))
    rect2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - rect1[0]), (t1[i][1] - rect1[1])))
        t2_rect.append(((t2[i][0] - rect2[0]), (t2[i][1] - rect2[1])))
        t2_rect_int.append(((t2[i][0] - rect2[0]), (t2[i][1] - rect2[1])))

    mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    size = (rect2[2], rect2[3])
    img2_rect = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped_img = cv2.warpAffine(img1_rect, img2_rect, size, None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    img2_rect = img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]
    img2_rect = img2_rect * (1 - mask)
    warped_img = warped_img * mask

    img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]] = img2_rect + warped_img

# Delaunay triangulation to split the face into triangles
def delaunay_triangulation(landmarks):
    landmarks = np.array(landmarks, dtype=np.float32)
    subdiv = cv2.Subdiv2D((0, 0, 500, 500))

    for point in landmarks:
        subdiv.insert((point[0], point[1]))

    triangles = subdiv.getTriangleList()

    triangle_indices = []
    for t in triangles:
        pt1 = np.array([t[0], t[1]])
        pt2 = np.array([t[2], t[3]])
        pt3 = np.array([t[4], t[5]])

        idx1 = np.argmin(np.linalg.norm(landmarks - pt1, axis=1))
        idx2 = np.argmin(np.linalg.norm(landmarks - pt2, axis=1))
        idx3 = np.argmin(np.linalg.norm(landmarks - pt3, axis=1))

        triangle_indices.append([idx1, idx2, idx3])

    return triangle_indices

# Swap faces based on the Delaunay triangulation (Updated)
def warp_face_delaunay(source_image, target_image, source_landmarks, target_landmarks):
    source_landmarks = np.array(source_landmarks, dtype=np.float32)
    target_landmarks = np.array(target_landmarks, dtype=np.float32)

    triangles = delaunay_triangulation(target_landmarks)

    for triangle in triangles:
        t1 = [source_landmarks[triangle[0]], source_landmarks[triangle[1]], source_landmarks[triangle[2]]]
        t2 = [target_landmarks[triangle[0]], target_landmarks[triangle[1]], target_landmarks[triangle[2]]]

        warp_triangle(source_image, target_image, t1, t2)

    return target_image

# Process the uploaded images and perform the face swap
def process_images(source_path, target_path):
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    source_image = adjust_brightness_and_saturation(source_image, target_image)

    source_faces = detect_faces(source_image)
    target_faces = detect_faces(target_image)

    if len(source_faces) == 0 or len(target_faces) == 0:
        raise ValueError("No faces detected in one or both images")

    source_landmarks = extract_landmarks(source_image, source_faces[0])
    target_landmarks = extract_landmarks(target_image, target_faces[0])

    output_image = warp_face_delaunay(source_image, target_image.copy(), source_landmarks, target_landmarks)

    result_path = os.path.join(RESULT_FOLDER, 'output.jpg')
    cv2.imwrite(result_path, output_image)

    return result_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/swap_faces', methods=['POST'])
def swap_faces_route():
    if 'source_image' not in request.files or 'target_image' not in request.files:
        return redirect(request.url)
    
    source_file = request.files['source_image']
    target_file = request.files['target_image']

    source_path = save_file(source_file)
    target_path = save_file(target_file)

    if source_path and target_path:
        try:
            result_path = process_images(source_path, target_path)
            return send_file(result_path, mimetype='image/jpeg')
        except ValueError as e:
            return str(e), 400

    return 'File upload failed. Please try again.'

if __name__ == '__main__':
    app.run(debug=True)
