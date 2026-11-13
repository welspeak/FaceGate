from flask import (Flask, render_template, request, redirect, url_for, flash, session, jsonify,
                   send_from_directory)
from config import get_connection
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from face_utils import (decode_base64_image, get_face_encoding, encoding_to_text,
                        recognize_faces, find_best_match, track_faces, warm_up_in_background,
                        FACES_DIR)
from plate_utils import read_plate_text
from gate_utils import open_gate
from werkzeug.utils import secure_filename
from PIL import Image
import os
import datetime
import uuid

print("=" * 50)
print("Running app from:")
print(__file__)
print("=" * 50)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def time_ago(dt):
    """Format a datetime as 'Just now', '5 minutes ago', 'Yesterday', '3 days ago', or a time for today."""
    if dt is None:
        return ''
    now = datetime.datetime.now()
    diff = now - dt
    if diff.total_seconds() < 60:
        return "Just now"
    if diff.total_seconds() < 3600:
        mins = int(diff.total_seconds() // 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    if dt.date() == now.date():
        return dt.strftime("%I:%M %p").lstrip("0")
    if dt.date() == (now - datetime.timedelta(days=1)).date():
        return "Yesterday"
    days = (now.date() - dt.date()).days
    if days < 7:
        return f"{days} days ago"
    return dt.strftime("%b %d, %Y")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_IMAGE_EXT = {'png', 'jpg', 'jpeg'}
ALLOWED_DOC_EXT = {'png', 'jpg', 'jpeg', 'pdf'}


def allowed_file(filename, allowed_ext):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext


def save_upload(file_storage, allowed_ext):
    """Save an uploaded file with a unique name. Returns the relative path, or None if no valid file."""
    if not file_storage or file_storage.filename == '':
        return None
    if not allowed_file(file_storage.filename, allowed_ext):
        return None
    ext = file_storage.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    file_storage.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_name))
    return f"uploads/{unique_name}"


# ---- Facial enrollment images (reference faces folder = FRE_DATASET_DIR in .env) ----
app.config['FACE_FOLDER'] = FACES_DIR
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FACE_FOLDER'], exist_ok=True)


def save_face_image(image_array, user_id):
    """Save the enrollment snapshot as <faces folder>/user_<id>.jpg (re-enrolling overwrites it).
    Returns the path relative to the project folder, e.g. 'static/faces/user_5.jpg'."""
    filename = f"user_{int(user_id)}.jpg"
    full_path = os.path.join(app.config['FACE_FOLDER'], filename)
    Image.fromarray(image_array).save(full_path, "JPEG", quality=90)
    return os.path.relpath(full_path, BASE_DIR).replace(os.sep, "/")


def store_face_enrollment(cur, user_id, image_array, encoding):
    """Save the face image to the faces folder and upsert the facial_data row."""
    vector_text = encoding_to_text(encoding)
    face_image_path = save_face_image(image_array, user_id)
    cur.execute("""
        INSERT INTO facial_data (user_id, biometric_vector, hash_value, face_image_path)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE biometric_vector = VALUES(biometric_vector),
                                hash_value = VALUES(hash_value),
                                face_image_path = VALUES(face_image_path),
                                status = 'active'
    """, (user_id, vector_text, vector_text[:64], face_image_path))
    return face_image_path


def ensure_face_image_column():
    """Add facial_data.face_image_path if the database doesn't have it yet."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'facial_data' AND COLUMN_NAME = 'face_image_path'
        """)
        row = cur.fetchone()
        count = row[0] if isinstance(row, (tuple, list)) else list(row.values())[0]
        if not count:
            cur.execute("ALTER TABLE facial_data ADD COLUMN face_image_path VARCHAR(255) NULL")
            conn.commit()
            print("Added facial_data.face_image_path column")
        conn.close()
    except Exception as exc:
        print("Could not verify facial_data.face_image_path column:", exc)


ensure_face_image_column()

print("====================================")
print("Project Folder :", BASE_DIR)
print("Templates Folder :", app.template_folder)
print("Static Folder :", app.static_folder)
print("====================================")

TOTAL_SLOTS = 50


def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if not (session.get("admin_id") or session.get("user_id")):
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return view(**kwargs)
    return wrapped_view


def role_required(*roles):
    """Require session['role'] to be one of the given roles. Redirects to login/dashboard as appropriate."""
    def decorator(view):
        @wraps(view)
        def wrapped_view(**kwargs):
            current = session.get('role')
            if current not in roles:
                flash('Unauthorized access.', 'danger')
                if session.get('user_id') or session.get('admin_id'):
                    return redirect(url_for('dashboard'))
                return redirect(url_for('auth_login'))
            return view(**kwargs)
        return wrapped_view
    return decorator


# -------------------------------------------------
# Dashboard
# -------------------------------------------------
@app.route("/")
@login_required
def dashboard():

    search = request.args.get("search", "").strip()

    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    # load spaces summary
    cur.execute("SELECT SUM(capacity) AS total_capacity, SUM(occupied) AS total_occupied FROM spaces")
    sums = cur.fetchone()
    total_capacity = sums.get('total_capacity') or TOTAL_SLOTS
    total_occupied = sums.get('total_occupied') or 0
    total_available = (total_capacity - total_occupied)

    # load spaces list
    cur.execute("SELECT id, name, capacity, occupied FROM spaces ORDER BY id")
    spaces = cur.fetchall()

    # preview of parking slots for the dashboard's mini grid widget
    cur.execute("""
        SELECT id, space_id, slot_code, status, plate_number
        FROM parking_slots
        WHERE space_id = (SELECT id FROM spaces ORDER BY id LIMIT 1)
        ORDER BY slot_code
        LIMIT 20
    """)
    slot_preview = cur.fetchall()

    if search:

        cur.execute("""
            SELECT
                u.*,
                (
                    SELECT action
                    FROM logs l
                    WHERE l.user_id=u.id
                    ORDER BY l.id DESC
                    LIMIT 1
                ) AS status
            FROM users u
            WHERE u.fullname LIKE %s
               OR u.plate_number LIKE %s
               OR u.vehicle_type LIKE %s
            ORDER BY u.fullname
        """,(
            f"%{search}%",
            f"%{search}%",
            f"%{search}%"
        ))

    else:

        cur.execute("""
            SELECT
                u.*,
                (
                    SELECT action
                    FROM logs l
                    WHERE l.user_id=u.id
                    ORDER BY l.id DESC
                    LIMIT 1
                ) AS status
            FROM users u
            ORDER BY u.fullname
        """)

    users = cur.fetchall()

    occupied = sum(
        1 for user in users if user["status"] == "ENTRY"
    )

    # keep dashboard counts from spaces aggregation
    available = total_available

    conn.close()

    return render_template(
        "dashboard.html",
        users=users,
        occupied=occupied,
        available=available,
        total=total_capacity,
        spaces=spaces,
        slot_preview=slot_preview,
        search=search
    )


@app.route("/logs")
@login_required
@role_required('admin', 'security')
def logs():
    def inner():

        conn = get_connection()
        cur = conn.cursor(dictionary=True)

        cur.execute("""
            SELECT
                logs.id,
                users.fullname,
                users.vehicle_type,
                users.plate_number,
                logs.action,
                logs.time_stamp,
                logs.auth_method,
                gates.name AS gate_name
            FROM logs
            INNER JOIN users ON logs.user_id = users.id
            LEFT JOIN gates ON logs.gate_id = gates.id
            ORDER BY logs.time_stamp DESC
        """)

        logs = cur.fetchall()
        conn.close()

        return render_template(
            "logs.html",
            logs=logs
        )

    return inner()


@app.route("/login", methods=["GET", "POST"])
def login():
    # Delegate to unified auth route (preserves POST data)
    return auth_login()


@app.route('/auth-login', methods=['GET', 'POST'])
def auth_login():
    # unified login for users and admins
    if session.get('user_id') or session.get('admin_id'):
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()

        conn = get_connection()
        cur = conn.cursor(dictionary=True)

        # try users table first
        cur.execute("SELECT id, fullname, username, password, role, approval_status FROM users WHERE username = %s", (username,))
        acct = cur.fetchone()
        if acct:
            stored = acct.get('password')
            if check_password_hash(stored or '', password):
                if acct.get('approval_status') == 'pending':
                    conn.close()
                    flash('Your registration is still pending admin approval.', 'warning')
                    return render_template('login.html')
                if acct.get('approval_status') == 'rejected':
                    conn.close()
                    flash('Your registration was not approved. Please contact an administrator.', 'danger')
                    return render_template('login.html')
                role = acct.get('role') or 'user'
                uid = acct['id']
                fullname = acct.get('fullname')
                conn.close()
                session.clear()
                session['user_id'] = uid
                session['user_name'] = fullname
                session['role'] = role
                if role == 'admin':
                    session['admin_id'] = uid
                    session['admin_name'] = fullname
                flash(f"Welcome, {fullname}!", 'success')
                return redirect(url_for('dashboard'))

        # fallback to admins table
        cur.execute("SELECT id, fullname, username, password FROM admins WHERE username = %s", (username,))
        admin = cur.fetchone()
        conn.close()
        if admin:
            stored = admin.get('password')
            if check_password_hash(stored or '', password):
                session.clear()
                session['admin_id'] = admin['id']
                session['admin_name'] = admin['fullname']
                session['role'] = 'admin'
                session['user_id'] = admin['id']
                session['user_name'] = admin['fullname']
                flash(f"Welcome back, {admin['fullname']}!", 'success')
                return redirect(url_for('dashboard'))

        flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route("/analytics")
@login_required
def analytics():
    # Admin-only
    @role_required('admin')
    def inner():
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM users")
        total_users = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM logs")
        total_logs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM logs WHERE action = 'ENTRY'")
        total_entries = cur.fetchone()[0] or 0

        cur.execute("SELECT COUNT(*) FROM logs WHERE action = 'EXIT'")
        total_exits = cur.fetchone()[0] or 0

        cur.execute("SELECT COUNT(*) FROM logs WHERE granted = 0")
        mismatch_alerts = cur.fetchone()[0] or 0

        cur.execute("SELECT COALESCE(SUM(fee), 0) FROM transactions WHERE payment_status = 'paid'")
        payment_collected = cur.fetchone()[0] or 0

        # Last 7 days of entries vs exits, for the trend chart
        cur.execute("""
            SELECT DATE(time_stamp) AS day,
                   SUM(CASE WHEN action = 'ENTRY' THEN 1 ELSE 0 END) AS entries,
                   SUM(CASE WHEN action = 'EXIT' THEN 1 ELSE 0 END) AS exits
            FROM logs
            WHERE time_stamp >= DATE_SUB(CURDATE(), INTERVAL 6 DAY)
            GROUP BY DATE(time_stamp)
            ORDER BY day
        """)
        daily_rows = cur.fetchall()

        conn.close()

        chart_labels = [str(r[0]) for r in daily_rows]
        chart_entries = [r[1] for r in daily_rows]
        chart_exits = [r[2] for r in daily_rows]

        return render_template(
            "analytics.html",
            total_users=total_users,
            total_logs=total_logs,
            total_entries=total_entries,
            total_exits=total_exits,
            mismatch_alerts=mismatch_alerts,
            payment_collected=payment_collected,
            chart_labels=chart_labels,
            chart_entries=chart_entries,
            chart_exits=chart_exits
        )

    return inner()


@app.route("/face-recognition")
@login_required
@role_required('admin', 'security')
def face_recognition():
    return render_template("face_recognition.html")


@app.route('/api/enroll-face', methods=['POST'])
@login_required
@role_required('admin', 'security')
def enroll_face():
    data = request.get_json(silent=True) or {}
    user_id = data.get('user_id')
    image_data = data.get('image')
    if not user_id or not image_data:
        return jsonify({'ok': False, 'error': 'user_id and image are required.'}), 400

    image_array = decode_base64_image(image_data)
    encoding = get_face_encoding(image_array)

    if encoding is None:
        return jsonify({'ok': False, 'error': 'No face detected. Try again with better lighting.'}), 400

    conn = get_connection()
    cur = conn.cursor()
    face_image_path = store_face_enrollment(cur, user_id, image_array, encoding)
    conn.commit()
    conn.close()

    return jsonify({'ok': True, 'message': 'Face enrolled successfully.', 'face_image_path': face_image_path})


@app.route('/api/verify-face', methods=['POST'])
@login_required
@role_required('admin', 'security')
def verify_face():
    """Recognize every face in a camera frame against the reference photos in static/faces.
    Used by the live recognition loop, so 'no face' is a normal 200 response."""
    data = request.get_json(silent=True) or {}
    image_data = data.get('image') or data.get('face_image')
    if not image_data:
        return jsonify({'ok': False, 'error': 'No image received.'}), 400

    try:
        image_array = decode_base64_image(image_data)
        results = recognize_faces(image_array)
    except Exception as exc:
        app.logger.exception("Face recognition failed")
        return jsonify({'ok': False, 'error': f'Face recognition failed: {exc}'}), 500

    users = lookup_face_users({r.user_id for r in results if r.matched})
    faces = []
    for r in results:
        user = users.get(r.user_id)
        faces.append({
            'bbox': [round(v, 1) for v in r.bbox],
            'matched': bool(user),
            'user_id': r.user_id if user else None,
            'fullname': user['fullname'] if user else None,
            'plate_number': user['plate_number'] if user else None,
            'approval_status': user['approval_status'] if user else None,
            'face_image_url': user['face_image_url'] if user else None,
            'similarity': round(r.similarity, 3),
        })

    best = max((f for f in faces if f['matched']), key=lambda f: f['similarity'], default=None)
    response = {
        'ok': True,
        'face_found': bool(faces),
        'faces': faces,
        'frame_width': int(image_array.shape[1]),
        'frame_height': int(image_array.shape[0]),
        'matched': best is not None,
        'face_score': best['similarity'] if best else (max((f['similarity'] for f in faces), default=0.0)),
    }
    if best:
        response.update({k: best[k] for k in ('user_id', 'fullname', 'plate_number', 'face_image_url')})
    return jsonify(response)


@app.route('/api/track-faces', methods=['POST'])
@login_required
@role_required('admin', 'security')
def track_faces_api():
    """Fast face boxes only (no names), polled many times a second so the overlay follows movement."""
    data = request.get_json(silent=True) or {}
    image_data = data.get('image')
    if not image_data:
        return jsonify({'ok': False, 'error': 'No image received.'}), 400
    try:
        image_array = decode_base64_image(image_data)
        boxes = track_faces(image_array)
    except Exception as exc:
        app.logger.exception("Face tracking failed")
        return jsonify({'ok': False, 'error': f'Face tracking failed: {exc}'}), 500
    return jsonify({
        'ok': True,
        'faces': [{'bbox': [round(v, 1) for v in bbox], 'score': round(score, 3)} for bbox, score in boxes],
        'frame_width': int(image_array.shape[1]),
        'frame_height': int(image_array.shape[0]),
    })


@app.route('/face-photo/<int:user_id>')
@login_required
@role_required('admin', 'security')
def face_photo(user_id):
    """Serve a reference face from the faces folder (FRE_DATASET_DIR), wherever it is configured."""
    return send_from_directory(app.config['FACE_FOLDER'], f"user_{user_id}.jpg", max_age=0)


def lookup_face_users(user_ids):
    """Fetch display info for recognized user ids. A reference photo whose user no longer
    exists in the database is ignored (treated as unknown)."""
    user_ids = [int(u) for u in user_ids if u is not None]
    if not user_ids:
        return {}
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    placeholders = ",".join(["%s"] * len(user_ids))
    cur.execute(f"""
        SELECT id, fullname, plate_number, approval_status
        FROM users WHERE id IN ({placeholders})
    """, user_ids)
    users = {}
    for row in cur.fetchall():
        row['face_image_url'] = url_for('face_photo', user_id=row['id'])
        users[row['id']] = row
    conn.close()
    return users


@app.route('/api/scan-plate', methods=['POST'])
@login_required
@role_required('admin', 'security')
def scan_plate():
    data = request.get_json()
    image_data = data.get('image')

    image_array = decode_base64_image(image_data)
    plate_text = read_plate_text(image_array)

    if not plate_text:
        return jsonify({'ok': False, 'error': 'No plate text detected.'}), 400

    return jsonify({'ok': True, 'plate_number': plate_text})

@app.route('/api/gate-decision', methods=['POST'])
@login_required
@role_required('admin', 'security')
def gate_decision():
    data = request.get_json()
    face_image = data.get('face_image')
    plate_image = data.get('plate_image')
    direction = data.get('direction', 'ENTRY')
    gate_code = data.get('gate_code', 'GATE-A')

    if not face_image or not plate_image:
        return jsonify({'ok': False, 'error': 'Both a face and plate image are required.'}), 400

    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    # --- Face check: match against the reference photos in static/faces ---
    face_array = decode_base64_image(face_image)
    matched_user_id, face_score, _ = find_best_match(face_array)

    face_match = None
    if matched_user_id is not None:
        cur.execute("SELECT id AS user_id, fullname, plate_number FROM users WHERE id = %s",
                    (matched_user_id,))
        face_match = cur.fetchone()

    # --- Plate check: OCR the plate camera frame ---
    plate_array = decode_base64_image(plate_image)
    detected_plate = read_plate_text(plate_array)

    # --- Dual authentication: both face AND plate must match the same registered user ---
    granted = bool(
        face_match and detected_plate
        and detected_plate.replace(" ", "") == (face_match['plate_number'] or "").replace(" ", "").upper()
    )

    user_id = face_match['user_id'] if face_match else None

    cur.execute("SELECT id FROM gates WHERE code = %s", (gate_code,))
    gate_row = cur.fetchone()
    gate_id = gate_row['id'] if gate_row else None

    cur2 = conn.cursor()
    cur2.execute("""
        INSERT INTO logs (user_id, action, granted, face_score, lpr_plate, gate_id, auth_method)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (user_id, direction, 1 if granted else 0, face_score, detected_plate, gate_id, 'Face + Plate'))
    log_id = cur2.lastrowid

    if not granted:
        if face_match and detected_plate:
            notif_type = 'mismatch'
            message = (
                f"Mismatch: face matched {face_match['fullname']} but plate read as "
                f"{detected_plate}, expected {face_match['plate_number']}."
            )
        else:
            notif_type = 'unauthorized'
            message = "Unauthorized attempt: face or plate could not be verified at the gate."

        cur2.execute("""
            INSERT INTO notification (user_id, access_id, message, type)
            VALUES (%s, %s, %s, %s)
        """, (user_id, log_id, message, notif_type))

    conn.commit()
    conn.close()

    if granted:
        gate_opened = open_gate()
        return jsonify({
            'ok': True, 'granted': True,
            'fullname': face_match['fullname'],
            'plate_number': detected_plate,
            'face_score': round(face_score, 3),
            'gate_opened': gate_opened
        })

    return jsonify({
        'ok': True, 'granted': False,
        'detected_plate': detected_plate,
        'face_score': round(face_score, 3)
    })

@app.route('/user-register', methods=['GET', 'POST'])
def user_register():
    if request.method == 'POST':
        # ---- Personal Information ----
        fullname = request.form.get('fullname', '').strip()
        school_employee_id = request.form.get('school_employee_id', '').strip()
        user_type = request.form.get('user_type', 'vehicle_owner').strip()
        email = request.form.get('email', '').strip()
        phone_number = request.form.get('phone_number', '').strip()
        address = request.form.get('address', '').strip()

        # ---- User Credentials ----
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        terms_accepted = request.form.get('terms_accepted')

        # ---- Vehicle Information ----
        vehicle = request.form.get('vehicle_type', '').strip()
        vehicle_category = request.form.get('vehicle_category', '').strip()
        plate = request.form.get('plate_number', '').strip().upper()
        color = request.form.get('color', '').strip()
        brand = request.form.get('brand', '').strip()
        model = request.form.get('model', '').strip()

        # ---- Facial Enrollment ----
        face_image = request.form.get('face_image')

        # ---- Validation ----
        if not fullname or not username or not plate:
            flash('Please complete all required fields.', 'warning')
            return redirect(url_for('user_register'))
        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'warning')
            return redirect(url_for('user_register'))
        if password != confirm_password:
            flash('Passwords do not match.', 'warning')
            return redirect(url_for('user_register'))
        if not terms_accepted:
            flash('You must agree to the Terms and Conditions to register.', 'warning')
            return redirect(url_for('user_register'))

        # ---- File uploads ----
        profile_photo_path = save_upload(request.files.get('profile_photo'), ALLOWED_IMAGE_EXT)
        registration_document_path = save_upload(request.files.get('registration_document'), ALLOWED_DOC_EXT)
        vehicle_photo_path = save_upload(request.files.get('vehicle_photo'), ALLOWED_IMAGE_EXT)

        hashed = generate_password_hash(password)

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO users (
                    fullname, vehicle_type, plate_number, username, password,
                    school_employee_id, user_type, email, phone_number, address,
                    profile_photo_path, registration_document_path, vehicle_photo_path,
                    vehicle_category, brand, model, color,
                    biometric_consent, consent_timestamp, approval_status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), 'pending')
            """, (
                fullname, vehicle, plate, username, hashed,
                school_employee_id, user_type, email, phone_number, address,
                profile_photo_path, registration_document_path, vehicle_photo_path,
                vehicle_category, brand, model, color,
                1 if face_image else 0
            ))
        except Exception as exc:
            conn.rollback()
            conn.close()
            if "Duplicate" in str(exc) or "duplicate" in str(exc):
                flash('That username or plate number is already registered.', 'danger')
            else:
                flash('Registration failed. Please check your details and try again.', 'danger')
            return redirect(url_for('user_register'))

        new_user_id = cur.lastrowid
        conn.commit()

        cur.execute("""
            INSERT INTO notification (user_id, message, type)
            VALUES (%s, %s, 'approval')
        """, (new_user_id, f"New registration pending approval: {fullname} ({plate})"))
        conn.commit()

        # Enroll the captured face, if one was provided
        face_enrolled = False
        if face_image:
            try:
                image_array = decode_base64_image(face_image)
                encoding = get_face_encoding(image_array)
                if encoding is not None:
                    store_face_enrollment(cur, new_user_id, image_array, encoding)
                    conn.commit()
                    face_enrolled = True
                else:
                    flash('Account created, but no face was detected in the capture — enroll your face later from the dashboard.', 'warning')
            except Exception:
                flash('Account created, but face enrollment failed — you can try again later.', 'warning')

        conn.close()

        return render_template('registration_success.html', fullname=fullname, face_enrolled=face_enrolled)

    return render_template('user_register.html')


@app.route('/user-login', methods=['GET', 'POST'])
def user_login():
    # delegate to unified auth login (preserves POST)
    return auth_login()

# -------------------------------------------------
# Register Page
# -------------------------------------------------
@app.route("/register")
@login_required
@role_required('admin', 'security')
def register():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT u.id, u.fullname, u.plate_number, u.vehicle_type, u.color, u.model,
               (fd.user_id IS NOT NULL) AS face_enrolled,
               latest.status AS current_status
        FROM users u
        LEFT JOIN facial_data fd ON fd.user_id = u.id AND fd.status = 'active'
        LEFT JOIN (
            SELECT t1.*
            FROM transactions t1
            INNER JOIN (
                SELECT plate_number, MAX(id) AS max_id
                FROM transactions
                GROUP BY plate_number
            ) t2 ON t1.plate_number = t2.plate_number AND t1.id = t2.max_id
        ) latest ON latest.plate_number = u.plate_number
        ORDER BY u.id DESC
    """)
    vehicles = cur.fetchall()
    conn.close()
    return render_template("register.html", vehicles=vehicles)

@app.route('/vehicle-approvals')
@login_required
@role_required('admin')
def vehicle_approvals():
    search = request.args.get('search', '').strip()

    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT COUNT(*) AS c FROM users")
    total_requests = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) AS c FROM users WHERE approval_status = 'pending'")
    pending_count = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) AS c FROM users WHERE approval_status = 'approved'")
    approved_count = cur.fetchone()['c']
    cur.execute("SELECT COUNT(*) AS c FROM users WHERE approval_status = 'rejected'")
    rejected_count = cur.fetchone()['c']

    query = """
        SELECT id, fullname, plate_number, vehicle_type, user_type, approval_status, requested_at
        FROM users
        WHERE 1=1
    """
    params = []
    if search:
        query += " AND (fullname LIKE %s OR plate_number LIKE %s)"
        params.extend([f"%{search}%", f"%{search}%"])
    query += " ORDER BY requested_at DESC"

    cur.execute(query, params)
    requests_list = cur.fetchall()
    conn.close()

    return render_template(
        "vehicle_approvals.html",
        requests=requests_list,
        total_requests=total_requests,
        pending_count=pending_count,
        approved_count=approved_count,
        rejected_count=rejected_count,
        search=search
    )



@app.route('/vehicle-approvals/approve/<int:id>', methods=['POST'])
@login_required
@role_required('admin')
def approve_vehicle(id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE users SET approval_status = 'approved', reviewed_by = %s, reviewed_at = NOW()
        WHERE id = %s
    """, (session.get('admin_id'), id))
    conn.commit()
    conn.close()
    flash('Request approved.', 'success')
    return redirect(url_for('vehicle_approvals'))


@app.route('/vehicle-approvals/reject/<int:id>', methods=['POST'])
@login_required
@role_required('admin')
def reject_vehicle(id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE users SET approval_status = 'rejected', reviewed_by = %s, reviewed_at = NOW()
        WHERE id = %s
    """, (session.get('admin_id'), id))
    conn.commit()
    conn.close()
    flash('Request rejected.', 'success')
    return redirect(url_for('vehicle_approvals'))

@app.route('/notifications')
@login_required
@role_required('admin', 'security')
def notifications():
    tab = request.args.get('tab', 'all')

    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    query = "SELECT * FROM notification WHERE 1=1"
    params = []
    if tab == 'unread':
        query += " AND is_read = 0"
    elif tab in ('mismatch', 'unauthorized', 'payment', 'approval'):
        query += " AND type = %s"
        params.append(tab)
    query += " ORDER BY created_at DESC LIMIT 200"

    cur.execute(query, params)
    rows = cur.fetchall()
    for r in rows:
        r['display_time'] = time_ago(r['created_at'])

    cur.execute("SELECT COUNT(*) AS c FROM notification WHERE is_read = 0")
    unread_count = cur.fetchone()['c']

    conn.close()

    return render_template('notifications.html', notifications=rows, tab=tab, unread_count=unread_count)


@app.route('/notifications/mark-all-read', methods=['POST'])
@login_required
@role_required('admin', 'security')
def mark_all_read():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE notification SET is_read = 1 WHERE is_read = 0")
    conn.commit()
    conn.close()
    flash('All notifications marked as read.', 'success')
    return redirect(url_for('notifications', tab=request.form.get('tab', 'all')))

# -------------------------------------------------
# Save User
# -------------------------------------------------
@app.route("/save", methods=["POST"])
@login_required
def save():

    fullname = request.form.get("fullname", "").strip()
    vehicle = request.form.get("vehicle_type", "").strip()
    plate = request.form.get("plate_number", "").strip().upper()
    color = request.form.get("color", "").strip()
    model = request.form.get("model", "").strip()

    if not fullname or not vehicle or not plate:
        flash("Please complete all registration fields.", "warning")
        return redirect(url_for("register"))

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO users
            (fullname, vehicle_type, plate_number, color, model)
            VALUES (%s, %s, %s, %s, %s)
        """, (fullname, vehicle, plate, color, model))
        conn.commit()
    except Exception as exc:
        conn.rollback()
        if "Duplicate" in str(exc) or "duplicate" in str(exc):
            flash("This plate number is already registered.", "danger")
        else:
            flash("Unable to register vehicle. Please try again.", "danger")
        conn.close()
        return redirect(url_for("register"))

    conn.close()
    flash("Vehicle registered successfully.", "success")
    return redirect(url_for("dashboard"))

@app.route('/vehicles/delete/<int:id>', methods=['POST'])
@login_required
@role_required('admin')
def delete_vehicle(id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE id = %s", (id,))
    conn.commit()
    conn.close()
    flash('Vehicle record deleted.', 'success')
    return redirect(url_for('register'))

# -------------------------------------------------
# Vehicle Entry
# -------------------------------------------------
@app.route("/entry/<int:id>")
@login_required
def entry(id):

    conn = get_connection()
    cur = conn.cursor()

    gate_code = request.args.get('gate', 'GATE-A')
    cur.execute("SELECT id FROM gates WHERE code = %s", (gate_code,))
    gate_row = cur.fetchone()
    gate_id = gate_row[0] if gate_row else None

    cur.execute("""
        INSERT INTO logs(user_id, action, gate_id, auth_method)
        VALUES(%s, 'ENTRY', %s, 'Manual (Dashboard)')
    """, (id, gate_id))

    conn.commit()
    conn.close()

    return redirect(url_for("dashboard"))


# -------------------------------------------------
# Vehicle Exit
# -------------------------------------------------
@app.route("/exit/<int:id>")
@login_required
def exit_vehicle(id):

    conn = get_connection()
    cur = conn.cursor()

    gate_code = request.args.get('gate', 'GATE-A')
    cur.execute("SELECT id FROM gates WHERE code = %s", (gate_code,))
    gate_row = cur.fetchone()
    gate_id = gate_row[0] if gate_row else None

    cur.execute("""
        INSERT INTO logs(user_id, action, gate_id, auth_method)
        VALUES(%s, 'EXIT', %s, 'Manual (Dashboard)')
    """, (id, gate_id))

    conn.commit()
    conn.close()

    return redirect("/")


# -----------------------------------
# Record Entry (API)
# -----------------------------------
@app.route('/record-entry', methods=['POST'])
@login_required
def record_entry():
    data = request.get_json() or request.form
    plate = data.get('plate') or data.get('plate_number')
    user_id = data.get('user_id') or None
    space_id = data.get('space_id') or None

    conn = get_connection()
    cur = conn.cursor()

    slot_id = None
    if space_id:
        cur.execute(
            "SELECT id, slot_code FROM parking_slots WHERE space_id = %s AND status = 'available' LIMIT 1",
            (space_id,)
        )
        slot_row = cur.fetchone()
        if slot_row:
            slot_id = slot_row[0]

    cur.execute(
        "INSERT INTO transactions (user_id, plate_number, space_id, slot_id, entry_time, status) VALUES (%s, %s, %s, %s, NOW(), 'entered')",
        (user_id, plate, space_id, slot_id)
    )
    tx_id = cur.lastrowid

    if space_id:
        try:
            cur.execute("UPDATE spaces SET occupied = occupied + 1 WHERE id = %s", (space_id,))
        except Exception:
            pass

    if slot_id:
        cur.execute(
            "UPDATE parking_slots SET status = 'occupied', plate_number = %s WHERE id = %s",
            (plate, slot_id)
        )

    conn.commit()
    conn.close()

    # If this was a regular HTML form POST, redirect back to dashboard
    if request.form:
        flash('Entry recorded.', 'success')
        return redirect(url_for('dashboard'))

    return jsonify({'ok': True, 'transaction_id': tx_id}), 201


# -----------------------------------
# Record Exit (API)
# -----------------------------------
@app.route('/record-exit', methods=['POST'])
@login_required
def record_exit():
    data = request.get_json() or request.form
    tx_id = data.get('transaction_id')
    plate = data.get('plate')

    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    if tx_id:
        cur.execute("SELECT * FROM transactions WHERE id = %s AND status = 'entered'", (tx_id,))
    else:
        cur.execute("SELECT * FROM transactions WHERE plate_number = %s AND status = 'entered' ORDER BY entry_time DESC LIMIT 1", (plate,))

    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({'ok': False, 'error': 'no active entry found'}), 404

    entry_time = row.get('entry_time')
    space_id = row.get('space_id')
    slot_id = row.get('slot_id')

    now = datetime.datetime.now()
    fee = 0.0
    if entry_time:
        delta = now - entry_time
        hours = delta.total_seconds() / 3600.0
        RATE = 1.50
        fee = round(hours * RATE, 2)

    cur2 = conn.cursor()
    cur2.execute("UPDATE transactions SET exit_time = NOW(), status = 'exited', fee = %s WHERE id = %s", (fee, row['id']))

    if space_id:
        try:
            cur2.execute("UPDATE spaces SET occupied = GREATEST(0, occupied - 1) WHERE id = %s", (space_id,))
        except Exception:
            pass

    if slot_id:
        cur2.execute(
            "UPDATE parking_slots SET status = 'available', plate_number = NULL WHERE id = %s",
            (slot_id,)
        )

    conn.commit()
    conn.close()

    if request.form:
        flash(f'Exit recorded. Fee: ${fee}', 'success')
        return redirect(url_for('dashboard'))

    return jsonify({'ok': True, 'transaction_id': row['id'], 'fee': float(fee)}), 200


@app.route('/transactions')
@login_required
def transactions():
    @role_required('admin')
    def inner():
        search = request.args.get('search', '').strip()

        conn = get_connection()
        cur = conn.cursor(dictionary=True)

        cur.execute("SELECT COUNT(*) AS c FROM transactions WHERE DATE(created_at) = CURDATE()")
        today_count = cur.fetchone()['c']
        cur.execute("SELECT COUNT(*) AS c FROM transactions WHERE payment_status = 'paid'")
        paid_count = cur.fetchone()['c']
        cur.execute("SELECT COUNT(*) AS c FROM transactions WHERE payment_status = 'unpaid'")
        unpaid_count = cur.fetchone()['c']
        cur.execute("SELECT COUNT(*) AS c FROM transactions WHERE payment_status = 'pending'")
        pending_count = cur.fetchone()['c']
        cur.execute("SELECT COALESCE(SUM(fee), 0) AS total FROM transactions WHERE payment_status = 'paid'")
        total_collected = cur.fetchone()['total']

        query = '''
            SELECT t.*, u.fullname AS user_name, a.fullname AS admin_name, s.name AS space_name
            FROM transactions t
            LEFT JOIN users u ON t.user_id = u.id
            LEFT JOIN admins a ON t.admin_id = a.id
            LEFT JOIN spaces s ON t.space_id = s.id
            WHERE 1=1
        '''
        params = []
        if search:
            query += " AND (t.plate_number LIKE %s OR u.fullname LIKE %s)"
            params.extend([f"%{search}%", f"%{search}%"])
        query += " ORDER BY t.created_at DESC LIMIT 200"

        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()

        return render_template(
            'transactions.html',
            transactions=rows,
            search=search,
            today_count=today_count,
            paid_count=paid_count,
            unpaid_count=unpaid_count,
            pending_count=pending_count,
            total_collected=total_collected
        )

    return inner()

@app.route('/transactions/record-payment/<int:id>', methods=['POST'])
@login_required
@role_required('admin')
def record_payment(id):
    payment_status = request.form.get('payment_status', 'pending')
    payment_method = request.form.get('payment_method', '').strip()
    reference_no = request.form.get('reference_no', '').strip()
    notes = request.form.get('notes', '').strip()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE transactions
        SET payment_status = %s, payment_method = %s, reference_no = %s, notes = %s,
            paid_at = CASE WHEN %s = 'paid' THEN NOW() ELSE paid_at END
        WHERE id = %s
    """, (payment_status, payment_method, reference_no, notes, payment_status, id))

    if payment_status == 'paid':
        cur.execute("""
            INSERT INTO notification (message, type)
            VALUES (%s, 'payment')
        """, (f"Payment recorded via {payment_method or 'unspecified method'} (ref: {reference_no or 'N/A'}).",))

    conn.commit()
    conn.close()

    flash('Payment record updated.', 'success')
    return redirect(url_for('transactions'))

# -----------------------------------
# Availability
# -----------------------------------
@app.route('/availability')
def availability():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, name, capacity, occupied, (capacity - occupied) AS available FROM spaces")
    spaces = cur.fetchall()
    conn.close()
    return jsonify(spaces)


# -----------------------------------
# Logout
# -----------------------------------

@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out successfully.", "success")

    return redirect(url_for("login"))

@app.route('/parking')
@login_required
@role_required('admin', 'security')
def parking():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, name, capacity, occupied FROM spaces ORDER BY id")
    spaces = cur.fetchall()
    cur.execute("SELECT id, space_id, slot_code, status, plate_number FROM parking_slots ORDER BY space_id, slot_code")
    slots = cur.fetchall()
    conn.close()
    return render_template("parking.html", spaces=spaces, slots=slots)


@app.route('/api/parking-grid')
@login_required
@role_required('admin', 'security')
def parking_grid_api():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, space_id, slot_code, status, plate_number FROM parking_slots ORDER BY space_id, slot_code")
    slots = cur.fetchall()
    conn.close()
    return jsonify(slots)

# -------------------------------------------------
# Run Flask
# -------------------------------------------------
if __name__ == "__main__":
    # With debug=True Flask runs the app in a reloader child process; load the
    # face models there only, so they are not loaded twice.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        warm_up_in_background()
    app.run(debug=True)