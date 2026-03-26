import mysql.connector
import streamlit as st
import base64

# =====================================================
# CONNECTION — Aiven MySQL requires SSL
# =====================================================
def get_connection():
    return mysql.connector.connect(
        host        = st.secrets["mysql"]["host"],
        port        = int(st.secrets["mysql"]["port"]),
        user        = st.secrets["mysql"]["user"],
        password    = st.secrets["mysql"]["password"],
        database    = st.secrets["mysql"]["database"],
        ssl_disabled= False   # Aiven requires SSL — do not change this
    )

# =====================================================
# SETUP — create all tables on first run
# =====================================================
def init_db():
    try:
        conn = get_connection()
        cur  = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id         INT AUTO_INCREMENT PRIMARY KEY,
                full_name  VARCHAR(100) NOT NULL,
                age        INT NOT NULL,
                degree     VARCHAR(50)  NOT NULL,
                hospital   VARCHAR(150),
                city       VARCHAR(100),
                email      VARCHAR(150) UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS symptom_contributions (
                id           INT AUTO_INCREMENT PRIMARY KEY,
                doctor_id    INT NOT NULL,
                disease_name VARCHAR(200) NOT NULL,
                symptoms     TEXT NOT NULL,
                notes        TEXT,
                submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS image_contributions (
                id           INT AUTO_INCREMENT PRIMARY KEY,
                doctor_id    INT NOT NULL,
                disease_name VARCHAR(200) NOT NULL,
                image_data   LONGBLOB NOT NULL,
                image_name   VARCHAR(200),
                notes        TEXT,
                submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            )
        """)

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")

# =====================================================
# DOCTORS
# =====================================================
def register_doctor(full_name, age, degree, hospital, city, email):
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO doctors (full_name, age, degree, hospital, city, email)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (full_name, age, degree, hospital, city, email))
        conn.commit()
        doctor_id = cur.lastrowid
        cur.close()
        conn.close()
        return True, "Registration successful!", doctor_id
    except mysql.connector.IntegrityError:
        return False, "This email is already registered. Please log in instead.", None
    except Exception as e:
        return False, f"Database error: {str(e)}", None

def login_doctor(email):
    try:
        conn   = get_connection()
        cur    = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM doctors WHERE email = %s", (email,))
        doctor = cur.fetchone()
        cur.close()
        conn.close()
        return doctor
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return None

# =====================================================
# CONTRIBUTIONS
# =====================================================
def submit_symptom_contribution(doctor_id, disease_name, symptoms_list, notes=""):
    try:
        conn = get_connection()
        cur  = conn.cursor()
        symptoms_str = ", ".join([s.strip() for s in symptoms_list if s.strip()])
        cur.execute("""
            INSERT INTO symptom_contributions (doctor_id, disease_name, symptoms, notes)
            VALUES (%s, %s, %s, %s)
        """, (doctor_id, disease_name, symptoms_str, notes))
        conn.commit()
        cur.close()
        conn.close()
        return True, "Symptom data submitted successfully!"
    except Exception as e:
        return False, f"Error saving data: {str(e)}"

def submit_image_contribution(doctor_id, disease_name, image_bytes, image_name, notes=""):
    try:
        conn    = get_connection()
        cur     = conn.cursor()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        cur.execute("""
            INSERT INTO image_contributions (doctor_id, disease_name, image_data, image_name, notes)
            VALUES (%s, %s, %s, %s, %s)
        """, (doctor_id, disease_name, encoded, image_name, notes))
        conn.commit()
        cur.close()
        conn.close()
        return True, "Image submitted successfully!"
    except Exception as e:
        return False, f"Error saving image: {str(e)}"

# =====================================================
# STATS
# =====================================================
def get_contribution_stats():
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM doctors")
        total_doctors = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM symptom_contributions")
        total_symptoms = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM image_contributions")
        total_images = cur.fetchone()[0]
        cur.close()
        conn.close()
        return {"doctors": total_doctors, "symptom_entries": total_symptoms, "image_entries": total_images}
    except Exception as e:
        return {"doctors": 0, "symptom_entries": 0, "image_entries": 0}
