import cv2
import numpy as np
import json
from ultralytics import YOLO
from threading import Thread
import time
from flask import Flask, request, jsonify
import os
from functools import wraps
import jwt  # Ensure this is PyJWT: pip install PyJWT
from flask import send_file, Response
import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import threading

# Config
DB_HOST = "postgres"
DB_PORT = 5432
DB_NAME = "tracking"
DB_USER = "postgres"
DB_PASSWORD = "postgres"

# Config
SECRET_KEY = "d2934071d4061bbcb42e94501c71522d2245c1bbe8169f24ea2d211f490f3dbdf0599076dbe5d16095593c70e90ec3423acfa5b6e98748e5e8d3d7e5bb1178b8"
USERS_FILE = "data/users.json"
CONFIG_FILE = "data/config_cameras.json"

# Utilitário de token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token or not token.startswith("Bearer "):
            return jsonify({"erro": "Token ausente"}), 401
        try:
            decoded = jwt.decode(token[7:], SECRET_KEY, algorithms=["HS256"])
            request.usuario = decoded["user"]
        except jwt.ExpiredSignatureError:
            return jsonify({"erro": "Token expirado"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"erro": "Token inválido"}), 401
        return f(*args, **kwargs)
    return decorated

# Classe de rastreamento (mantida)
class PessoaTracker:
    def __init__(self, camera_id, model_path, linha, direcao_entrada, inatividade_max=30):
        self.ultimo_salvamento = 0
        self.camera_id = camera_id
        self.model = YOLO(model_path)
        self.model.trackers = {"default": "botsort.yaml"}
        self.linha = linha
        self.direcao_entrada = direcao_entrada
        self.inatividade_max = inatividade_max
        self.frame_count = 0
        self.ids_ativos = {}
        self.count_in = 0
        self.count_out = 0

    def direcao_movimento(self, prev, atual):
        dx = atual[0] - prev[0]
        dy = atual[1] - prev[1]
        if abs(dy) > abs(dx):
            return "cima_para_baixo" if dy > 0 else "baixo_para_cima"
        else:
            return "esquerda_para_direita" if dx > 0 else "direita_para_esquerda"

    def cruzou_linha(self, ponto_ant, ponto_atual):
        x1, y1 = self.linha[0]
        x2, y2 = self.linha[1]
        def lado(p):
            return np.sign((x2 - x1)*(p[1] - y1) - (y2 - y1)*(p[0] - x1))
        return lado(ponto_ant) != lado(ponto_atual)

    def salvar_frame_live(self, frame):
        # salva miniatura otimizada também
        agora = time.time()
        if agora - self.ultimo_salvamento < 1:
            return
        self.ultimo_salvamento = agora
        output_dir = f"public/cameras/{self.camera_id}"
        os.makedirs(output_dir, exist_ok=True)
        temp_path = os.path.join(output_dir, "live_temp.jpg")
        final_path = os.path.join(output_dir, "live.jpg")
        try:
            qualidade = 80 if frame.shape[1] >= 1280 else 60  # qualidade maior se resolução for alta
            cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
            os.replace(temp_path, final_path)

            # gerar miniatura 320x240
            thumb_path = os.path.join(output_dir, "live_thumb.jpg")
            thumb = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            cv2.imwrite(thumb_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 40])
        except Exception as e:
            print(f"[ERRO] Falha ao salvar live.jpg para {self.camera_id}: {e}")

    def ajustar_linha_para_resolucao(self, frame):
        h, w = frame.shape[:2]
        x1, y1 = self.linha[0]
        x2, y2 = self.linha[1]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        self.linha = [(x1, y1), (x2, y2)]

    def processar_frame(self, frame):
        self.ajustar_linha_para_resolucao(frame)
        self.frame_count += 1
        results = self.model.track(frame, persist=True, tracker="botsort.yaml")[0]
        if results.boxes is not None and results.boxes.id is not None:
            classes = results.boxes.cls.int().tolist()
            ids = results.boxes.id.int().tolist()
            bboxes = results.boxes.xywh.cpu().numpy()
            for i, track_id in enumerate(ids):
                if classes[i] != 0:  # Apenas pessoas (classe 0 no COCO)
                    continue
                x, y, w, h = bboxes[i]
                centro_atual = (int(x), int(y))
                if track_id not in self.ids_ativos:
                    self.ids_ativos[track_id] = {"centro": centro_atual, "frame": self.frame_count, "contado": False}
                else:
                    centro_ant = self.ids_ativos[track_id]["centro"]
                    contado = self.ids_ativos[track_id]["contado"]
                    if not contado and self.cruzou_linha(centro_ant, centro_atual):
                        direcao = self.direcao_movimento(centro_ant, centro_atual)
                        if direcao == self.direcao_entrada:
                            self.count_in += 1
                        else:
                            self.count_out += 1
                        self.ids_ativos[track_id]["contado"] = True
                    # Desenhar seta antes de sobrescrever o centro
                    dx = centro_atual[0] - centro_ant[0]
                    dy = centro_atual[1] - centro_ant[1]
                    norm = max(1, np.hypot(dx, dy))
                    escala = max(int(h * 0.4), 50)
                    ponta = (int(centro_atual[0] + escala * dx / norm), int(centro_atual[1] + escala * dy / norm))
                    cor_seta = (0, 255, 0) if self.direcao_movimento(centro_ant, centro_atual) == self.direcao_entrada else (0, 0, 255)
                    cv2.arrowedLine(frame, centro_atual, ponta, cor_seta, 2, tipLength=0.5)
                    self.ids_ativos[track_id]["centro"] = centro_atual
                    self.ids_ativos[track_id]["frame"] = self.frame_count
                # Desenhar ID e centro
                cv2.circle(frame, centro_atual, 3, (0, 255, 0), -1)


                cv2.putText(frame, f"ID {track_id}", (centro_atual[0] + 5, centro_atual[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        inativos = [tid for tid, dados in self.ids_ativos.items() if self.frame_count - dados["frame"] > self.inatividade_max]
        for tid in inativos:
            del self.ids_ativos[tid]

        cv2.line(frame, self.linha[0], self.linha[1], (255, 0, 0), 2)
        cv2.putText(frame, f"In: {self.count_in}  Out: {self.count_out}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        self.salvar_frame_live(frame)
        return frame


class CameraWorker(Thread):
    def __init__(self, camera_id, rtsp_url, config):
        super().__init__()
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.config = config
        self.running = True
        self.tracker = PessoaTracker(
            camera_id=camera_id,
            model_path=config["model"],
            linha=tuple(map(tuple, config["linha"])),
            direcao_entrada=config["direcao"],
            inatividade_max=config.get("inatividade", 30)
        )
        self.cap = None

    def abrir_stream(self):
        self.cap = cv2.VideoCapture(self.rtsp_url)
        tentativas = 0
        while not self.cap.isOpened() and tentativas < 5:
            print(f"[{self.tracker.camera_id}] Tentando abrir stream RTSP...")
            time.sleep(2)
            self.cap.open(self.rtsp_url)
            tentativas += 1
        return self.cap.isOpened()

    def run(self):
        if not self.abrir_stream():
            print(f"[{self.tracker.camera_id}] Falha ao conectar no stream RTSP")
            return
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"[{self.tracker.camera_id}] Frame inválido. Tentando reconectar...")
                self.cap.release()
                time.sleep(2)
                if not self.abrir_stream():
                    print(f"[{self.tracker.camera_id}] Reconexão falhou. Encerrando thread.")
                    break
                continue
            frame = self.tracker.processar_frame(frame)
            cv2.imshow(f"Camera {self.tracker.camera_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def parar(self):
        self.running = False

    def resultados(self):
        return self.tracker.exportar_json()

# Flask
app = Flask(__name__)
workers = {}

# Carrega usuários e câmeras
def carregar_usuarios():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return []

def carregar_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

def salvar_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

usuarios = carregar_usuarios()
camera_configs = carregar_config()

# Inicia câmeras existentes
for camera_id, cam_cfg in camera_configs.items():
    worker = CameraWorker(camera_id, cam_cfg["rtsp"], cam_cfg)
    worker.start()
    workers[camera_id] = worker

@app.route("/login", methods=["POST"])
def login():
    dados = request.json
    username = dados.get("username")
    password = dados.get("password")

    for user in usuarios:
        if user["username"] == username and user["password"] == password:
            token = jwt.encode({
                "user": username,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=8)
            }, SECRET_KEY, algorithm="HS256")
            # PyJWT returns bytes in v2+, so decode to str
            if isinstance(token, bytes):
                token = token.decode('utf-8')
            return jsonify({"token": token})

    return jsonify({"erro": "Usuário ou senha inválidos"}), 401

@app.route("/cameras", methods=["GET"])
@token_required
def listar_cameras():
    return jsonify(camera_configs)

@app.route("/cameras", methods=["POST"])
@token_required
def adicionar_camera():
    data = request.json
    camera_id = data["camera_id"]
    if camera_id in workers:
        return jsonify({"erro": "Camera já existe"}), 400
    camera_configs[camera_id] = data
    salvar_config(camera_configs)
    worker = CameraWorker(camera_id, data["rtsp"], data)
    worker.start()
    workers[camera_id] = worker
    return jsonify({"mensagem": "Camera adicionada"})

@app.route("/cameras/<camera_id>", methods=["PUT"])
@token_required
def editar_camera(camera_id):
    if camera_id not in workers:
        return jsonify({"erro": "Camera não encontrada"}), 404
    nova_config = request.json
    atual_rtsp = camera_configs[camera_id]["rtsp"]
    camera_configs[camera_id] = nova_config
    salvar_config(camera_configs)
    if nova_config["rtsp"] != atual_rtsp:
        workers[camera_id].parar()
        workers[camera_id].join()
        worker = CameraWorker(camera_id, nova_config["rtsp"], nova_config)
        worker.start()
        workers[camera_id] = worker
    return jsonify({"mensagem": "Camera atualizada"})

@app.route("/cameras/<camera_id>", methods=["DELETE"])
@token_required
def remover_camera(camera_id):
    if camera_id not in workers:
        return jsonify({"erro": "Camera não encontrada"}), 404
    workers[camera_id].parar()
    workers[camera_id].join()
    del workers[camera_id]
    if camera_id in camera_configs:
        del camera_configs[camera_id]
        salvar_config(camera_configs)
    return jsonify({"mensagem": "Camera removida"})

@app.route("/cameras/<camera_id>/imagem", methods=["GET"])
@token_required
def obter_imagem(camera_id):
    path = f"public/cameras/{camera_id}/live.jpg"
    if os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return jsonify({"erro": "Imagem não encontrada"}), 404

"""
Endpoint to retrieve tracking data from the database.

GET /tracking
Query Parameters:
    camera_id (str, optional): Filter results by camera ID.
    limit (int, optional): Maximum number of records to return (default: 100).

Returns:
    JSON array of tracking data records, ordered by timestamp descending.
    If an error occurs, returns a JSON object with an "error" message and HTTP status 500.

Usage:
    Send a GET request to /tracking with optional query parameters 'camera_id' and 'limit'.
    Example: GET /tracking?camera_id=CAM123&limit=50
    Requires authentication via token.
"""
@app.route("/tracking", methods=["GET"])
@token_required
def get_tracking_data():
    camera_id = request.args.get("camera_id")
    limit = int(request.args.get("limit", 100))
    try:
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if camera_id:
            cur.execute("SELECT * FROM tracking_data WHERE camera_id = %s ORDER BY timestamp DESC LIMIT %s", (camera_id, limit))
        else:
            cur.execute("SELECT * FROM tracking_data ORDER BY timestamp DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/cameras/<camera_id>/thumb", methods=["GET"])
@token_required
def obter_thumbnail(camera_id):
    path = f"public/cameras/{camera_id}/live_thumb.jpg"
    if os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return jsonify({"erro": "Miniatura não encontrada"}), 404





def init_db():
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE DATABASE tracking")
    cur.close()
    conn.close()
    time.sleep(1)

def ensure_schema():
    with psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tracking_data (
                    id SERIAL PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    count_in INTEGER NOT NULL,
                    count_out INTEGER NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS system_log (
                    id SERIAL PRIMARY KEY,
                    event TEXT NOT NULL,
                    camera_id TEXT,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    details JSONB
                )
            """)

def log_tracking(camera_id, count_in, count_out):
    with psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tracking_data (camera_id, timestamp, count_in, count_out)
                VALUES (%s, NOW(), %s, %s)
                """,
                (camera_id, count_in, count_out)
            )

def start_periodic_db_logger(trackers):
    def loop():
        while True:
            time.sleep(60)
            for tracker in trackers.values():
                log_tracking(tracker.camera_id, tracker.count_in, tracker.count_out)
    t = threading.Thread(target=loop, daemon=True)
    t.start()


if __name__ == "__main__":
    try:
        psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD).close()
    except:
        init_db()
    ensure_schema()
    start_periodic_db_logger(workers)

    os.makedirs("data", exist_ok=True)
    app.run(host="0.0.0.0", port=8000, threaded=True)
