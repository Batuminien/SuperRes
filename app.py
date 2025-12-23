import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from pipeline import run_classical_example_sr

app = Flask(__name__)
app.secret_key = "local-dev-secret"

UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/run")
def run():
    method = request.form.get("method", "classical")
    metric = request.form.get("metric", "off")
    use_degradation = (metric == "on")
    
    scale_raw = request.form.get("scale", "2")
    try:
        scale = int(scale_raw)
        if scale not in (2, 3, 4):
            scale = 2
    except:
        scale = 2
    
    if method != "classical":
        flash("Bu yöntem şimdilik sadece arayüzde var. Classical yöntemi seç.")
        return redirect(url_for("index"))
    
    if "image" not in request.files:
        flash("Görsel seçemedim. Tekrar dener misin?")
        return redirect(url_for("index"))
    
    file = request.files["image"]
    if file.filename == "":
        flash("Dosya adı boş. Tekrar dene.")
        return redirect(url_for("index"))
    
    if not allowed_file(file.filename):
        flash("Sadece png/jpeg/jpg/webp destekli.")
        return redirect(url_for("index"))
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)
    
    if method != "classical":
        # input'u gösteren şık placeholder result
        result = {
            "job_id": "preview",
            "method": method,
            "use_degradation": False,
            "input": f"uploads/{filename}",
            "lr_degraded": None,
            "bicubic": None,
            "ours": None,
            "metrics": None,
            "message": "Bu yöntem şimdilik arayüzde var. Sonraki adımda fonksiyonelliğini ekleyeceğiz.",
        }
        return render_template("result.html", result=result)

    # Classical çalıştır
    try:
        result = run_classical_example_sr(
            input_path=save_path,
            use_degradation=use_degradation,
            target_scale=scale,
            out_dir=os.path.join("static", "results"),
        )
        result["method"] = "classical"
        result["scale"] = scale
        return render_template("result.html", result=result)
    except Exception as e:
        flash(f"Çalıştırırken hata oldu: {e}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)