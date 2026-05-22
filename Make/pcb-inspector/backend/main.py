"""
main.py — Backend FastAPI PCB Inspector.

Endpoints :
  POST /analyze     → pipeline complet (YOLO + OpenCV + Gemini)
  GET  /health      → statut du service
  GET  /model-info  → infos sur le modèle YOLO chargé
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import asyncio
import traceback
from typing import Optional

from yolo_detector     import detect_from_bytes, detect
from opencv_comparator import comparer, anomalies_to_dict
from gemini_explainer  import expliquer
from annotator         import annoter_image, annoter_reference, image_to_base64
from webhook           import envoyer_rapport
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

BASE_DIR = Path(__file__).parent

app = FastAPI(
    title       = "PCB Inspector API",
    description = "Pipeline YOLO + OpenCV + Gemini pour inspection AOI de cartes PCB",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # En prod: restreindre à ton domaine
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── /health ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "PCB Inspector API v1.0.0"}


# ── /model-info ───────────────────────────────────────────────────────────────
@app.get("/model-info")
async def model_info():
    try:
        from yolo_detector import MODEL_PATH, CLASS_NAMES
        return {
            "model_path":  str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "classes":     CLASS_NAMES
        }
    except Exception as e:
        return {"error": str(e)}


# ── /analyze ─────────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(
    image_inspect: UploadFile = File(...,   description="Image de la carte à inspecter"),
    image_ref:     UploadFile = File(None,  description="Image de référence (optionnel)"),
    gemini_key:    str        = Form(...,   description="Clé API Gemini"),
    webhook_url:   str        = Form("",    description="URL webhook Make (optionnel)"),
    batch_id:      str        = Form("",    description="ID de lot"),
    chain_context: str        = Form("",    description="Contexte chaîne d'assemblage"),
    yolo_conf:     float      = Form(0.25,  description="Seuil de confiance YOLO"),
):
    """
    Pipeline complet d'inspection PCB :
      1. YOLO → détecte composants sur REF et INSP
      2. OpenCV → compare positions géométriques
      3. Gemini → explique les anomalies (2 passes)
      4. Annotator → dessine les bboxes sur les images
      5. Webhook → envoie tout à Make (si URL fournie)
    """

    logs = []
    def log(msg: str):
        logs.append(msg)
        print(msg)

    try:
        # ── Lecture des images ────────────────────────────────────────────────
        insp_bytes = await image_inspect.read()
        ref_bytes  = await image_ref.read() if image_ref else None

        arr_insp = np.frombuffer(insp_bytes, np.uint8)
        img_insp = cv2.imdecode(arr_insp, cv2.IMREAD_COLOR)
        if img_insp is None:
            raise HTTPException(400, "Image d'inspection invalide")

        img_ref = None
        if ref_bytes:
            arr_ref = np.frombuffer(ref_bytes, np.uint8)
            img_ref = cv2.imdecode(arr_ref, cv2.IMREAD_COLOR)

        H, W = img_insp.shape[:2]
        log(f"[IMAGE] Inspection {W}x{H}px | REF: {'oui' if img_ref is not None else 'non'}")

        # ── ÉTAPE 1 : YOLO ────────────────────────────────────────────────────
        log("[YOLO] Détection en cours...")

        detections_insp = detect(img_insp, conf=yolo_conf)
        log(f"[YOLO] INSP: {len(detections_insp)} composant(s) détecté(s)")

        detections_ref = []
        if img_ref is not None:
            detections_ref = detect(img_ref, conf=yolo_conf)
            log(f"[YOLO] REF:  {len(detections_ref)} composant(s) détecté(s)")

        # ── ÉTAPE 2 : OpenCV comparaison ─────────────────────────────────────
        log("[OPENCV] Comparaison géométrique...")
        opencv_result = comparer(detections_ref, detections_insp, W, H)
        anomalies_raw = anomalies_to_dict(opencv_result["anomalies"])
        score_aoi     = opencv_result["score_aoi"]
        log(f"[OPENCV] {len(anomalies_raw)} anomalie(s) — Score AOI: {score_aoi}/100")

        # ── ÉTAPE 3 : Gemini (2 passes) ───────────────────────────────────────
        log("[GEMINI] Analyse visuelle passe 1...")
        rapport_gemini = await expliquer(
            api_key          = gemini_key,
            image_ref        = img_ref,
            image_insp       = img_insp,
            anomalies_opencv = anomalies_raw,
            detections_ref   = detections_ref,
            detections_insp  = detections_insp,
            chain_context    = chain_context
        )
        log(f"[GEMINI] Passe 2 terminée — Statut: {rapport_gemini.get('statut','?')} | Score Gemini: {rapport_gemini.get('score_gemini','?')}/100")

        # ── ÉTAPE 4 : Annotation des images ──────────────────────────────────
        log("[ANNOTATOR] Génération des images annotées...")
        anomalies_finales = rapport_gemini.get("anomalies_enrichies", anomalies_raw)

        img_insp_annotee = annoter_image(
            img_insp,
            anomalies_finales,
            detections_yolo = detections_insp,
            show_yolo       = True
        )
        insp_b64 = image_to_base64(img_insp_annotee)

        ref_b64 = None
        if img_ref is not None:
            img_ref_annotee = annoter_reference(img_ref, detections_ref)
            ref_b64 = image_to_base64(img_ref_annotee)

        # ── ÉTAPE 5 : Webhook Make ────────────────────────────────────────────
        webhook_result = None
        if webhook_url.strip():
            log("[MAKE] Envoi webhook...")
            webhook_result = await envoyer_rapport(
                webhook_url      = webhook_url,
                batch_id         = batch_id or f"LOT-{int(asyncio.get_event_loop().time())}",
                fichier_name     = image_inspect.filename or "inconnu",
                chain_context    = chain_context,
                detections_ref   = detections_ref,
                detections_insp  = detections_insp,
                anomalies_opencv = anomalies_raw,
                rapport_gemini   = rapport_gemini,
                score_aoi        = score_aoi,
                image_insp_b64   = insp_b64,
                image_ref_b64    = ref_b64
            )
            status = "OK" if webhook_result.get("ok") else f"ERREUR — {webhook_result.get('error')}"
            log(f"[MAKE] Webhook {status}")
        else:
            log("[MAKE] Pas d'URL webhook — envoi ignoré")

        # ── Réponse ───────────────────────────────────────────────────────────
        return JSONResponse({
            "statut":         rapport_gemini.get("statut", "nok"),
            "score_aoi":      score_aoi,
            "score_gemini":   rapport_gemini.get("score_gemini", score_aoi),
            "message":        rapport_gemini.get("message", ""),
            "recommandation": rapport_gemini.get("recommandation", ""),
            "description_visuelle": rapport_gemini.get("description_visuelle", ""),

            "yolo": {
                "nb_ref":         len(detections_ref),
                "nb_insp":        len(detections_insp),
                "detections_ref":  detections_ref,
                "detections_insp": detections_insp
            },

            "anomalies_opencv": anomalies_raw,
            "anomalies":        anomalies_finales,

            "images": {
                "inspection_annotee": insp_b64,
                "reference_annotee":  ref_b64
            },

            "webhook": webhook_result,
            "logs":    logs
        })

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        log(f"[ERREUR] {e}")
        raise HTTPException(500, detail={"error": str(e), "traceback": tb})



@app.get("/")
async def frontend():
    return FileResponse(BASE_DIR / "../frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
