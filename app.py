# ✅ NUTRISNAP — AI Food Recognition & Nutrition Tracker
# Fixed version with session_state for persistent feedback buttons
# 100% Free · No API Key · MobileNet Food101 · Indian Food Database

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import json, os
from datetime import datetime, date, timedelta
from transformers import pipeline

st.set_page_config(
    page_title="NutriSnap",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif !important; }
.header-box {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 28px 32px; border-radius: 20px; color: white;
    margin-bottom: 24px; box-shadow: 0 8px 32px rgba(17,153,142,0.3);
}
.meal-card {
    background: white; border-radius: 16px; padding: 20px;
    margin: 10px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    border-left: 5px solid #11998e;
}
.food-item-card {
    background: #f8fffe; border-radius: 12px; padding: 14px 18px;
    margin: 6px 0; border-left: 4px solid #11998e;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.feedback-box {
    background: #f0f8ff; border-radius: 14px; padding: 16px 20px;
    margin: 12px 0; border: 1.5px dashed #11998e;
}
.correction-badge {
    background: linear-gradient(135deg,#7c4dff,#e040fb);
    color: white; padding: 4px 12px; border-radius: 100px;
    font-size: 11px; font-weight: 700; display: inline-block;
}
.healthy  { background:linear-gradient(135deg,#56ab2f,#a8e063);
            color:white; padding:4px 12px; border-radius:100px;
            font-size:12px; font-weight:700; display:inline-block; }
.moderate { background:linear-gradient(135deg,#f7971e,#ffd200);
            color:#333; padding:4px 12px; border-radius:100px;
            font-size:12px; font-weight:700; display:inline-block; }
.junk     { background:linear-gradient(135deg,#cb2d3e,#ef473a);
            color:white; padding:4px 12px; border-radius:100px;
            font-size:12px; font-weight:700; display:inline-block; }
.stButton > button {
    background: linear-gradient(135deg,#11998e,#38ef7d) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f2027,#203a43,#2c5364) !important;
}
div[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    MODELS = [
        "nateraw/food",
        "Kaludi/food-category-classification-v2.0",
        "microsoft/resnet-50",
    ]
    for m in MODELS:
        try:
            clf = pipeline("image-classification", model=m, top_k=5)
            print(f"✅ Loaded: {m}")
            return clf
        except Exception as e:
            print(f"❌ {m}: {e}")
    raise Exception("No model loaded!")


# ══════════════════════════════════════════════════════════════════
# FOOD DATABASE
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_db():
    return pd.read_csv("food_database.csv")



# ══════════════════════════════════════════════════════════════════
# ONLINE NUTRITION FETCH — searches web if food not in DB
# ══════════════════════════════════════════════════════════════════
def fetch_nutrition_online(food_name: str):
    """
    Fetch nutrition info from Open Food Facts API (free, no key needed).
    Falls back to reasonable defaults if not found.
    """
    try:
        import urllib.request, urllib.parse, urllib.error
        query   = urllib.parse.quote(food_name.strip())
        url     = (
            f"https://world.openfoodfacts.org/cgi/search.pl"
            f"?search_terms={query}&search_simple=1"
            f"&action=process&json=1&page_size=1"
        )
        req  = urllib.request.Request(url,
                   headers={"User-Agent": "NutriSnap/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        products = data.get("products", [])
        if not products:
            return None

        p    = products[0]
        nutr = p.get("nutriments", {})

        cal     = nutr.get("energy-kcal_100g") or \
                  nutr.get("energy_100g", 0) / 4.184
        protein = nutr.get("proteins_100g", 0)
        carbs   = nutr.get("carbohydrates_100g", 0)
        fats    = nutr.get("fat_100g", 0)
        fiber   = nutr.get("fiber_100g", 0)

        cal = round(float(cal or 0), 1)

        # Simple health score formula
        score = 50
        if protein > 10  : score += 10
        if fiber   > 3   : score += 10
        if fats    < 5   : score += 10
        if carbs   < 20  : score += 10
        if cal     > 400 : score -= 20
        if fats    > 20  : score -= 15
        score = max(0, min(100, score))

        cls = ("Healthy"  if score >= 70 else
               "Junk"     if score <  40 else "Moderate")

        return {
            "indian_name"       : food_name.strip().title(),
            "food_name"         : food_name.strip().lower().replace(" ","_"),
            "classification"    : cls,
            "health_score"      : score,
            "calories_per_100g" : cal,
            "protein_g"         : round(float(protein or 0), 1),
            "carbs_g"           : round(float(carbs   or 0), 1),
            "fats_g"            : round(float(fats    or 0), 1),
            "fiber_g"           : round(float(fiber   or 0), 1),
            "tip"               : f"Nutrition fetched online for {food_name.title()}.",
            "category"          : "Online",
            "_was_corrected"    : True,
            "_original_ai_label": "",
            "_correction_count" : 1,
            "_source"           : "Open Food Facts",
        }
    except Exception:
        return None


FEEDBACK_FILE = "/tmp/user_feedback.json"

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE) as f:
            return json.load(f)
    return {
        "corrections"  : {},
        "confirmations": {},
        "total_corrections"  : 0,
        "total_confirmations": 0,
    }

def save_feedback(data):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

def add_correction(ai_label: str, correct_label: str):
    data = load_feedback()
    if ai_label not in data["corrections"]:
        data["corrections"][ai_label] = {}
    prev = data["corrections"][ai_label].get(correct_label, 0)
    data["corrections"][ai_label][correct_label] = prev + 1
    data["total_corrections"] = data.get("total_corrections", 0) + 1
    save_feedback(data)

def add_confirmation(ai_label: str):
    data = load_feedback()
    data["confirmations"][ai_label] = \
        data["confirmations"].get(ai_label, 0) + 1
    data["total_confirmations"] = \
        data.get("total_confirmations", 0) + 1
    save_feedback(data)

def apply_corrections(ai_label: str):
    data = load_feedback()
    corrections = data["corrections"].get(ai_label, {})
    if corrections:
        best_correction = max(corrections, key=corrections.get)
        count           = corrections[best_correction]
        return best_correction, True, count
    return ai_label, False, 0

def get_correction_stats():
    data = load_feedback()
    return {
        "total_corrections"  : data.get("total_corrections",   0),
        "total_confirmations": data.get("total_confirmations", 0),
        "unique_corrected"   : len(data.get("corrections",     {})),
    }


# ══════════════════════════════════════════════════════════════════
# NUTRITION LOOKUP
# ══════════════════════════════════════════════════════════════════
def get_nutrition(label: str, db: pd.DataFrame):
    label = label.replace("food101_", "").strip().lower()
    corrected, was_corrected, corr_count = apply_corrections(label)
    original_label = label
    if was_corrected:
        label = corrected

    match = db[db["food_name"] == label]

    if match.empty:
        for _, row in db.iterrows():
            if label in str(row["food_name"]) or \
               str(row["food_name"]) in label:
                match = db[db["food_name"] == row["food_name"]]
                break

    if match.empty:
        words = [w for w in label.split("_") if len(w) > 3]
        for word in words:
            for _, row in db.iterrows():
                if word in str(row["food_name"]):
                    match = db[db["food_name"] == row["food_name"]]
                    break
            if not match.empty:
                break

    if match.empty:
        return {
            "indian_name"       : label.replace("_"," ").title(),
            "classification"    : "Moderate",
            "health_score"      : 55,
            "calories_per_100g" : 200,
            "protein_g"         : 6.0,
            "carbs_g"           : 28.0,
            "fats_g"            : 7.0,
            "fiber_g"           : 2.0,
            "tip"               : "Eat in moderation.",
            "category"          : "General",
            "_was_corrected"    : was_corrected,
            "_original_ai_label": original_label,
            "_correction_count" : corr_count,
        }

    result = match.iloc[0].to_dict()
    result["_was_corrected"]     = was_corrected
    result["_original_ai_label"] = original_label
    result["_correction_count"]  = corr_count
    return result


# ══════════════════════════════════════════════════════════════════
# MULTI-FOOD / THALI DETECTION
# ══════════════════════════════════════════════════════════════════
def detect_multiple_foods(img, model, db, grid_size=3):
    img_rgb = img.convert("RGB")
    w, h    = img_rgb.size
    rw, rh  = w // grid_size, h // grid_size
    all_preds = {}

    for row in range(grid_size):
        for col in range(grid_size):
            x1     = col * rw
            y1     = row * rh
            region = img_rgb.crop((x1, y1, x1+rw, y1+rh))
            if region.width < 50 or region.height < 50:
                continue
            try:
                preds = model(region)
                if not preds: continue
                top   = preds[0]
                label = top["label"]
                conf  = round(top["score"]*100, 1)
                if conf < 15: continue
                nutrition = get_nutrition(label, db)
                if label not in all_preds or \
                   conf > all_preds[label]["confidence"]:
                    all_preds[label] = {
                        "label"    : label,
                        "nutrition": nutrition,
                        "confidence": conf,
                        "bbox"     : (x1, y1, x1+rw, y1+rh),
                    }
            except:
                continue

    seen, unique = set(), []
    for label, item in sorted(all_preds.items(),
                               key=lambda x: -x[1]["confidence"]):
        name = item["nutrition"]["indian_name"]
        if name not in seen:
            seen.add(name)
            unique.append(item)
    return unique


def draw_detections(img, foods):
    from PIL import ImageDraw
    out   = img.convert("RGB").copy()
    draw  = ImageDraw.Draw(out)
    clrs  = ["#11998e","#f7971e","#cb2d3e","#7c4dff",
              "#0288d1","#c62828","#2e7d32","#f57f17"]
    for i, item in enumerate(foods):
        x1,y1,x2,y2 = item["bbox"]
        c    = clrs[i % len(clrs)]
        name = item["nutrition"]["indian_name"]
        conf = item["confidence"]
        draw.rectangle([x1,y1,x2,y2], outline=c, width=3)
        txt = f"{name[:14]} {conf}%"
        draw.rectangle([x1,y1,x1+len(txt)*7+8,y1+22], fill=c)
        draw.text((x1+4, y1+3), txt, fill="white")
    return out


# ══════════════════════════════════════════════════════════════════
# DATA STORAGE
# ══════════════════════════════════════════════════════════════════
LOG = "/tmp/food_log.json"

def load_log():
    if os.path.exists(LOG):
        with open(LOG) as f: return json.load(f)
    return {"meals": []}

def save_log(d):
    with open(LOG, "w") as f: json.dump(d, f, indent=2)

def add_meal(m):
    d = load_log(); d["meals"].append(m); save_log(d)

def get_df():
    d = load_log()
    return pd.DataFrame(d["meals"]) if d["meals"] else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════
def gauge(score, title="Health Score"):
    color = "#56ab2f" if score>=70 else "#f7971e" if score>=40 else "#cb2d3e"
    fig   = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text":title,"font":{"size":16,"family":"Poppins"}},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":color},
            "steps":[
                {"range":[0,40],"color":"#ffe0e0"},
                {"range":[40,70],"color":"#fff3cd"},
                {"range":[70,100],"color":"#d4edda"},
            ],
        }
    ))
    fig.update_layout(height=220,margin=dict(t=40,b=0,l=20,r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def badge(cls):
    m = {"Healthy":"healthy","Moderate":"moderate","Junk":"junk"}
    i = {"Healthy":"✅","Moderate":"⚠️","Junk":"🚫"}
    return f'<span class="{m.get(cls,"moderate")}">{i.get(cls,"⚠️")} {cls}</span>'


# ══════════════════════════════════════════════════════════════════
# SESSION STATE INIT — FIXES FEEDBACK BUTTONS DISAPPEARING
# ══════════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "nutrition"      : None,
        "raw_label"      : None,
        "conf"           : None,
        "preds"          : None,
        "show_correction": False,
        "meal_saved"     : False,
        "confirmed"      : False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🥗 NutriSnap")
    st.markdown("**✅ 100% Free — No API Key!**")
    st.markdown("*Learns from your corrections!*")
    st.markdown("---")

    page = st.radio("Navigate", [
        "📸 Log Meal",
        "📊 Daily Report",
        "📅 Weekly Report",
        "🏆 Monthly Report",
        "📋 History",
        "🧠 AI Learning Stats",
    ])

    st.markdown("---")
    df_s = get_df()
    if not df_s.empty:
        st.markdown("### 📌 Quick Stats")
        st.metric("Total Meals",   len(df_s))
        st.metric("Today's Meals",
                  len(df_s[df_s["date"]==date.today().isoformat()]))
        if "health_score" in df_s.columns:
            st.metric("Avg Score",
                      f"{df_s['health_score'].mean():.1f}/100")
    else:
        st.info("No meals yet!")

    stats = get_correction_stats()
    if stats["total_corrections"] > 0 or \
       stats["total_confirmations"] > 0:
        st.markdown("---")
        st.markdown("### 🧠 AI Learning")
        st.metric("Corrections taught", stats["total_corrections"])
        st.metric("Confirmed correct",  stats["total_confirmations"])
        st.metric("Foods corrected",    stats["unique_corrected"])

    st.markdown("---")
    if st.button("🗑️ Clear Meal Data"):
        save_log({"meals": []})
        st.success("Meals cleared!")
        st.rerun()
    if st.button("🔄 Reset AI Learning"):
        if os.path.exists(FEEDBACK_FILE):
            os.remove(FEEDBACK_FILE)
        st.success("AI learning reset!")
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-box">
  <h1 style="margin:0;font-size:2.2em;font-weight:800">
    🥗 NutriSnap
  </h1>
  <p style="margin:6px 0 0;opacity:.9">
    Upload meal photos → AI identifies → Tracks health →
    <b>Learns from your corrections!</b><br>
    ✅ 100% Free · No API Key · Gets smarter every day
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: LOG MEAL
# ══════════════════════════════════════════════════════════════════
if page == "📸 Log Meal":
    db = load_db()

    with st.spinner("⚡ Loading AI model..."):
        model = load_model()
    st.success("✅ AI model ready!")

    mode = st.radio(
        "Detection Mode:",
        ["🍽️ Single Food", "🥘 Multiple Foods / Thali"],
        horizontal=True
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        meal_type = st.selectbox("🍽️ Meal Type",
            ["Breakfast","Lunch","Evening Snack","Dinner","Other"])
        meal_date = st.date_input("📅 Date", value=date.today())
        notes     = st.text_input("📝 Notes",
                        placeholder="e.g. homemade / restaurant...")

        if "Thali" in mode or "Multiple" in mode:
            grid_size = st.slider("🔲 Grid Size",
                min_value=2, max_value=4, value=3,
                help="3×3 scans 9 regions")
            st.info(f"Scanning {grid_size}×{grid_size} = "
                    f"{grid_size*grid_size} regions")
        else:
            grid_size = 1

        uploaded = st.file_uploader("📷 Upload Food Photo",
                    type=["jpg","jpeg","png","webp"])
        if uploaded:
            st.image(uploaded, use_container_width=True)

    with c2:
        if uploaded:
            if st.button("🤖 Identify Food with AI",
                         use_container_width=True):
                img = Image.open(uploaded).convert("RGB")

                # ── SINGLE FOOD MODE ──────────────────────────────
                if "Single" in mode:
                    with st.spinner("Identifying... ⚡"):
                        try:
                            preds = model(img)
                            top   = preds[0]
                            # ✅ Store in session_state so it persists
                            st.session_state.raw_label  = top["label"]
                            st.session_state.conf       = round(top["score"]*100, 1)
                            st.session_state.nutrition  = get_nutrition(
                                st.session_state.raw_label, db)
                            st.session_state.preds      = preds
                            st.session_state.meal_saved = False
                            st.session_state.confirmed  = False
                            st.session_state.show_correction = False
                        except Exception as e:
                            st.error(f"Error: {e}")

                # ── MULTIPLE FOOD / THALI MODE ────────────────────
                else:
                    with st.spinner(
                        f"Scanning {grid_size}×{grid_size} regions... 🔍"
                    ):
                        try:
                            foods = detect_multiple_foods(
                                img, model, db, grid_size)

                            if not foods:
                                st.warning(
                                    "No food detected. Try a clearer photo!")
                            else:
                                annotated = draw_detections(img, foods)
                                st.image(
                                    annotated,
                                    caption=f"✅ {len(foods)} food(s) detected!",
                                    use_container_width=True
                                )
                                st.markdown(f"### 🍱 Found {len(foods)} items!")

                                tot_cal = tot_pro = 0
                                tot_carbs = tot_fats = 0
                                scores, all_cls = [], []

                                for i, item in enumerate(foods):
                                    n   = item["nutrition"]
                                    cls = n["classification"]
                                    cc  = {
                                        "Healthy" : "#56ab2f",
                                        "Moderate": "#f7971e",
                                        "Junk"    : "#cb2d3e"
                                    }.get(cls, "#888")

                                    corr_note = ""
                                    if n.get("_was_corrected"):
                                        corr_note = (
                                            f'<span class="correction-badge">'
                                            f'🧠 Corrected</span>'
                                        )

                                    st.markdown(f"""
                                    <div class="food-item-card">
                                      <div style="display:flex;
                                            justify-content:space-between;
                                            align-items:center">
                                        <div>
                                          <b>{i+1}. {n['indian_name']}</b>
                                          &nbsp;{badge(cls)}
                                          &nbsp;{corr_note}
                                        </div>
                                        <div style="font-size:.85em;color:{cc}">
                                          Score: <b>{n['health_score']}/100</b>
                                          &nbsp;🔥 {n['calories_per_100g']} kcal
                                        </div>
                                      </div>
                                      <div style="font-size:.82em;
                                                  color:#555;margin-top:6px">
                                        💪 {n['protein_g']}g &nbsp;
                                        🌾 {n['carbs_g']}g &nbsp;
                                        🧈 {n['fats_g']}g
                                      </div>
                                      <div style="font-size:.8em;
                                                  color:#11998e;margin-top:4px">
                                        💡 {n['tip']}
                                      </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    tot_cal   += n["calories_per_100g"]
                                    tot_pro   += n["protein_g"]
                                    tot_carbs += n["carbs_g"]
                                    tot_fats  += n["fats_g"]
                                    scores.append(n["health_score"])
                                    all_cls.append(cls)

                                avg_score = round(sum(scores)/len(scores), 1)

                                st.markdown("---")
                                st.markdown("### 📊 Combined Summary")
                                s1,s2,s3,s4,s5 = st.columns(5)
                                s1.metric("🏆 Score",   f"{avg_score}/100")
                                s2.metric("🔥 Cal",     int(tot_cal))
                                s3.metric("💪 Protein", f"{tot_pro:.1f}g")
                                s4.metric("🌾 Carbs",   f"{tot_carbs:.1f}g")
                                s5.metric("🧈 Fats",    f"{tot_fats:.1f}g")

                                st.plotly_chart(
                                    gauge(avg_score, "Overall Score"),
                                    use_container_width=True
                                )

                                healthy_ct  = all_cls.count("Healthy")
                                junk_ct     = all_cls.count("Junk")
                                moderate_ct = all_cls.count("Moderate")

                                if junk_ct == 0 and healthy_ct >= moderate_ct:
                                    st.success(
                                        f"🌟 Great thali! "
                                        f"{healthy_ct} healthy · "
                                        f"{moderate_ct} moderate · 0 junk!"
                                    )
                                elif junk_ct > 0:
                                    st.warning(
                                        f"⚠️ {junk_ct} junk item(s). "
                                        f"Try swapping for healthy!"
                                    )

                                food_names = " + ".join(
                                    [f["nutrition"]["indian_name"] for f in foods]
                                )
                                add_meal({
                                    "date"          : meal_date.isoformat(),
                                    "meal_type"     : meal_type,
                                    "food_detected" : food_names[:80],
                                    "food101_label" : "multi_food",
                                    "classification": max(set(all_cls),
                                                          key=all_cls.count),
                                    "health_score"  : avg_score,
                                    "calories"      : int(tot_cal),
                                    "protein_g"     : round(tot_pro,1),
                                    "carbs_g"       : round(tot_carbs,1),
                                    "fats_g"        : round(tot_fats,1),
                                    "fiber_g"       : 0.0,
                                    "tip"           : f"{len(foods)} items detected",
                                    "notes"         : notes,
                                    "confidence"    : round(
                                        sum(f["confidence"] for f in foods)
                                        /len(foods), 1),
                                    "was_corrected" : False,
                                    "logged_at"     : datetime.now().isoformat(),
                                })
                                st.success(f"✅ {len(foods)} items saved!")

                        except Exception as e:
                            st.error(f"Error: {e}")

            # ══════════════════════════════════════════════════════
            # ✅ SHOW RESULTS FROM SESSION STATE — ALWAYS VISIBLE
            # ══════════════════════════════════════════════════════
            if st.session_state.nutrition is not None and "Single" in mode:
                nutrition = st.session_state.nutrition
                raw_label = st.session_state.raw_label
                conf      = st.session_state.conf
                preds     = st.session_state.preds

                was_corrected = nutrition.get("_was_corrected", False)
                orig_label    = nutrition.get("_original_ai_label", raw_label)
                corr_count    = nutrition.get("_correction_count", 0)

                # ✅ FIX: render card and correction note separately
                st.markdown(f"""
                <div class="meal-card">
                  <h3>🍽️ {nutrition['indian_name']}</h3>
                  <p>{badge(nutrition['classification'])}</p>
                  <p style="color:#666;font-size:.9em;margin-top:6px">
                    AI detected:
                    <b>{raw_label.replace('_',' ').title()}</b>
                    ({conf}% confident)
                  </p>
                </div>
                """, unsafe_allow_html=True)

                # ✅ FIX: show correction note as separate element
                if was_corrected:
                    st.markdown(
                        f'<div style="background:linear-gradient(135deg,#7c4dff22,#e040fb22);'
                        f'border-radius:10px;padding:10px 14px;margin:6px 0;'
                        f'border-left:4px solid #7c4dff">'
                        f'<span style="background:linear-gradient(135deg,#7c4dff,#e040fb);'
                        f'color:white;padding:3px 10px;border-radius:100px;'
                        f'font-size:11px;font-weight:700">'
                        f'🧠 Learned from your correction ({corr_count}x)</span><br>'
                        f'<small style="color:#555;margin-top:4px;display:block">'
                        f'AI said: <b>{orig_label.replace("_"," ").title()}</b> '
                        f'→ You taught: <b>{nutrition["indian_name"]}</b>'
                        f'</small></div>',
                        unsafe_allow_html=True
                    )

                if preds and len(preds) > 1:
                    with st.expander("🔍 Other possible foods"):
                        for p in preds[1:]:
                            n   = get_nutrition(p["label"], db)
                            pct = round(p["score"]*100, 1)
                            st.write(f"• **{n['indian_name']}** — {pct}%")

                st.plotly_chart(
                    gauge(nutrition["health_score"]),
                    use_container_width=True
                )

                n1,n2,n3,n4 = st.columns(4)
                n1.metric("🔥 Cal",    nutrition["calories_per_100g"])
                n2.metric("💪 Protein",f"{nutrition['protein_g']}g")
                n3.metric("🌾 Carbs",  f"{nutrition['carbs_g']}g")
                n4.metric("🧈 Fats",   f"{nutrition['fats_g']}g")

                cls = nutrition["classification"]
                tip = nutrition["tip"]
                if cls == "Healthy":
                    st.success(f"💡 {tip}")
                elif cls == "Moderate":
                    st.warning(f"💡 {tip}")
                else:
                    st.error(f"💡 {tip}")

                # ── SAVE MEAL BUTTON ──────────────────────────────
                if not st.session_state.meal_saved:
                    if st.button("💾 Save This Meal",
                                 use_container_width=True,
                                 key="save_meal_btn"):
                        add_meal({
                            "date"          : meal_date.isoformat(),
                            "meal_type"     : meal_type,
                            "food_detected" : str(nutrition["indian_name"]),
                            "food101_label" : raw_label,
                            "classification": str(nutrition["classification"]),
                            "health_score"  : int(nutrition["health_score"]),
                            "calories"      : int(nutrition["calories_per_100g"]),
                            "protein_g"     : float(nutrition["protein_g"]),
                            "carbs_g"       : float(nutrition["carbs_g"]),
                            "fats_g"        : float(nutrition["fats_g"]),
                            "fiber_g"       : float(nutrition["fiber_g"]),
                            "tip"           : str(nutrition["tip"]),
                            "notes"         : notes,
                            "confidence"    : conf,
                            "was_corrected" : was_corrected,
                            "logged_at"     : datetime.now().isoformat(),
                        })
                        st.session_state.meal_saved = True
                        st.success("✅ Meal saved!")
                else:
                    st.success("✅ Meal already saved!")

                # ── FEEDBACK BUTTONS ──────────────────────────────
                st.markdown("""
                <div class="feedback-box">
                  <b>🤔 Was this identification correct?</b>
                </div>
                """, unsafe_allow_html=True)

                fb1, fb2 = st.columns(2)

                with fb1:
                    if st.button("✅ Yes, correct!",
                                 key="confirm_btn",
                                 use_container_width=True):
                        add_confirmation(raw_label)
                        st.session_state.confirmed = True
                        st.session_state.show_correction = False

                with fb2:
                    if st.button("❌ No, wrong food!",
                                 key="wrong_btn",
                                 use_container_width=True):
                        st.session_state.show_correction = True
                        st.session_state.confirmed = False

                if st.session_state.confirmed:
                    st.success("🧠 Thanks! AI remembers this is correct.")

                # ── CORRECTION UI — STAYS VISIBLE ─────────────────
                if st.session_state.show_correction:
                    st.markdown("**🔍 What food is this actually?**")

                    # ── INPUT MODE TOGGLE ──────────────────────────
                    input_mode = st.radio(
                        "How do you want to enter the food?",
                        ["🔽 Pick from list", "⌨️ Type food name"],
                        horizontal=True,
                        key="input_mode_radio"
                    )

                    correct_choice = None
                    custom_nutrition = None

                    # ── MODE 1: DROPDOWN WITH SEARCH ───────────────
                    if input_mode == "🔽 Pick from list":
                        # Search filter
                        search_term = st.text_input(
                            "🔍 Filter list (type to search):",
                            placeholder="e.g. dal, rice, roti...",
                            key="search_filter"
                        )
                        all_foods = db["indian_name"].tolist()
                        if search_term:
                            filtered_foods = [
                                f for f in all_foods
                                if search_term.lower() in f.lower()
                            ]
                            if not filtered_foods:
                                st.warning(
                                    "No match in database. "
                                    "Try typing the food name directly!"
                                )
                                filtered_foods = all_foods
                        else:
                            filtered_foods = all_foods

                        correct_choice = st.selectbox(
                            f"Select ({len(filtered_foods)} foods shown):",
                            filtered_foods,
                            key="correction_select"
                        )

                    # ── MODE 2: TYPE CUSTOM FOOD NAME ──────────────
                    else:
                        st.info(
                            "💡 Type any food name — we'll search "
                            "nutrition online if not in database!"
                        )
                        custom_name = st.text_input(
                            "Food name:",
                            placeholder="e.g. Pav Bhaji, Momos, Biryani...",
                            key="custom_food_name"
                        )

                        if custom_name:
                            # Try exact match in DB first
                            db_match = db[
                                db["indian_name"].str.lower() ==
                                custom_name.strip().lower()
                            ]
                            partial_match = db[
                                db["indian_name"].str.lower().str.contains(
                                    custom_name.strip().lower(), na=False
                                )
                            ]

                            if not db_match.empty:
                                st.success(
                                    f"✅ Found in database: "
                                    f"{db_match.iloc[0]['indian_name']}"
                                )
                                correct_choice  = db_match.iloc[0]["indian_name"]
                                custom_nutrition = None

                            elif not partial_match.empty:
                                st.info(
                                    f"🔍 Similar found: "
                                    f"{partial_match.iloc[0]['indian_name']}"
                                )
                                use_similar = st.checkbox(
                                    f"Use '{partial_match.iloc[0]['indian_name']}' instead?",
                                    key="use_similar"
                                )
                                if use_similar:
                                    correct_choice   = partial_match.iloc[0]["indian_name"]
                                    custom_nutrition = None
                                else:
                                    correct_choice = None
                                    # Try online fetch
                                    if st.button("🌐 Search Nutrition Online",
                                                 key="search_online_partial"):
                                        with st.spinner(
                                            f"Searching nutrition for "
                                            f"'{custom_name}'... 🌐"
                                        ):
                                            result = fetch_nutrition_online(custom_name)
                                        if result:
                                            st.session_state["online_nutrition"] = result
                                            st.success(
                                                f"✅ Found online! "
                                                f"{result['calories_per_100g']} kcal · "
                                                f"{result['protein_g']}g protein"
                                            )
                                        else:
                                            st.session_state["online_nutrition"] = None
                                            st.warning(
                                                "Not found online either. "
                                                "Fill details manually below."
                                            )
                                    custom_nutrition = st.session_state.get(
                                        "online_nutrition"
                                    )

                            else:
                                # Not in DB — search online automatically
                                correct_choice = None
                                col_srch, _ = st.columns([1,2])
                                with col_srch:
                                    search_clicked = st.button(
                                        "🌐 Search Nutrition Online",
                                        key="search_online_btn"
                                    )
                                if search_clicked:
                                    with st.spinner(
                                        f"Searching nutrition for "
                                        f"'{custom_name}'... 🌐"
                                    ):
                                        result = fetch_nutrition_online(custom_name)
                                    if result:
                                        st.session_state["online_nutrition"] = result
                                    else:
                                        st.session_state["online_nutrition"] = "not_found"

                                online = st.session_state.get("online_nutrition")

                                if online and online != "not_found":
                                    st.success(
                                        f"✅ Found on Open Food Facts!"
                                    )
                                    o1,o2,o3,o4 = st.columns(4)
                                    o1.metric("🔥 Cal",
                                        online["calories_per_100g"])
                                    o2.metric("💪 Protein",
                                        f"{online['protein_g']}g")
                                    o3.metric("🌾 Carbs",
                                        f"{online['carbs_g']}g")
                                    o4.metric("🧈 Fats",
                                        f"{online['fats_g']}g")
                                    custom_nutrition = online

                                elif online == "not_found":
                                    st.warning(
                                        f"'{custom_name}' not found online. "
                                        f"Fill details manually:"
                                    )
                                    cls_choice = st.selectbox(
                                        "Health classification:",
                                        ["Healthy","Moderate","Junk"],
                                        key="custom_cls2"
                                    )
                                    score_choice = st.slider(
                                        "Health score (0-100):",
                                        0, 100, 55, key="custom_score2"
                                    )
                                    cal_choice = st.number_input(
                                        "Calories per 100g:",
                                        0, 1000, 200, key="custom_cal2"
                                    )
                                    custom_nutrition = {
                                        "indian_name"       : custom_name.strip().title(),
                                        "food_name"         : custom_name.strip().lower().replace(" ","_"),
                                        "classification"    : cls_choice,
                                        "health_score"      : score_choice,
                                        "calories_per_100g" : cal_choice,
                                        "protein_g"         : 6.0,
                                        "carbs_g"           : 28.0,
                                        "fats_g"            : 7.0,
                                        "fiber_g"           : 2.0,
                                        "tip"               : "Custom food added by user.",
                                        "_was_corrected"    : True,
                                        "_original_ai_label": raw_label,
                                        "_correction_count" : 1,
                                    }
                                else:
                                    custom_nutrition = None

                    # ── SAVE CORRECTION BUTTON ─────────────────────
                    if st.button("💾 Save Correction",
                                 key="save_correction"):

                        if custom_nutrition:
                            # Save custom food correction
                            add_correction(
                                raw_label,
                                custom_nutrition["food_name"]
                            )
                            st.session_state.nutrition = custom_nutrition
                            st.success(
                                f"🧠 Saved! '{custom_nutrition['indian_name']}' "
                                f"will be shown next time!"
                            )
                            st.session_state.show_correction = False
                            st.rerun()

                        elif correct_choice:
                            row = db[
                                db["indian_name"]==correct_choice
                            ].iloc[0]
                            add_correction(raw_label, row["food_name"])
                            st.success(
                                f"🧠 Saved! Next time AI sees "
                                f"'{raw_label.replace('_',' ').title()}' "
                                f"it will show '{correct_choice}'!"
                            )
                            st.session_state.show_correction = False
                            st.session_state.nutrition = get_nutrition(
                                raw_label, db)
                            st.rerun()
                        else:
                            st.warning(
                                "Please enter or select a food name first!"
                            )

        else:
            st.markdown("""
            <div style='text-align:center;padding:48px 20px;
                        background:#f0fff4;border-radius:16px;
                        border:2px dashed #11998e;margin-top:20px'>
              <div style='font-size:3.5em'>🍱</div>
              <h3 style='color:#11998e;margin-top:10px'>
                Upload any food photo!
              </h3>
              <p style='color:#666'>
                Single dish or full thali — AI detects everything<br>
                <b>Correct AI mistakes → App learns → Gets smarter!</b>
              </p>
              <p style='color:#11998e;font-weight:700'>
                ✅ No API Key · 100% Free!
              </p>
            </div>
            """, unsafe_allow_html=True)

        # Manual entry
        st.markdown("---")
        st.markdown("#### ✏️ Or add manually")
        manual = st.selectbox("Select food", db["indian_name"].tolist())
        if st.button("➕ Add this food"):
            r = db[db["indian_name"]==manual].iloc[0]
            add_meal({
                "date"          : meal_date.isoformat(),
                "meal_type"     : meal_type,
                "food_detected" : r["indian_name"],
                "food101_label" : r["food_name"],
                "classification": r["classification"],
                "health_score"  : int(r["health_score"]),
                "calories"      : int(r["calories_per_100g"]),
                "protein_g"     : float(r["protein_g"]),
                "carbs_g"       : float(r["carbs_g"]),
                "fats_g"        : float(r["fats_g"]),
                "fiber_g"       : float(r["fiber_g"]),
                "tip"           : r["tip"],
                "notes"         : notes,
                "confidence"    : 100,
                "was_corrected" : False,
                "logged_at"     : datetime.now().isoformat(),
            })
            st.success(f"✅ {manual} added!")


# ══════════════════════════════════════════════════════════════════
# DAILY REPORT
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Daily Report":
    st.subheader("📊 Daily Health Report")
    sel = st.date_input("Select Date", value=date.today())
    df  = get_df()
    if df.empty:
        st.info("No meals logged yet!")
    else:
        day = df[df["date"]==sel.isoformat()]
        if day.empty:
            st.warning(f"No meals on {sel.strftime('%B %d, %Y')}")
        else:
            avg = round(day["health_score"].mean(),1)
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("🏆 Score",    f"{avg}/100")
            m2.metric("🔥 Calories", int(day["calories"].sum()))
            m3.metric("💪 Protein",  f"{day['protein_g'].sum():.1f}g")
            m4.metric("✅ Healthy",
                      len(day[day["classification"]=="Healthy"]))
            m5.metric("🚫 Junk",
                      len(day[day["classification"]=="Junk"]))

            c1,c2 = st.columns(2)
            with c1:
                st.plotly_chart(gauge(avg,"Today's Score"),
                                use_container_width=True)
            with c2:
                fig = px.bar(day,x="meal_type",y="health_score",
                    color="classification",
                    color_discrete_map={
                        "Healthy":"#56ab2f",
                        "Moderate":"#f7971e",
                        "Junk":"#cb2d3e"},
                    title="Score by Meal",text="health_score")
                fig.update_layout(height=220,
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig,use_container_width=True)

            p1,p2 = st.columns(2)
            with p1:
                fig2 = px.pie(
                    values=[day["protein_g"].sum(),
                            day["carbs_g"].sum(),
                            day["fats_g"].sum()],
                    names=["Protein","Carbs","Fats"],
                    color_discrete_sequence=["#56ab2f","#f7971e","#cb2d3e"],
                    title="Macronutrients")
                fig2.update_layout(height=260,
                                   paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2,use_container_width=True)
            with p2:
                fig3 = px.pie(day,names="classification",
                    color="classification",
                    color_discrete_map={
                        "Healthy":"#56ab2f",
                        "Moderate":"#f7971e",
                        "Junk":"#cb2d3e"},
                    title="Classification")
                fig3.update_layout(height=260,
                                   paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig3,use_container_width=True)

            st.markdown("#### 🍽️ Today's Meals")
            for _,row in day.iterrows():
                corr_tag = " 🧠" if row.get("was_corrected") else ""
                with st.expander(
                    f"{row['meal_type']} — "
                    f"{str(row['food_detected'])[:50]}"
                    f"{corr_tag} | Score: {row['health_score']}/100"
                ):
                    a,b = st.columns(2)
                    with a:
                        st.write(f"**Food:** {row['food_detected']}")
                        st.write(f"**Calories:** {row['calories']} kcal")
                        st.write(f"**Protein:** {row['protein_g']}g")
                    with b:
                        st.write(f"**Carbs:** {row['carbs_g']}g")
                        st.write(f"**Fats:** {row['fats_g']}g")
                        st.write(f"**Fiber:** {row['fiber_g']}g")
                    if row.get("was_corrected"):
                        st.info("🧠 This result was improved by your correction!")
                    if row.get("tip"):
                        st.info(f"💡 {row['tip']}")


# ══════════════════════════════════════════════════════════════════
# WEEKLY REPORT
# ══════════════════════════════════════════════════════════════════
elif page == "📅 Weekly Report":
    st.subheader("📅 Weekly Report — Last 7 Days")
    df = get_df()
    if df.empty:
        st.info("No meals logged yet!")
    else:
        df["date"] = pd.to_datetime(df["date"])
        wk = df[df["date"] >= pd.Timestamp(date.today()-timedelta(days=7))]
        if wk.empty:
            st.warning("No meals in last 7 days.")
        else:
            daily = wk.groupby("date").agg(
                avg_score=("health_score","mean"),
                total_cal=("calories","sum"),
                count    =("meal_type","count")
            ).reset_index()
            daily["avg_score"] = daily["avg_score"].round(1)

            hp = round(len(wk[wk["classification"]=="Healthy"])/len(wk)*100)
            jp = round(len(wk[wk["classification"]=="Junk"])/len(wk)*100)

            w1,w2,w3,w4 = st.columns(4)
            w1.metric("📊 Weekly Score", f"{daily['avg_score'].mean():.1f}/100")
            w2.metric("🍽️ Total Meals",  len(wk))
            w3.metric("✅ Healthy %",    f"{hp}%")
            w4.metric("🚫 Junk %",       f"{jp}%")

            fig1 = px.line(daily,x="date",y="avg_score",
                title="📈 Daily Health Score Trend",
                markers=True,
                color_discrete_sequence=["#11998e"])
            fig1.add_hline(y=70,line_dash="dash",
                line_color="#56ab2f",annotation_text="Healthy (70)")
            fig1.add_hline(y=40,line_dash="dash",
                line_color="#cb2d3e",annotation_text="Junk (40)")
            fig1.update_layout(height=300,
                paper_bgcolor="rgba(0,0,0,0)",yaxis_range=[0,100])
            st.plotly_chart(fig1,use_container_width=True)

            fig2 = px.bar(daily,x="date",y="total_cal",
                title="🔥 Daily Calorie Intake",
                color_discrete_sequence=["#f7971e"])
            fig2.add_hline(y=2000,line_dash="dash",
                line_color="#333",annotation_text="2000 kcal recommended")
            fig2.update_layout(height=280,paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2,use_container_width=True)

            aw = daily["avg_score"].mean()
            if aw>=70:
                st.success(f"🌟 Excellent week! {aw:.1f}/100")
            elif aw>=50:
                st.warning(f"⚠️ Decent. {aw:.1f}/100 — add veggies!")
            else:
                st.error(f"🚨 Needs work. {aw:.1f}/100 — cut junk!")


# ══════════════════════════════════════════════════════════════════
# MONTHLY REPORT
# ══════════════════════════════════════════════════════════════════
elif page == "🏆 Monthly Report":
    st.subheader("🏆 Monthly Report — Last 30 Days")
    df = get_df()
    if df.empty:
        st.info("No meals logged yet!")
    else:
        df["date"] = pd.to_datetime(df["date"])
        mo = df[df["date"] >= pd.Timestamp(date.today()-timedelta(days=30))]
        if mo.empty:
            st.warning("No meals in last 30 days.")
        else:
            avg_mo = round(mo["health_score"].mean(),1)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("🏆 Score",  f"{avg_mo}/100")
            m2.metric("🍽️ Meals", len(mo))
            m3.metric("✅ Healthy",len(mo[mo["classification"]=="Healthy"]))
            m4.metric("🚫 Junk",   len(mo[mo["classification"]=="Junk"]))

            c1,c2 = st.columns(2)
            with c1:
                st.plotly_chart(gauge(avg_mo,"Monthly Score"),
                                use_container_width=True)
            with c2:
                cc  = mo["classification"].value_counts()
                fig = go.Figure(go.Pie(
                    labels=cc.index, values=cc.values, hole=0.5,
                    marker_colors=["#56ab2f","#f7971e","#cb2d3e"]))
                fig.update_layout(title="30-day Classification",
                    height=280,paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig,use_container_width=True)

            fc   = mo["food_detected"].value_counts().head(10)
            fig2 = px.bar(x=fc.values,y=fc.index,orientation="h",
                title="🍽️ Most Eaten Foods",
                color=fc.values,
                color_continuous_scale=["#cb2d3e","#f7971e","#56ab2f"])
            fig2.update_layout(height=360,paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2,use_container_width=True)

            st.markdown("### 🏅 Monthly Verdict")
            if avg_mo>=75:
                st.balloons()
                st.success("🌟 EXCELLENT MONTH!")
            elif avg_mo>=60:
                st.success("👍 GOOD MONTH! Reduce junk snacks!")
            elif avg_mo>=45:
                st.warning("📈 AVERAGE. One healthy swap daily!")
            else:
                st.error("⚠️ NEEDS IMPROVEMENT. Start small!")


# ══════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════
elif page == "📋 History":
    st.subheader("📋 Complete Meal History")
    df = get_df()
    if df.empty:
        st.info("No meals yet!")
    else:
        f1,f2 = st.columns(2)
        with f1:
            fcls = st.multiselect("Classification",
                ["Healthy","Moderate","Junk"],
                default=["Healthy","Moderate","Junk"])
        with f2:
            sort = st.selectbox("Sort by",[
                "Date (newest)",
                "Health Score (high)",
                "Calories (high)"
            ])

        filtered = df[df["classification"].isin(fcls)]
        sm = {
            "Date (newest)"      :("date",False),
            "Health Score (high)":("health_score",False),
            "Calories (high)"    :("calories",False)
        }
        c,a = sm[sort]
        filtered = filtered.sort_values(c,ascending=a)
        st.markdown(f"**{len(filtered)} meals**")

        for _,row in filtered.iterrows():
            corr_tag = " 🧠" if row.get("was_corrected") else ""
            with st.expander(
                f"{row['date']} | {row['meal_type']} | "
                f"{str(row['food_detected'])[:50]}"
                f"{corr_tag} | {row['health_score']}/100"
            ):
                a,b = st.columns(2)
                with a:
                    st.write(f"**Food:** {row['food_detected']}")
                    st.write(f"**Class:** {row['classification']}")
                    st.write(f"**Cal:** {row['calories']} kcal")
                with b:
                    st.write(f"**Protein:** {row['protein_g']}g")
                    st.write(f"**Carbs:** {row['carbs_g']}g")
                    st.write(f"**Fats:** {row['fats_g']}g")
                if row.get("was_corrected"):
                    st.info("🧠 Result corrected by you!")
                if row.get("tip"):
                    st.info(f"💡 {row['tip']}")

        st.download_button("📥 Download CSV",
            filtered.to_csv(index=False),
            "food_history.csv","text/csv")


# ══════════════════════════════════════════════════════════════════
# AI LEARNING STATS
# ══════════════════════════════════════════════════════════════════
elif page == "🧠 AI Learning Stats":
    st.subheader("🧠 AI Learning Stats")
    st.markdown("This page shows how the app has learned from your feedback.")

    data  = load_feedback()
    stats = get_correction_stats()

    m1,m2,m3 = st.columns(3)
    m1.metric("Total Corrections",     stats["total_corrections"])
    m2.metric("Confirmations",          stats["total_confirmations"])
    m3.metric("Unique foods corrected", stats["unique_corrected"])

    if data["corrections"]:
        st.markdown("### 📋 What AI Learned from Your Corrections")
        rows = []
        for ai_label, corrections in data["corrections"].items():
            for correct_label, count in corrections.items():
                db_match = load_db()
                match  = db_match[db_match["food_name"]==correct_label]
                indian = match.iloc[0]["indian_name"] \
                    if not match.empty else correct_label
                rows.append({
                    "AI originally said": ai_label.replace("_"," ").title(),
                    "You corrected to"  : indian,
                    "Times corrected"   : count,
                    "Trust level"       : "✅ Applied always"
                                          if count>=1 else "⏳ Learning..."
                })

        corrections_df = pd.DataFrame(rows)
        st.dataframe(corrections_df,use_container_width=True,hide_index=True)

        fig = px.bar(
            corrections_df,
            x="AI originally said",
            y="Times corrected",
            title="Most Frequently Corrected AI Predictions",
            color="Times corrected",
            color_continuous_scale=["#f7971e","#56ab2f"]
        )
        fig.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)

    else:
        st.info(
            "No corrections yet! Upload a food photo and correct "
            "the AI if it's wrong — it will learn from you!"
        )

    if data["confirmations"]:
        st.markdown("### ✅ Foods AI Got Right (Confirmed by You)")
        conf_data = [
            {"Food": k.replace("_"," ").title(), "Confirmed": v}
            for k,v in data["confirmations"].items()
        ]
        conf_df = pd.DataFrame(conf_data).sort_values(
            "Confirmed", ascending=False)
        st.dataframe(conf_df,use_container_width=True,hide_index=True)

    st.markdown("---")
    st.markdown("### 💡 How AI Learning Works in This App")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **When you click ❌ Wrong food:**
        1. App saves: *"when AI says X, show Y instead"*
        2. Stored in `/tmp/user_feedback.json`
        3. Next upload → checks this file first
        4. Shows YOUR correction, not AI's guess
        """)
    with col2:
        st.markdown("""
        **When you click ✅ Correct:**
        1. App saves: *"AI got this right!"*
        2. Builds confidence over time
        3. Helps understand which foods AI knows well
        4. Your confirmation history shown here
        """)

    st.success(
        "🧠 The more you correct the AI, the smarter it gets "
        "for YOUR specific food habits and region!"
    )