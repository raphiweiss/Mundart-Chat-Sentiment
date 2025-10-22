# ===== Mundart Basis-Datensatz (sauber & einfach) =====
import re
import random
import numpy as np
import pandas as pd

# -----------------------------
# Teil 1: Basis-Datensatz
# -----------------------------
def make_mundart_base(n_per_class=300, seed=42, include_emojis=True):
    rng = np.random.default_rng(seed)

    # Kurze, klare Seeds – neutral gehaltene Mundart/Umgangssprache
    POS = [
        "mega guet", "sehr guet", "top sache", "gfallt mir", "bin zfriede",
        "cool gmacht", "stark", "hammer", "tiptop", "empfehlenswert",
        "gute idee", "scheen", "liebs", "passt perfekt", "nice"
    ]
    NEG = [
        "mega blöd", "so en quatsch", "schlecht", "nervt", "funktioniert nid",
        "enttäuschend", "gar nöd guet", "schwach", "peinlich", "nie wieder",
        "tüür", "schlächt", "unnütz", "katastrophe", "schrott", "mist"
    ]
    NEU = [
        "ok", "passt so", "so lala", "geht so", "neutral gseh",
        "zur kenntnis gnomme", "ist okay", "standard", "kann man machen", "jo passt scho",
        "mal luege", "nichts speziells", "grad so i.o.", "sachlich ok", "unentschieden"
    ]

    POS_EMO = ["😊","😍","👍","✨","🎉","❤️"]
    NEG_EMO = ["😡","😤","👎","😭","💔"]
    NEU_EMO = ["🤔","😐","🙂"]

    def sample(pool, k):  # mit Replacement
        return [str(rng.choice(pool)) for _ in range(k)]

    def maybe_emoji(label, txt):
        if not include_emojis:
            return txt
        # moderat: nur etwa 40% der Fälle, jeweils 1 Emoji
        if rng.random() < 0.4:
            if label == "positiv":
                emo = rng.choice(POS_EMO)
            elif label == "negativ":
                emo = rng.choice(NEG_EMO)
            else:
                emo = rng.choice(NEU_EMO)
            return f"{txt} {emo}"
        return txt

    pos = [maybe_emoji("positiv", t) for t in sample(POS, n_per_class)]
    neg = [maybe_emoji("negativ", t) for t in sample(NEG, n_per_class)]
    neu = [maybe_emoji("neutral", t) for t in sample(NEU, n_per_class)]

    df = pd.DataFrame(
        {"text": pos + neg + neu,
         "label": (["positiv"]*n_per_class + ["negativ"]*n_per_class + ["neutral"]*n_per_class)}
    )

    # Deduplizieren & mischen (stabil)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df

# erzeugen & speichern
df_base = make_mundart_base(n_per_class=300, seed=42, include_emojis=True)
df_base.to_csv("mundart_base.csv", index=False)
print("Gespeichert: mundart_base.csv | Zeilen:", len(df_base))
print(df_base["label"].value_counts())
print(df_base.head(10).to_string(index=False))


# ===== Halbsynthetische Augmentierung für Mundart-Kurzmeinungen =====

# --- Konfiguration: Wahrscheinlichkeiten & Bausteine ---
AUG_CFG = {
    "p_dialect": 0.6,      # Wörter → Dialektvariante ersetzen
    "p_intens":  0.40,     # Intensifier voranstellen
    "p_filler":  0.25,     # Füllwort irgendwo einfügen
    "p_irony":   0.20,     # Ironie-Marker (🙄/lol/…)
    "p_emoji":   0.60,     # Emojis passend zum Label (etwas reduziert)
    "p_typos":   0.25,     # kleine Tippfehler (Swap/Double)
    "p_elong":   0.15,     # Dehnungen (z.B. meega, guuut)
}

INTENSIFIERS = ["mega", "huere", "voll", "richtig", "extrem", "eifach"]
FILLERS      = ["halt", "mal", "eh", "grad", "einfach", "eigentlich"]
IRONY_MARKER = ["🙄", "😉", "lol", "haha", "…", "🤨"]

POS_EMO = ["😊","😍","😎","👍","✨","🎉","❤️","😂","🤣"]
NEG_EMO = ["😤","😡","😢","👎","🤮","💔","😭","🤬"]
NEU_EMO = ["🤔","😶","🙃","😐","🙂"]

# Häufige Normalformen → Dialektvarianten (erweiterbar)
DIALECT_MAP = {
    "nicht": ["nid","ned","nöd"],
    "ist":   ["isch"],
    "bist":  ["bisch"],
    "sehr":  ["huere","mega","richtig"],
    "gut":   ["guet","guetli"],
    "blöd":  ["bloed","bloöd","blööd"],
    "schon": ["scho","shon","schoo"],
    "gesehen":["gseh"],
    "gemacht":["gmacht"],
}

# --- Emoji-Erkennung & -Kappung (max. 2 insgesamt) ---
# Grobe Emoji-Spanne: Misc Symbols + Emoticons + Supplemental Symbols/Pictographs
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")

def _cap_emojis(text, max_emojis=2):
    out, cnt = [], 0
    for ch in text:
        if EMOJI_RE.match(ch):
            if cnt < max_emojis:
                out.append(ch)
                cnt += 1
            else:
                # überschüssige Emojis verwerfen
                continue
        else:
            out.append(ch)
    # Doppelleerzeichen bereinigen
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()

# --- Hilfsfunktionen ---
def _inject_dialect(text, rng, p=AUG_CFG["p_dialect"]):
    def repl(m):
        w = m.group(0)
        key = w.lower()
        if key in DIALECT_MAP and rng.random() < p:
            v = rng.choice(DIALECT_MAP[key])
            # respektiere Gross-/Kleinschreibung
            return v.capitalize() if w[0].isupper() else v
        return w
    return re.sub(r"[A-Za-zÄÖÜäöüß]+", repl, text)

def _add_intensifier(text, rng, p=AUG_CFG["p_intens"]):
    if rng.random() < p:
        return rng.choice(INTENSIFIERS) + " " + text
    return text

def _add_filler(text, rng, p=AUG_CFG["p_filler"]):
    if rng.random() >= p:
        return text
    toks = text.split()
    if not toks:
        return text
    pos = int(rng.integers(0, len(toks)+1))
    toks.insert(pos, rng.choice(FILLERS))
    return " ".join(toks)

def _add_irony(text, rng, p=AUG_CFG["p_irony"]):
    if rng.random() < p:
        # meist ans Satzende
        return (text + " " + rng.choice(IRONY_MARKER)).strip()
    return text

def _add_emojis_by_label(text, label, rng, p=AUG_CFG["p_emoji"]):
    if rng.random() >= p:
        return text

    # Standard: 1 Emoji, gelegentlich 2
    def k12():
        return 1 if rng.random() < 0.8 else 2

    if label == "positiv":
        k = k12()
        emo = rng.choice(POS_EMO, size=k, replace=False)
    elif label == "negativ":
        k = k12()
        emo = rng.choice(NEG_EMO, size=k, replace=False)
    else:  # neutral
        emo = rng.choice(NEU_EMO, size=1, replace=False)

    return (text + " " + " ".join(emo)).strip()

def _typo_noise(text, rng, p=AUG_CFG["p_typos"], p_elong=AUG_CFG["p_elong"]):
    def typo_token(t):
        if len(t) < 4 or rng.random() >= p:
            return t
        # Swap zweier Nachbarn
        i = int(rng.integers(1, len(t)-1))
        t = t[:i-1] + t[i] + t[i-1] + t[i+1:]
        # optional Double
        if len(t) >= 5 and rng.random() < 0.35:
            j = int(rng.integers(1, len(t)))
            t = t[:j] + t[j-1] + t[j:]
        return t

    def elongate_token(t):
        if len(t) < 3 or rng.random() >= p_elong:
            return t
        i = int(rng.integers(1, len(t)-1))
        return t[:i] + t[i] + t[i] + t[i+1:]  # Dehnung

    parts = re.split(r"(\W+)", text)
    out = []
    for pce in parts:
        if pce.isalpha():
            pce = typo_token(pce)
            pce = elongate_token(pce)
        out.append(pce)
    return "".join(out)

def augment_one(text, label, rng):
    # Reihenfolge so gewählt, dass es natürlich wirkt
    t = text.strip()
    t = _inject_dialect(t, rng)                # Dialektvarianten
    t = _add_intensifier(t, rng)               # Verstärker vorn
    t = _add_filler(t, rng)                    # Füllwort an zufälliger Stelle
    t = _typo_noise(t, rng)                    # leichte Tippfehler/Dehnungen
    t = _add_irony(t, rng)                     # Ironiemarker (🙄/lol/…)
    t = _add_emojis_by_label(t, label, rng)    # passende Emojis ans Ende
    t = _cap_emojis(t, max_emojis=2)           # harte Obergrenze: max. 2 Emojis
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def augment_dataset(df, factor=3, seed=42):
    """
    Erzeugt pro Originalzeile 'factor' zusätzliche Varianten (Labels bleiben gleich).
    Entfernt exakte Duplikate, balanciert Klassen, mischt am Ende durch.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for text, label in zip(df["text"].astype(str), df["label"]):
        rows.append((text, label))  # Original
        for _ in range(factor):
            rows.append((augment_one(text, label, rng), label))

    df_aug = pd.DataFrame(rows, columns=["text","label"])
    before = len(df_aug)

    # Deduplizieren
    df_aug = df_aug.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Klassen balancieren (clip to min class size)
    cls_counts = df_aug["label"].value_counts()
    m = cls_counts.min()
    df_bal = (
        df_aug.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), m),
                                        random_state=int(rng.integers(1_000_000_000))))
    )
    df_bal = df_bal.sample(frac=1.0, random_state=int(rng.integers(1_000_000_000))).reset_index(drop=True)
    print(f"Augment: {before} → {len(df_bal)} Zeilen (balanciert: {m} je Klasse). factor={factor}")
    return df_bal

# --- Augmentieren & speichern ---
df = pd.read_csv("mundart_base.csv")
df_aug = augment_dataset(df, factor=3, seed=42)
df_aug.to_csv("mundart_augmented.csv", index=False)
print("Gespeichert: mundart_augmented.csv | Zeilen:", len(df_aug))
print(df_aug["label"].value_counts())
print(df_aug.head(10).to_string(index=False))
