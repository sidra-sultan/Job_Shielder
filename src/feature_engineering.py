import re
import pandas as pd
NEG = re.compile(r"\b(no|never|without|not|zero)\b")


def has_fee_request(text):
    # 1 sirf tab, jab fee wala phrase ho aur usi jumle mein pehle koi nafi (no/never...) na ho
    for w in FEE_WORDS:
        for m in re.finditer(re.escape(w), text):
            window = text[max(0, m.start() - 40):m.start()].split(".")[-1]
            if not NEG.search(window):
                return 1
    return 0

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]

# Yeh meri chuni hui heuristic lists hain, saboot nahi. Sirf risk indicators.
FEE_WORDS = ["registration fee", "processing fee", "security deposit",
             "training fee", "send money", "wire transfer", "western union"]
URGENCY_WORDS = ["urgent", "immediate start", "apply now", "limited seats",
                 "hurry", "guaranteed", "no experience needed"]


def build_text(df):
    # Saare text columns ko jod kar ek column banata hai
    return df[TEXT_COLS].fillna("").agg(" ".join, axis=1)


def clean_text(s):
    s = s.lower()
    s = re.sub(r"<[^>]+>", " ", s)                      # HTML tags
    s = re.sub(r"#(url|email|phone)_[a-z0-9]+#", " ", s)  # dataset placeholders
    s = re.sub(r"[^a-z\s]", " ", s)                     # sirf letters
    return re.sub(r"\s+", " ", s).strip()


def add_features(df):
    out = df.copy()
    raw = build_text(df)
    low = raw.str.lower()

    out["text_clean"] = raw.apply(clean_text)
    out["text_len"] = raw.str.len()
    out["word_count"] = raw.str.split().str.len()
    out["has_salary"] = df["salary_range"].notna().astype(int)
    out["missing_company_profile"] = df["company_profile"].isna().astype(int)
    out["has_url_placeholder"] = raw.str.contains(r"#url_", case=False).astype(int)
    out["has_email_placeholder"] = raw.str.contains(r"#email_", case=False).astype(int)
    out["fee_request"] = low.apply(has_fee_request)
    out["urgency_words"] = low.apply(lambda t: sum(w in t for w in URGENCY_WORDS))
    out["caps_ratio"] = raw.apply(lambda t: sum(c.isupper() for c in t) / max(len(t), 1))
    return out