from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

TEXT_COL = "text_clean"
NUM_COLS = ["text_len", "word_count", "caps_ratio", "urgency_words"]
BIN_COLS = ["telecommuting", "has_company_logo", "has_questions", "has_salary",
            "missing_company_profile", "has_url_placeholder", "has_email_placeholder",
            "fee_request"]
CAT_COLS = ["employment_type", "required_experience", "required_education",
            "industry", "function"]
ALL_FEATURES = [TEXT_COL] + NUM_COLS + BIN_COLS + CAT_COLS


def build_preprocessor(scaler="standard"):
    num_scaler = MinMaxScaler(clip=True) if scaler == "minmax" else StandardScaler()
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    return ColumnTransformer([
        ("text", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                 min_df=3, stop_words="english"), TEXT_COL),
        ("num", num_scaler, NUM_COLS),
        ("bin", "passthrough", BIN_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])