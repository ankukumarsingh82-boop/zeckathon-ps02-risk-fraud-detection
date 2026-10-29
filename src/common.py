import os, random
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
SEED = 42
def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)

# Polish attributes that can be rebuilt from Indian annual statements (size-free ratios only).
# name -> (plain-language label, monotone direction of risk: +1 higher=riskier, -1 lower=riskier, 0 none)
TRANSFER_FEATURES = {
    "Attr1":  ("Return on assets (net profit / total assets)", -1),
    "Attr2":  ("Leverage (total liabilities / total assets)", +1),
    "Attr3":  ("Working capital / total assets", -1),
    "Attr4":  ("Current ratio (current assets / short-term liabilities)", -1),
    # Attr6 (retained earnings / TA) dropped: Polish median is 0.000 (retained-earnings line is mostly empty) while Indian
    # "reserves" include securities premium and general reserves (median 0.46) - not the same concept, so not transferable.
    "Attr7":  ("EBIT / total assets", -1),
    "Attr8":  ("Book equity / total liabilities", -1),
    "Attr9":  ("Asset turnover (sales / total assets)", -1),
    "Attr10": ("Equity / total assets", -1),
    "Attr13": ("(Pre-tax profit + depreciation) / sales", -1),
    "Attr16": ("(Pre-tax profit + depreciation) / total liabilities (debt-service capacity)", -1),
    "Attr18": ("Pre-tax profit / total assets", -1),
    "Attr21": ("Sales growth (sales / prior-year sales)", 0),
    "Attr22": ("Operating profit / total assets", -1),
    "Attr23": ("Net margin (net profit / sales)", -1),
    "Attr26": ("(Net profit + depreciation) / total liabilities", -1),
    "Attr27": ("Interest cover (operating profit / finance costs)", -1),
    "Attr30": ("(Total liabilities - cash) / sales", +1),
    "Attr38": ("Long-term capital (equity + long-term debt) / total assets", -1),
    "Attr40": ("Cash ratio ((current assets - inventory - receivables) / short-term liabilities)", -1),
    "Attr44": ("Receivable days", 0),
    "Attr46": ("Quick ratio ((current assets - inventory) / short-term liabilities)", -1),
    "Attr50": ("Current assets / total liabilities", -1),
    "Attr51": ("Short-term liabilities / total assets", +1),
    "Attr59": ("Long-term liabilities / equity", +1),
    "Attr61": ("Receivable turnover (sales / receivables)", 0),
    "Attr62": ("Short-term liabilities in days of sales", +1),
    "Attr63": ("Sales / short-term liabilities", -1),
    "Attr64": ("Sales / fixed assets", 0),
}
FEATS = list(TRANSFER_FEATURES)
COST_FN, COST_FP = 20.0, 1.0   # missed default costs 20x a false alarm
ALERT_RATE = 0.05


class Prep:
    """Winsorise at train 1/99 pct, median-impute, standardise (for the linear model)."""
    def fit(self, X):
        self.lo, self.hi = X.quantile(0.01), X.quantile(0.99)
        Xc = X.clip(self.lo, self.hi, axis=1); self.med = Xc.median()
        Xc = Xc.fillna(self.med); self.mu, self.sd = Xc.mean(), Xc.std().replace(0, 1); return self
    def clip(self, X): return X[self.lo.index].clip(self.lo, self.hi, axis=1)
    def tree(self, X): return self.clip(X).fillna(self.med)   # trees: winsorised + train-median imputed
    def lin(self, X): return ((self.clip(X).fillna(self.med) - self.mu) / self.sd)

