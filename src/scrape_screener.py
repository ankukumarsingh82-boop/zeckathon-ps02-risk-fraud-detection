"""Polite, cached scraper for public screener.in company pages (annual P&L, balance sheet,
cash flow, schedule breakdowns, weekly price). Each fetch is logged with URL and access timestamp
so every downstream data point carries provenance.

Usage: python src/scrape_screener.py <slug> [<slug> ...]   or   --file list.txt
Output: data/raw/screener/<SLUG>.json  (parsed tables + provenance metadata)
"""
import json, os, re, sys, time, datetime as dt
import requests
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "screener")
UA = {"User-Agent": "Mozilla/5.0 (research; zeckathon-ps02 early-warning study)"}
DELAY = float(os.environ.get("SCREENER_DELAY", "1.0"))
S = requests.Session(); S.headers.update(UA)

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

def get(url, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, timeout=30)
        except Exception as e:
            time.sleep(5 * (i + 1)); continue
        if r.status_code == 429:
            time.sleep(30 * (i + 1)); continue
        time.sleep(DELAY)
        return r
    return None

def parse_table(sec):
    if sec is None: return {}
    tb = sec.find("table")
    if tb is None: return {}
    rows = tb.find_all("tr")
    hdr = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])][1:]
    out = {}
    for r in rows[1:]:
        cells = [c.get_text(strip=True) for c in r.find_all(["th", "td"])]
        if not cells: continue
        name = cells[0].replace("\xa0", " ").rstrip("+").strip()
        out[name] = dict(zip(hdr, cells[1:]))
    return out

def ncols(rec):
    return len(rec.get("tables", {}).get("balance-sheet", {}).get("Total Assets", {}) or {})

def scrape(slug, prefer_consolidated=True, force_variant=None):
    """Consolidated statements preferred; if the consolidated history has <10 annual columns
    (e.g. a company that only began consolidating recently) the standalone page is used when it is longer."""
    path = os.path.join(OUT, f"{slug.replace('&','_and_')}.json")
    if os.path.exists(path) and force_variant is None:
        rec = json.load(open(path))
        if rec.get("variant") == "consolidated" and ncols(rec) < 10 and not rec.get("checked_standalone"):
            alt = scrape(slug, force_variant="standalone")
            if ncols(alt) > ncols(rec): rec = alt
            rec["checked_standalone"] = True; json.dump(rec, open(path, "w"))
        return rec
    rec = {"slug": slug, "fetches": []}
    html = None
    variants = [force_variant] if force_variant else (["consolidated", "standalone"] if prefer_consolidated else ["standalone"])
    for variant in variants:
        url = f"https://www.screener.in/company/{slug}/" + ("consolidated/" if variant == "consolidated" else "")
        r = get(url)
        if r is None or r.status_code != 200: 
            rec["fetches"].append({"url": url, "status": None if r is None else r.status_code, "accessed_utc": now()}); continue
        soup = BeautifulSoup(r.text, "lxml")
        pl = parse_table(soup.find("section", id="profit-loss"))
        if not pl or all(all(v in ("", "0") for v in d.values()) for d in pl.values()):
            rec["fetches"].append({"url": url, "status": "empty", "accessed_utc": now()}); continue
        rec["fetches"].append({"url": url, "status": 200, "accessed_utc": now()})
        rec["variant"] = variant; rec["page_url"] = url; html = soup; break
    if html is None:
        rec["error"] = "no data"
        if force_variant is None: json.dump(rec, open(path, "w"))
        return rec
    m = re.search(r'data-company-id="(\d+)"', str(html))
    cid = m.group(1) if m else None
    rec["company_id"] = cid
    rec["name"] = (html.find("h1").get_text(strip=True) if html.find("h1") else slug)
    # sector breadcrumb
    peers = html.find("section", id="peers")
    rec["sector_links"] = [a.get_text(strip=True) for a in peers.find_all("a")][:4] if peers else []
    rec["tables"] = {k: parse_table(html.find("section", id=k)) for k in ["profit-loss", "balance-sheet", "cash-flow", "ratios"]}
    rec["financial_format"] = any(k.startswith("Financing Profit") or k == "Revenue" for k in rec["tables"]["profit-loss"])
    cons = "true" if rec["variant"] == "consolidated" else ""
    rec["schedules"] = {}
    if cid and not rec["financial_format"]:
        for parent, sec in [("Other Assets", "balance-sheet"), ("Other Liabilities", "balance-sheet"),
                            ("Borrowings", "balance-sheet"), ("Expenses", "profit-loss")]:
            url = f"https://www.screener.in/api/company/{cid}/schedules/?parent={requests.utils.quote(parent)}&section={sec}&consolidated={cons}"
            r = get(url)
            ok = r is not None and r.status_code == 200
            rec["fetches"].append({"url": url, "status": r.status_code if r is not None else None, "accessed_utc": now()})
            try: rec["schedules"][parent] = r.json() if ok else {}
            except Exception: rec["schedules"][parent] = {}
    if cid:
        url = f"https://www.screener.in/api/company/{cid}/chart/?q=Price&days=10000"
        r = get(url)
        rec["fetches"].append({"url": url, "status": r.status_code if r is not None else None, "accessed_utc": now()})
        try:
            ds = r.json()["datasets"]; rec["price"] = [v for d in ds if d["metric"] == "Price" for v in d["values"]]
        except Exception: rec["price"] = []
    if force_variant is None:
        json.dump(rec, open(path, "w"))
    return rec

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--file":
        args = [l.strip() for l in open(args[1]) if l.strip() and not l.startswith("#")]
    for i, s in enumerate(args):
        try:
            r = scrape(s)
            print(i, s, r.get("variant"), r.get("financial_format"), len(r.get("tables", {}).get("profit-loss", {}).get("Sales", {}) or {}), flush=True)
        except Exception as e:
            print(i, s, "ERROR", e, flush=True)
