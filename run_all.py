#!/usr/bin/env python
"""One-command reproducible pipeline: python run_all.py [--scrape]"""
import os, subprocess, sys, zipfile
ROOT = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(ROOT, "src")
env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"), PYTHONHASHSEED="42")

def run(script, extra=None, args=()):
    print(f"\n=== {script} {' '.join(args)} {extra or ''}", flush=True)
    subprocess.run([sys.executable, script, *args], cwd=SRC, env=dict(env, **(extra or {})), check=True)

uci = os.path.join(ROOT, "data", "raw", "uci")
if not os.path.exists(os.path.join(uci, "5year.arff")):
    zipfile.ZipFile(os.path.join(uci, "polish.zip")).extractall(uci)
if "--scrape" in sys.argv:
    for f in ["panel_slugs.txt", "universe_slugs.txt"]:
        run("scrape_screener.py", args=("--file", os.path.join(ROOT, "data", "reference", f)))
run("polish_model.py")
run("score_india.py")
run("score_india.py", {"AVAIL_LAG_DAYS": "60", "RUN_TAG": "lag60"})
run("score_india.py", {"UNIVERSE": "totalmarket", "RUN_TAG": "totalmarket"})
run("make_report.py")
print("\nDone. Report: reports/report.pdf | App: streamlit run app.py | Static dashboard: dashboard/index.html")
