import os, sys, json, csv, shutil
from pathlib import Path

REPO = "/mnt/user-data/outputs/hts-trust-eval"
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ["HTS_MODEL_A"] = "claude-sonnet-5-5"
os.environ["HTS_MODEL_B"] = "llama-3.3-70b-versatile"
os.environ["HTS_JUDGE_MODEL"] = "llama-3.1-8b-instant"

ROOT = "results/rehearsal"
shutil.rmtree(ROOT, ignore_errors=True)
for f in ("rehearsal_val.jsonl", "rehearsal_test.jsonl", "rehearsal_corpus.jsonl"):
    Path("results").mkdir(exist_ok=True); Path("results", f).unlink(missing_ok=True)

def row(i, desc, code):
    return {"id": f"c{i}", "source_ruling_id": f"CASEHQ{i}", "messages": [
        {"role": "user", "content": f"What is the HTS US Code for {desc}?"},
        {"role": "assistant", "content": f"HTS US Code -> {code}\nReasoning -> demo rationale."}]}

val = [("L-Lysine HCl from China","2922410010"),("PVC powder for faux leather","3904100000"),
       ("2-Phenylpropionic acid","2916397900"),("DC brushless display motor","8501104040"),
       ("Lithium bis(fluorosulfonyl)imide","2826909000"),("women's nylon-spandex slip","6108110010"),
       ("overheight frame attachment","8426190000"),("seamless copper tubes","7411101090")]
test = [("women's knit wool cape","6102100000"),("offset printing posters","4911912020"),
        ("wafer-test probe card","8536908085"),("HRSG steam drum parts","8402900090"),
        ("ceramic beer dispense tower","8481801050"),("cast iron birdfeeder","7323915000"),
        ("tablet computer ADP machine","8471300000"),("liquid level gauge instrument","9015804000"),
        ("passenger motor vehicle","8703230010"),("knitted cotton t-shirt","6109100012")]

with open("results/rehearsal_val.jsonl","w") as f:
    for i,(d,c) in enumerate(val): f.write(json.dumps(row(i,d,c))+"\n")
with open("results/rehearsal_test.jsonl","w") as f:
    for i,(d,c) in enumerate(test): f.write(json.dumps(row(100+i,d,c))+"\n")

corpus = [{"id":f"EV{i}","text":t,"source_url":"https://rulings.cbp.gov/demo",
           "source_ruling_id":f"HQDEMO{i}","accessed_at":"2026-10-24","version":"demo-v1"}
          for i,t in enumerate([
            "Chapter 29 covers separate chemically defined organic compounds and amino acids.",
            "Heading 3904 covers polymers of vinyl chloride in primary forms.",
            "Chapter 85 covers electric motors and electrical machinery.",
            "Heading 7411 covers refined copper tubes and pipes.",
            "Chapter 61 covers knitted or crocheted apparel articles.",
            "Heading 8471 covers automatic data processing machines.",
            "Chapter 84 covers machinery and mechanical appliances and parts.",
            "Heading 8703 covers motor cars for the transport of persons."])]
with open("results/rehearsal_corpus.jsonl","w") as f:
    for r in corpus: f.write(json.dumps(r)+"\n")

from pipeline import init, Study
from generation import MockClient
from empirical_report import build_report

log = lambda k,v: print(f"[{k:14s}] {v}")

init(ROOT, "rehearsal-sample", "retrieved", "exclude", 2000, 4, "platt", 200)
log("init", "protocol frozen")
s = Study(ROOT)
log("prepare", {k: v["retained"] if isinstance(v,dict) and "retained" in v else "ok"
                for k,v in s.prepare("results/rehearsal_val.jsonl","results/rehearsal_test.jsonl").items()})
log("evidence", s.evidence("results/rehearsal_corpus.jsonl", 3))

def replies(split, over):
    rows = [json.loads(l) for l in open(f"{ROOT}/{split}/records.jsonl")]
    out = []
    for i, r in enumerate(rows):
        g = r["gold"]
        if i % 2 == 0: code, conf = g, 0.90
        elif i % 3 == 0: code, conf = g[:6]+"99"+g[8:], 0.70
        else: code, conf = "9999"+g[4:], 0.55
        if over: conf = min(0.97, conf+0.20)
        out.append(json.dumps({"hts_code":code,"confidence":conf,"rationale":"demo","evidence_ids":[]}))
    return out

for split in ("validation","test"):
    for label, over in (("A",False),("B",True)):
        m = MockClient(os.environ[f"HTS_MODEL_{label}"], replies(split, over))
        log(f"gen {split[:3]}-{label}", s.generate(split, label, client=m))

met = s.metrics()
log("metrics", {k:{"all_acc":round(v["all_case_exact10_accuracy"],2),"valid":v["n_valid"]} for k,v in met.items()})
cal = s.calibrate()
log("calibrate", {k:{"raw_ece":round(v["raw"]["ece"],3),"cal_ece":round(v["calibrated"]["ece"],3)} for k,v in cal.items()})
rt = s.route()
log("route", {"raw_risk":round(rt["selective_risk_raw"],3),"cal_risk":round(rt["selective_risk_calibrated"],3),
              "clusters":rt["n_case_clusters"]})
log("prepare-judge", s.prepare_judge())
mj = MockClient(os.environ["HTS_JUDGE_MODEL"], ['{"label":"supported","reason":"demo"}'])
log("judge", s.judge("J", client=mj))
log("export-human", s.export_human())

# fill the blinded annotations (labels/reasons/annotator only; content preserved)
with open(f"{ROOT}/annotations/blinded.csv") as f:
    rows = list(csv.DictReader(f)); fields = rows[0].keys()
for r in rows:
    r["label"] = "supported"; r["reason"] = "demo agreement"; r["annotator_id"] = "anno1"
with open(f"{ROOT}/annotations/filled.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(fields)); w.writeheader(); w.writerows(rows)
attest = {"annotator_id":"anno1","expertise":"demo trained second annotator",
          "independent_of_generation":True,"blinded_to_gold_and_judge":True,"rubric_version":"v2"}
json.dump(attest, open(f"{ROOT}/annotations/attestation.json","w"))
hv = s.validate_human(f"{ROOT}/annotations/filled.csv", f"{ROOT}/annotations/attestation.json")
log("validate-human", {"n":hv["n"],"kappa":hv["kappa"],"agreement":hv["agreement"]})

rep = build_report(ROOT, f"{ROOT}/report", require_human=True)
log("report", rep if isinstance(rep,str) else "built")
print("\nOUTPUT FILES:")
for p in sorted(Path(ROOT).rglob("*")):
    if p.is_file(): print("  ", p.relative_to(ROOT))
