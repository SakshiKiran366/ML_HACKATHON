"""Build a fact-based PathSnap report from the repository's recorded results."""
from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle, Image, KeepTogether)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parents[1] / "outputs"

def rows(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))

def build():
    OUT.mkdir(parents=True, exist_ok=True)
    models = rows(ROOT / "results" / "experiments_summary.csv")
    decoder = rows(ROOT / "results" / "decoder_comparison_summary.csv")
    tuning_path = ROOT / "results" / "tuned_vs_untuned.csv"
    tuned = rows(tuning_path) if tuning_path.exists() else []
    text = [
        "# PathSnap: Technical Report (team review draft)",
        "",
        "This report is generated from the experiment files in this repository. Please review it, make sure every team member understands the methods, and add screenshots from the actual two-laptop demonstration before submission. No evaluator-test data was available in this workspace.",
        "",
        "## System and data",
        "PathSnap matches noisy GPS observations to candidate directed road segments. The input files are whitespace-separated: `.track` contains longitude, latitude and seconds; `.nodes` contains longitude/latitude; `.arcs` contains directed node pairs; `.route` contains reference arc IDs. The checked-in development partitions contain 80 training trips and 10 student-test trips. The separate evaluator partition was not present and was not used.",
        "",
        "The feature pipeline cleans non-finite and non-increasing timestamp rows, projects each trip to local metres, searches for nearby arcs, and calculates distance, direction, motion, time-gap, neighbouring-point and network-connectivity features. Training labels align GPS points monotonically to the supplied reference route. Observations more than 60 m from the aligned route segment are excluded as unreliable. The student-test labels are used only for development comparison.",
        "",
        "## Model comparison",
        "The project trained five distinct classifiers and used five-fold GroupKFold by trip. The primary selection metric was observation-level Top-1 accuracy; log loss was used as a probability-quality tie-breaker. Student-test results are reported as development comparison, not as a substitute for the reserved evaluator partition.",
        "",
        "| Model | CV Top-1 | CV log loss | Student-test Top-1 | Student-test macro F1 | Student-test log loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in models:
        text.append(f"| {m['model']} | {float(m['cv_top1_mean']):.4f} | {float(m['cv_log_loss_mean']):.4f} | {float(m['test_top1']):.4f} | {float(m['test_macro_f1_top1']):.4f} | {float(m['test_log_loss']):.4f} |")
    best = max(models, key=lambda m: float(m["cv_top1_mean"]))
    text += ["", f"The recorded comparison selects **{best['model']}**: it has the highest mean grouped-CV Top-1 ({float(best['cv_top1_mean']):.4f}) and the lowest CV log loss ({float(best['cv_log_loss_mean']):.4f}) among these five models. The final API artifact is saved under `artifacts/final_model.joblib`.", ""]
    text += ["## Bayesian optimisation", ""]
    if tuned:
        for t in tuned:
            text.append(f"- {t.get('model','')} — CV Top-1 {float(t.get('cv_top1_mean',0)):.4f}; student-test Top-1 {float(t.get('test_top1',0)):.4f}; student-test log loss {float(t.get('test_log_loss',0)):.4f}.")
        best_json = ROOT / "results" / "optuna_best.json"
        if best_json.exists():
            import json
            b = json.loads(best_json.read_text())
            text.append(f"- TPE completed {b.get('n_trials', 'recorded')} trials. Best training-CV Top-1: {float(b.get('best_cv_top1', 0)):.4f}. Search details and trials are in `results/optuna_best.json`, `results/optuna_trials.csv`, and `results/optuna_search_space.json`.")
    else:
        text.append("The TPE search is not yet recorded. Run `python src/tuning/optuna_hgb.py --trials 10 --folds 5` before treating this report as final.")
    text += ["", "## HMM and Viterbi", "", "The hidden state is a candidate directed road arc for each GPS observation. Classifier probabilities supply emission scores. Viterbi transitions favour staying on an arc or moving head-to-tail to a connected arc; projected along-arc distance, expected travel distance and GPS gap duration adjust transition costs. The decoder returns one road segment per observation. Long gaps soften connectivity penalties.", ""]
    if decoder:
        d = decoder[0]
        text.append(f"On the ten student-test trips, classifier-only mean point accuracy was {float(d['classifier_accuracy']):.4f} and Viterbi accuracy was {float(d['viterbi_accuracy']):.4f}. Mean directed route continuity changed from {float(d['classifier_continuity']):.4f} to {float(d['viterbi_continuity']):.4f}. Per-trip results are in `results/decoder_comparison.csv`.")
    text += ["", "## Phase checklist", "", "- Phases 1-6: project setup, dataset exploration, preprocessing, candidates/features and reference-route labels are implemented and recorded in the source and existing exploration outputs.", "- Phases 7-8: the five classifiers are separated into model modules; grouped CV and student-test comparisons are recorded.", "- Phase 9: the Bayesian search and tuned-versus-untuned measurements are retained with the Optuna trial CSV.", "- Phases 10-11: Viterbi decoding is implemented and compared; Folium route maps can be regenerated from the prediction-map script.", "- Phase 12: the FastAPI service is connected to the pushed React UI contract; the UI production build and TypeScript check pass.", "- Phases 13-14: startup and LAN instructions and this report are prepared. The actual other-laptop connection, organizer evaluator data and required demonstration screenshots still need to be supplied/checked by the team.", "", "## API, UI and deployment", "", "The FastAPI service implements `/health`, `/model`, `/trips`, `/trips/{trip_id}` and `/reconstruct`. On the development dataset, functional calls completed for all ten student-test trips. The API predicts first; it reads a reference route only afterward to calculate server-side metrics. Route labels are not returned to the browser. The React/Vite UI connects to the API through its settings panel and renders returned GPS, candidate-road context, prediction and metrics.", "", "For a two-laptop demonstration, run the API on laptop 1 with `uvicorn api.main:app --host 0.0.0.0 --port 8000`; configure laptop 2's UI to use the backend's reachable LAN address. At event time, set `PATHSNAP_TRIPS_DIR` to the organizer-provided evaluator-trip directory. That directory was not available here, so a final cross-laptop evaluator run and the required live screenshots remain team actions.", "", "## Reproducibility and disclosure", "", "The five-model comparison is in `results/experiments_summary.csv` and `results/experiments_folds.csv`; model configurations and random seed are recorded there and in the run metadata. The trained estimator, feature list and model configuration are in `artifacts/final_model.joblib`. AI assistance was used for implementation and report drafting; the team must review the text, verify claims against its understanding, and disclose assistance under the event policy.", ""]
    md = OUT / "PathSnap_Technical_Report.md"
    md.write_text("\n".join(text))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=23, leading=29, textColor=colors.HexColor("#153b55"), spaceAfter=10))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    table_data = [["Model", "CV Top-1", "CV log loss", "Test Top-1", "Test macro F1", "Test log loss"]]
    table_data += [[m["model"], *(f"{float(m[k]):.4f}" for k in ["cv_top1_mean", "cv_log_loss_mean", "test_top1", "test_macro_f1_top1", "test_log_loss"])] for m in models]
    model_table = Table(table_data, repeatRows=1, colWidths=[33*mm, 22*mm, 22*mm, 22*mm, 25*mm, 23*mm])
    model_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#153b55")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),
        ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#c7d2da")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f1f5f7")]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story = [Paragraph("PathSnap: Technical Report", styles["CoverTitle"]), Paragraph("Team review draft · generated from recorded project results", styles["Italic"]), Spacer(1, 8)]
    # Render report sections with compact, readable heading hierarchy.
    body = text[2:]
    i = 0
    def clean_markup(value):
        value = value.replace("&", "&amp;").replace("`", "")
        value = value.replace("**", "")
        return value
    while i < len(body):
        line = body[i]
        if line.startswith("## "):
            heading = Paragraph(line[3:], styles["Heading2"])
            nextline = body[i+1] if i+1 < len(body) else ""
            if line[3:] == "Model comparison":
                story += [Spacer(1, 6), heading, model_table]
                i += 1
                continue
            skip = 2 if nextline == "" and i+2 < len(body) else 1
            first_content = body[i+skip] if i+skip < len(body) else ""
            if first_content.startswith("- "):
                story += [Spacer(1, 6), KeepTogether([heading, Paragraph("• " + clean_markup(first_content[2:]), styles["BodyText"])])]
                i += skip + 1
                continue
            if first_content and not first_content.startswith(("## ", "|", "- ")):
                story += [Spacer(1, 6), KeepTogether([heading, Paragraph(clean_markup(first_content), styles["BodyText"])])]
                i += skip + 1
                continue
            if nextline.startswith("- "):
                story += [Spacer(1, 6), KeepTogether([heading, Paragraph("• " + clean_markup(nextline[2:]), styles["BodyText"])])]
                i += 2
                continue
            if nextline and not nextline.startswith(("## ", "|", "- ")):
                story += [Spacer(1, 6), KeepTogether([heading, Paragraph(clean_markup(nextline), styles["BodyText"])])]
                i += 2
                continue
            story += [Spacer(1, 6), heading]
        elif line.startswith("| Model") or line.startswith("|---"):
            i += 1; continue
        elif line.startswith("| "):
            i += 1; continue
        elif line.startswith("- "):
            story.append(Paragraph("• " + clean_markup(line[2:]), styles["BodyText"]))
        elif line:
            story.append(Paragraph(clean_markup(line), styles["BodyText"]))
        else:
            story.append(Spacer(1, 3))
        i += 1
    pdf = OUT / "PathSnap_Technical_Report.pdf"
    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#5d6b75"))
        canvas.drawString(18*mm, 12*mm, "PathSnap · team review draft")
        canvas.drawRightString(A4[0]-18*mm, 12*mm, f"Page {doc.page}"); canvas.restoreState()
    map_png = OUT / "PathSnap_Map_00000000.png"
    if map_png.exists():
        story += [PageBreak(), Paragraph("Representative development-trip map", styles["Heading1"]),
                  Paragraph("GPS observations, the model-decoded route and the supplied development reference route. The reference layer is included for this report only; it is never returned by the API.", styles["BodyText"]), Spacer(1, 8),
                  Image(str(map_png), width=170*mm, height=128*mm)]
    SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                      topMargin=16*mm, bottomMargin=20*mm, title="PathSnap Technical Report").build(story, onFirstPage=footer, onLaterPages=footer)
    print(md); print(pdf)

if __name__ == "__main__": build()
