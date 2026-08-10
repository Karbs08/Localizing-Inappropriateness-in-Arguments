from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
HUMAN_STUDY_DIR = REPO_ROOT / "human_study"
SURVEY_INPUT_DIR = HUMAN_STUDY_DIR / "survey_input"
SURVEY_OUTPUT_DIR = HUMAN_STUDY_DIR / "survey_output"

sys.path.insert(0, str(REPO_ROOT))

from src.survey.limesurvey_tsv import LimeSurveyTSVBuilder
from src.survey.survey_design import build_variant_rows


SURVEY_CONFIG = {
    "survey_title": "Evaluating Span Explanations for Inappropriate Arguments",
    "base_language": "en",
    "rating_scale_max": 7,
    "n_arguments": 7,
    "seed": 20260803,
    "use_cookie_to_prevent_repeat": True,
    "save_timings": True,
    "method_order": [
        "shap",
        "ig",
        "attention",
        "tfidf",
        "mil",
        "random",
        "llm",
    ],
}


CRITERIA = (
    (
        "REL",
        "Relevance: The highlighted spans identify text that contributes to the argument being inappropriate.",
    ),
    (
        "SUF",
        "Sufficiency: Taken together, the highlighted spans provide enough evidence to justify classifying the argument as inappropriate.",
    ),
    (
        "COM",
        "Completeness: All or nearly all important reasons for the argument's inappropriateness have been highlighted.",
    ),
    (
        "PRE",
        "Precision: The highlights contain little or no unnecessary or unrelated text.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the LimeSurvey import files for the human study."
    )
    parser.add_argument(
        "--items",
        type=Path,
        default=SURVEY_INPUT_DIR / "study_items.csv",
    )
    parser.add_argument(
        "--argument-ids-file",
        type=Path,
        default=SURVEY_INPUT_DIR / "selected_argument_ids.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SURVEY_OUTPUT_DIR,
    )
    return parser.parse_args()


def escape_lime_text(value: object) -> str:
    # Literal braces can be parsed as ExpressionScript. HTML entities avoid that.
    return (
        html.escape(str(value), quote=True)
        .replace("{", "&#123;")
        .replace("}", "&#125;")
        .replace("\n", "<br>")
    )


def highlighted_html(text: str, spans_json: str) -> str:
    spans = json.loads(spans_json)
    if not isinstance(spans, list):
        raise ValueError("spans_json must contain a list")

    parts: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: (int(s["start"]), int(s["end"]))):
        start = int(span["start"])
        end = int(span["end"])
        if start < cursor or start < 0 or end > len(text) or start >= end:
            raise ValueError(f"Invalid or overlapping span ({start}, {end})")
        parts.append(escape_lime_text(text[cursor:start]))
        parts.append(
            '<span style="background-color:#fff1a8;padding:1px 3px;'
            'border-radius:3px;font-weight:600;">'
            + escape_lime_text(text[start:end])
            + "</span>"
        )
        cursor = end
    parts.append(escape_lime_text(text[cursor:]))
    return "".join(parts)


def box(title: str, content: str, *, accent: str = "#667085") -> str:
    return f"""
<div style="border:1px solid #d0d5dd;border-left:5px solid {accent};border-radius:7px;
            padding:15px 17px;margin:12px 0;background:#ffffff;line-height:1.65;">
  <div style="font-weight:700;margin-bottom:7px;">{escape_lime_text(title)}</div>
  <div>{content}</div>
</div>
""".strip()


def add_scale_answers(
    builder: LimeSurveyTSVBuilder,
    question_id: int,
    max_value: int,
    *,
    include_na: bool,
) -> None:
    for value in range(1, max_value + 1):
        if value == 1:
            label = "1 – Not at all"
        elif value == max_value:
            label = f"{max_value} – Completely"
        elif value == (max_value + 1) // 2:
            label = f"{value} – Partially"
        else:
            label = str(value)
        builder.answer(question_id, str(value), label, assessment_value=value)
    if include_na:
        builder.answer(
            question_id,
            "NA",
            "Not assessable – I do not consider the argument inappropriate",
        )


def add_rating_array(
    builder: LimeSurveyTSVBuilder,
    qcode: str,
    display_label: str,
    issue: str,
    text: str,
    spans_json: str,
    rating_scale_max: int,
) -> int:
    question_html = (
        f"<h4 style=\"margin-bottom:8px;\">Explanation {display_label}</h4>"
        + box("Highlighted argument", highlighted_html(text, spans_json), accent="#d4a72c")
        + "<p style=\"margin-top:10px;\">Evaluate the highlighted spans using all four criteria.</p>"
    )
    qid = builder.question(
        "F",
        qcode,
        question_html,
        mandatory=True,
        hide_tip=True,
    )
    for code, criterion_text in CRITERIA:
        builder.subquestion(qid, code, criterion_text)
    add_scale_answers(builder, qid, rating_scale_max, include_na=True)
    return qid


def select_arguments(
    items: pd.DataFrame,
    n_arguments: int,
    seed: int,
    explicit_path: Path | None,
    method_order: list[str],
) -> list[str]:
    items = items.copy()
    items["global_row_id"] = items["global_row_id"].astype(str)
    complete = (
        items.groupby("global_row_id")["method_id"].nunique()
        .loc[lambda series: series == len(method_order)]
        .index.tolist()
    )

    if explicit_path is not None:
        selected = [
            line.strip()
            for line in explicit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(selected) != n_arguments:
            raise ValueError(
                f"Expected {n_arguments} argument IDs, found {len(selected)}."
            )
        missing = sorted(set(selected) - set(complete))
        if missing:
            raise ValueError(f"Incomplete or missing argument IDs: {missing}")
        return selected

    if len(complete) < n_arguments:
        raise ValueError(
            f"Only {len(complete)} complete arguments are available; "
            f"{n_arguments} are required."
        )
    return (
        pd.Series(sorted(complete))
        .sample(n=n_arguments, random_state=seed, replace=False)
        .tolist()
    )


def add_survey_settings(builder: LimeSurveyTSVBuilder, config: dict) -> None:
    language = config.get("base_language", "en")
    builder.survey_parameter("language", language)
    builder.survey_parameter("format", "G")
    builder.survey_parameter("anonymized", "Y")
    builder.survey_parameter("savetimings", "Y" if config.get("save_timings", True) else "N")
    builder.survey_parameter("datestamp", "N")
    builder.survey_parameter("ipaddr", "N")
    builder.survey_parameter("refurl", "N")
    builder.survey_parameter(
        "usecookie", "Y" if config.get("use_cookie_to_prevent_repeat", True) else "N"
    )
    builder.survey_parameter("allowsave", "N")
    builder.survey_parameter("allowprev", "Y")
    builder.survey_parameter("listpublic", "N")
    builder.survey_parameter("showwelcome", "N")
    builder.survey_parameter("showprogress", "Y")

    builder.language_parameter("surveyls_title", config["survey_title"])
    builder.language_parameter(
        "surveyls_description",
        "A blinded human evaluation of localized evidence for argument inappropriateness.",
    )
    builder.language_parameter("surveyls_welcometext", "")
    builder.language_parameter(
        "surveyls_endtext",
        "<h3>Thank you for participating.</h3><p>Your answers have been recorded.</p>",
    )
    builder.language_parameter("surveyls_url", "")
    builder.language_parameter("surveyls_urldescription", "")
    builder.language_parameter("surveyls_dateformat", "1")
    builder.language_parameter("surveyls_numberformat", "0")


def add_intro(builder: LimeSurveyTSVBuilder, rating_scale_max: int) -> None:
    builder.group("Consent", "Study information and consent")
    consent_text = """
<h2>Study information</h2>
<p>This study evaluates highlighted text spans that are intended to explain why an argument may be inappropriate in a constructive discussion.</p>
<p><strong>Content warning:</strong> Some arguments may contain insulting, aggressive, offensive, or otherwise inappropriate language.</p>
<ul>
  <li>Participation is voluntary and you may stop at any time.</li>
  <li>No name, email address, IP address, or referrer URL is requested or stored.</li>
  <li>The survey stores your ratings and, if enabled, page-level response times.</li>
  <li>The results will be used for a master's thesis.</li>
</ul>
<p>By selecting “I consent”, you confirm that you are at least 18 years old and agree to participate.</p>
""".strip()
    qid = builder.question("L", "CONSENT", consent_text, mandatory=True)
    builder.answer(qid, "Y", "I consent to participate")
    builder.answer(qid, "N", "I do not consent")

    builder.group(
        "Declined",
        "Participation declined",
        relevance="CONSENT == 'N'",
    )
    builder.question(
        "X",
        "DECLINEINFO",
        "<p>You chose not to participate. No study ratings will be collected. You may now close this page.</p>",
    )

    builder.group(
        "Introduction",
        "Understanding inappropriateness",
        relevance="CONSENT == 'Y'",
    )
    intro_html = f"""
<h2>What does “inappropriate” mean in this study?</h2>
<p>This study follows the concept of <strong>inappropriateness in argumentation</strong> introduced by
<a href="https://aclanthology.org/2023.acl-long.238/" target="_blank" rel="noopener noreferrer"><strong>Ziegenbein et al. (2023)</strong></a>.
They treat appropriateness as a minimal quality requirement for an argument to be valuable in a debate.
Importantly, an argument can be inappropriate for more reasons than simply containing insults or offensive words.</p>

<p>In light of its <strong>discussion context</strong>, an argument can be considered inappropriate if it exhibits one or more of the following four core types:</p>

<div style="margin:18px 0 22px 0;">
  <div style="text-align:center;background:#f2f4f7;border:1px solid #d0d5dd;border-radius:8px;padding:11px 14px;font-weight:700;font-size:1.08em;margin-bottom:12px;">
    Inappropriateness
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:stretch;">
    <div style="flex:1 1 190px;min-width:180px;border:1px solid #f3b5c5;border-top:4px solid #e83e73;border-radius:8px;padding:12px;background:#fff;">
      <div style="font-weight:700;margin-bottom:5px;">Toxic Emotions</div>
      <div style="font-size:.94em;margin-bottom:7px;">Emotions are excessively intense or used deceptively in a way that obstructs critical discussion.</div>
      <div style="font-size:.88em;color:#475467;"><strong>Examples:</strong> Excessive Intensity, Emotional Deception</div>
    </div>
    <div style="flex:1 1 190px;min-width:180px;border:1px solid #c9c2ff;border-top:4px solid #7a5af8;border-radius:8px;padding:12px;background:#fff;">
      <div style="font-weight:700;margin-bottom:5px;">Missing Commitment</div>
      <div style="font-size:.94em;margin-bottom:7px;">The issue is not taken seriously or the author is not open to engaging with opposing arguments on their merits.</div>
      <div style="font-size:.88em;color:#475467;"><strong>Examples:</strong> Missing Seriousness, Missing Openness</div>
    </div>
    <div style="flex:1 1 190px;min-width:180px;border:1px solid #ffd19a;border-top:4px solid #f79009;border-radius:8px;padding:12px;background:#fff;">
      <div style="font-weight:700;margin-bottom:5px;">Missing Intelligibility</div>
      <div style="font-size:.94em;margin-bottom:7px;">The meaning is unclear or irrelevant to the issue, or the reasoning is difficult to understand.</div>
      <div style="font-size:.88em;color:#475467;"><strong>Examples:</strong> Unclear Meaning, Missing Relevance, Confusing Reasoning</div>
    </div>
    <div style="flex:1 1 190px;min-width:180px;border:1px solid #d0d5dd;border-top:4px solid #667085;border-radius:8px;padding:12px;background:#fff;">
      <div style="font-weight:700;margin-bottom:5px;">Other Reasons</div>
      <div style="font-size:.94em;margin-bottom:7px;">Other severe language problems can also make an argument inappropriate.</div>
      <div style="font-size:.88em;color:#475467;"><strong>Examples:</strong> Detrimental Orthography, Reason Unclassified</div>
    </div>
  </div>
  <div style="font-size:.82em;color:#667085;margin-top:7px;text-align:center;">
    Simplified representation based on the taxonomy of Ziegenbein et al. (2023), Figure 3.
  </div>
</div>

<div style="border:1px solid #b2ddff;border-left:5px solid #2e90fa;border-radius:7px;padding:13px 15px;background:#eff8ff;margin:18px 0;">
  <div style="font-weight:700;margin-bottom:5px;">Inappropriateness is subjective</div>
  <div>There is no single objectively correct judgment for every argument. In the original annotation study, the annotators fully agreed on overall inappropriateness in 60% of cases, and the reported Krippendorff’s α was <strong>.45</strong>. Therefore, we are interested in <strong>your own judgment</strong> based on the definition above.</div>
</div>

<p><strong>The discussion issue matters.</strong> Always judge an argument in the context of the issue shown above it.
Also, do <strong>not</strong> judge whether you personally agree with the argument's position. An opinion you disagree with can still be expressed appropriately, and an opinion you agree with can be expressed inappropriately.</p>

<p>For every argument, you will first rate its overall inappropriateness. Afterwards, you will see highlighted passages proposed as explanations and rate them on <strong>Relevance, Sufficiency, Completeness, and Precision</strong> using a <strong>1–{rating_scale_max}</strong> scale. The meaning of each criterion is repeated with the rating questions.</p>

<p style="font-size:.9em;color:#475467;">
  <strong>Further reading:</strong>
  <a href="https://aclanthology.org/2023.acl-long.238/" target="_blank" rel="noopener noreferrer">Ziegenbein et al. (2023): <em>Modeling Appropriate Language in Argumentation</em>, ACL 2023</a>.
</p>
""".strip()
    builder.question("X", "INTROTEXT", intro_html)
    check_id = builder.question(
        "L",
        "CHECK1",
        "Which statement is consistent with the definition used in this study?",
        mandatory=True,
    )
    builder.answer(
        check_id,
        "A",
        "An argument is inappropriate only when it contains insults or swear words",
    )
    builder.answer(
        check_id,
        "B",
        "An argument may be inappropriate because of toxic emotions, missing commitment, missing intelligibility, or other severe language problems",
    )
    builder.answer(
        check_id,
        "C",
        "An argument is inappropriate whenever I disagree with its position",
    )

    builder.group(
        "Practice",
        "Short practice example",
        relevance="CONSENT == 'Y'",
    )
    practice_text = (
        box(
            "Issue",
            "Should public transport be free?",
            accent="#2e90fa",
        )
        + box(
            "Argument",
            "Only a complete idiot could oppose this policy. Free transport would also reduce congestion.",
        )
        + box(
            "Explanation A",
            'Only a <span style="background-color:#fff1a8;padding:1px 3px;border-radius:3px;font-weight:600;">complete idiot</span> could oppose this policy. Free transport would also reduce congestion.',
            accent="#d4a72c",
        )
        + box(
            "Explanation B",
            '<span style="background-color:#fff1a8;padding:1px 3px;border-radius:3px;font-weight:600;">Only a complete idiot could oppose this policy. Free transport would also reduce congestion.</span>',
            accent="#d4a72c",
        )
        + "<p>Explanation A is more precise because it marks the insulting phrase without including the unrelated policy justification.</p>"
    )
    builder.question("X", "PRACTICETEXT", practice_text)
    practice_id = builder.question(
        "L",
        "PRACTICECHECK",
        "Which explanation is more precise?",
        mandatory=True,
    )
    builder.answer(practice_id, "A", "Explanation A")
    builder.answer(practice_id, "B", "Explanation B")

    builder.group(
        "Randomization",
        "Internal randomization",
        relevance="CONSENT == 'Y'",
    )
    builder.question(
        "*",
        "VERSION",
        "{if(is_empty(VERSION.NAOK), rand(1,7), VERSION.NAOK)}",
        always_hide=True,
        hide_tip=True,
    )


def build_survey(
    items: pd.DataFrame,
    config: dict,
    selected_ids: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    method_order = list(config["method_order"])
    rating_scale_max = int(config.get("rating_scale_max", 7))
    language = config.get("base_language", "en")

    items = items.copy()
    items["global_row_id"] = items["global_row_id"].astype(str)
    items["method_id"] = items["method_id"].astype(str)
    item_lookup = {
        (row.global_row_id, row.method_id): row
        for row in items.itertuples(index=False)
    }

    variant_rows = build_variant_rows(selected_ids, method_order)
    variant_df = pd.DataFrame(variant_rows)
    variant_df.to_csv(output_dir / "variant_design.csv", index=False)

    selected_df = (
        items[items["global_row_id"].isin(selected_ids)]
        .drop_duplicates("global_row_id")
        .set_index("global_row_id")
        .loc[selected_ids]
        .reset_index()[["global_row_id", "issue", "post_text"]]
    )
    selected_df.insert(0, "argument_index", range(1, len(selected_df) + 1))
    selected_df.to_csv(output_dir / "selected_arguments.csv", index=False)

    builder = LimeSurveyTSVBuilder(language=language)
    add_survey_settings(builder, config)
    add_intro(builder, rating_scale_max)
    intro_preview = next(
        row["text"]
        for row in builder.rows
        if row.get("class") == "Q" and row.get("name") == "INTROTEXT"
    )

    mapping_rows: list[dict] = []
    preview_pages: list[str] = []

    # Order groups by displayed page position. For a given version, only one of
    # the seven groups at each position is relevant.
    for design in variant_rows:
        version = int(design["version"])
        display_position = int(design["display_position"])
        argument_index = int(design["argument_index"])
        global_row_id = str(design["global_row_id"])
        relevance = f"CONSENT == 'Y' && VERSION == {version}"
        group_code = f"P{display_position:02d}V{version}"
        builder.group(
            group_code,
            f"Argument {display_position} of 7",
            relevance=relevance,
        )

        base_row = item_lookup[(global_row_id, method_order[0])]
        issue = str(base_row.issue)
        post_text = str(base_row.post_text)

        overall_code = f"A{argument_index:02d}V{version}O"
        overall_html = (
            f"<h2>Argument {display_position} of 7</h2>"
            + box("Discussion issue", escape_lime_text(issue), accent="#2e90fa")
            + box("Argument without highlights", escape_lime_text(post_text))
            + "<p>Before evaluating the explanations, rate the argument itself.</p>"
        )
        overall_id = builder.question(
            "L",
            overall_code,
            overall_html,
            mandatory=True,
        )
        for value in range(1, rating_scale_max + 1):
            if value == 1:
                label = "1 – Not inappropriate at all"
            elif value == rating_scale_max:
                label = f"{rating_scale_max} – Clearly inappropriate"
            elif value == (rating_scale_max + 1) // 2:
                label = f"{value} – Moderately inappropriate"
            else:
                label = str(value)
            builder.answer(overall_id, str(value), label, assessment_value=value)
        builder.answer(overall_id, "U", "Unsure / cannot assess")

        for display_label in ("A", "B", "C"):
            method_id = str(design[f"method_{display_label}"])
            row = item_lookup[(global_row_id, method_id)]
            rating_code = f"A{argument_index:02d}V{version}{display_label}"
            add_rating_array(
                builder,
                rating_code,
                display_label,
                issue,
                post_text,
                str(row.spans_json),
                rating_scale_max,
            )
            mapping_rows.append(
                {
                    "global_row_id": global_row_id,
                    "argument_index": argument_index,
                    "display_position": display_position,
                    "version": version,
                    "display_label": display_label,
                    "method_id": method_id,
                    "rating_question_code": rating_code,
                    "preference_question_code": f"A{argument_index:02d}V{version}P",
                    "overall_question_code": overall_code,
                }
            )

        pref_code = f"A{argument_index:02d}V{version}P"
        pref_id = builder.question(
            "L",
            pref_code,
            "Which explanation best localizes the reasons for this argument's inappropriateness?",
            mandatory=True,
        )
        builder.answer(pref_id, "A", "Explanation A")
        builder.answer(pref_id, "B", "Explanation B")
        builder.answer(pref_id, "C", "Explanation C")
        builder.answer(pref_id, "N", "No meaningful difference / no preference")

        comment_code = f"A{argument_index:02d}V{version}CMT"
        builder.question(
            "T",
            comment_code,
            "Optional: Briefly explain your preference or mention anything unclear.",
            mandatory=False,
        )

        if version == 1:
            preview_parts = [
                f"<h2>Argument {display_position} of 7</h2>",
                box("Discussion issue", escape_lime_text(issue), accent="#2e90fa"),
                box("Argument without highlights", escape_lime_text(post_text)),
            ]
            for display_label in ("A", "B", "C"):
                method_id = str(design[f"method_{display_label}"])
                row = item_lookup[(global_row_id, method_id)]
                preview_parts.append(
                    box(
                        f"Explanation {display_label}",
                        highlighted_html(post_text, str(row.spans_json)),
                        accent="#d4a72c",
                    )
                )
            preview_pages.append("\n".join(preview_parts))

    builder.group("Final", "Final feedback", relevance="CONSENT == 'Y'")
    builder.question(
        "T",
        "FINALCOMMENT",
        "Optional final feedback: Was anything unclear or difficult to evaluate?",
        mandatory=False,
    )
    builder.question(
        "X",
        "THANKYOU",
        "<h3>Thank you.</h3><p>Submit the survey to save your answers.</p>",
    )

    import_path = output_dir / "span_human_study_import.txt"
    builder.write(import_path)
    pd.DataFrame(mapping_rows).to_csv(output_dir / "question_mapping.csv", index=False)

    preview_html = """<!doctype html><html><head><meta charset="utf-8"><title>Survey preview</title>
<style>body{font-family:Arial,sans-serif;max-width:920px;margin:30px auto;padding:0 20px;color:#1d2939}.page{border-bottom:4px solid #eee;padding:15px 0 35px}</style></head><body>
<h1>Static preview – questionnaire version 1</h1>
<p>This preview includes the inappropriateness introduction and all stimuli for version 1. The imported LimeSurvey survey additionally contains consent, comprehension checks, rating matrices, preference questions, and final feedback.</p>
<section class="page">""" + intro_preview + """</section>
""" + "\n".join(f'<section class="page">{page}</section>' for page in preview_pages) + "</body></html>"
    (output_dir / "preview_version_1.html").write_text(preview_html, encoding="utf-8")

    report = [
        "LIMESURVEY SURVEY GENERATION REPORT",
        "=" * 39,
        f"Import file: {import_path.name}",
        f"Survey rows: {len(builder.rows)}",
        f"Arguments: {len(selected_ids)}",
        f"Variants: 7",
        f"Methods: {', '.join(method_order)}",
        f"Rating scale: 1-{rating_scale_max}",
        f"Explanation ratings per participant: {len(selected_ids) * 3}",
        "",
        "Balance guarantees per completed participant:",
        "- every method appears exactly 3 times",
        "- every method pair appears together exactly once",
        "- every method appears exactly once as A, B, and C",
        "",
        "Before activation:",
        "1. Import the .txt file in LimeSurvey.",
        "2. Open Tools / Survey logic file and resolve any reported errors.",
        "3. Preview and complete at least two test responses.",
        "4. Verify anonymous-response, IP, referrer, cookie and timing settings.",
    ]
    (output_dir / "generation_report.txt").write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote {import_path}")
    print(f"Wrote {output_dir / 'question_mapping.csv'}")
    print(f"Wrote {output_dir / 'variant_design.csv'}")
    print(f"Wrote {output_dir / 'preview_version_1.html'}")


def main() -> None:
    args = parse_args()
    config = SURVEY_CONFIG

    items = pd.read_csv(args.items)
    selected_ids = [
        line.strip()
        for line in args.argument_ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    build_survey(
        items=items,
        config=config,
        selected_ids=selected_ids,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
