"""
P0 PDF Source Reconciliation tests (T1–T40).

These tests validate that the P0 stage correctly identified SOURCE_NOT_FOUND,
audited the PDF, and produced the appropriate blocker artifacts. They do NOT
modify the PDF or fabricate a manuscript source.
"""

import json
import os
import re
import hashlib

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P0_DIR = os.path.join(REPO_ROOT, "results", "P0_pdf_source_reconciliation")
PDF_PATH = os.path.join(REPO_ROOT, "paper", "PR_DRL_DMOEA_UAV_New_Design_2026-09-14.pdf")
SRC_ORACLE = os.path.join(REPO_ROOT, "src", "prdrl", "oracle", "oracle.py")
D2D_VERDICT = os.path.join(REPO_ROOT, "results", "D2D_O1_production_spec_update", "D2D_verdict.json")

PDF_SHA256_EXPECTED = "66e667e63a7e9570680b2115bf3604836156415197c222e3ad7080b1746b9eae"


@pytest.fixture(scope="session")
def p0_verdict():
    with open(os.path.join(P0_DIR, "P0_verdict.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def p0_inventory():
    with open(os.path.join(P0_DIR, "P0_source_inventory.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def p0_location_map():
    with open(os.path.join(P0_DIR, "P0_pdf_location_map.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def p0_constant_audit():
    with open(os.path.join(P0_DIR, "P0_pdf_constant_audit.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def p0_alignment():
    with open(os.path.join(P0_DIR, "P0_three_way_alignment.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# T1-T5 — Output file existence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    "P0_source_inventory.json",
    "P0_pdf_location_map.json",
    "P0_pdf_constant_audit.json",
    "P0_three_way_alignment.json",
    "P0_verdict.json",
    "P0_source_blocker.md",
    "P0_report.md",
])
def test_t1_p0_output_files_exist(filename):
    full = os.path.join(P0_DIR, filename)
    assert os.path.exists(full), f"Missing P0 output: {filename}"


def test_t2_pdf_file_exists():
    assert os.path.exists(PDF_PATH), "Target PDF not found"


def test_t3_pdf_sha256_matches(p0_verdict):
    with open(PDF_PATH, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    assert actual == PDF_SHA256_EXPECTED
    assert p0_verdict["pdf_audit"]["pdf_sha256"] == PDF_SHA256_EXPECTED


def test_t4_d2d_verdict_precondition():
    assert os.path.exists(D2D_VERDICT), "D2D_verdict.json must exist before P0"
    with open(D2D_VERDICT) as f:
        v = json.load(f)
    assert v["NEXT_STAGE"] == "PDF_SOURCE_RECONCILIATION"


def test_t5_p0_report_nonempty():
    with open(os.path.join(P0_DIR, "P0_report.md")) as f:
        content = f.read()
    assert len(content) > 500, "P0_report.md is too short"


# ---------------------------------------------------------------------------
# T6-T10 — Source inventory correctness
# ---------------------------------------------------------------------------

def test_t6_source_decision_not_found(p0_inventory):
    assert p0_inventory["source_decision"] == "SOURCE_NOT_FOUND"


def test_t7_ten_searches_performed(p0_inventory):
    assert len(p0_inventory["searches_performed"]) >= 10


def test_t8_searches_cover_tex_docx_md(p0_inventory):
    descs = " ".join(s["description"].lower() for s in p0_inventory["searches_performed"])
    assert "tex" in descs
    assert "docx" in descs
    assert "md" in descs or "markdown" in descs


def test_t9_searches_cover_git_history(p0_inventory):
    descs = " ".join(s["description"].lower() for s in p0_inventory["searches_performed"])
    assert "git" in descs and ("history" in descs or "log" in descs or "stash" in descs)


def test_t10_wrong_paper_candidates_labeled(p0_inventory):
    wrong = [c for c in p0_inventory["candidate_sources"] if c["verdict"] == "WRONG_PAPER"]
    assert len(wrong) >= 2, "Must identify at least 2 wrong-paper candidates (SA-DRL-DMOEA .md and .docx)"


# ---------------------------------------------------------------------------
# T11-T15 — PDF location map
# ---------------------------------------------------------------------------

def test_t11_location_map_has_occurrences(p0_location_map):
    assert len(p0_location_map["occurrences"]) >= 3


def test_t12_sigma_p_on_page_22(p0_location_map):
    numeric = [o for o in p0_location_map["occurrences"] if o.get("sigma_P_value") is not None]
    assert len(numeric) >= 1
    page22 = [o for o in numeric if o["page"] == 22]
    assert len(page22) >= 1, "Numeric sigma_P occurrence must be on page 22"


def test_t13_sigma_p_pdf_value_is_0_05(p0_location_map):
    page22 = [o for o in p0_location_map["occurrences"] if o.get("page") == 22 and o.get("sigma_P_value") is not None]
    assert len(page22) >= 1
    assert abs(page22[0]["sigma_P_value"] - 0.05) < 1e-9, f"Expected sigma_P=0.05 in PDF, got {page22[0]['sigma_P_value']}"


def test_t14_alignment_summary_sigma_p_not_aligned(p0_location_map):
    sig = p0_location_map["alignment_summary"]["sigma_P"]
    assert sig["aligned"] is False
    assert sig["pdf_value"] == pytest.approx(0.05)
    assert sig["production_value"] == pytest.approx(0.005)


def test_t15_alignment_summary_sigma_l_aligned(p0_location_map):
    sig = p0_location_map["alignment_summary"]["sigma_L"]
    assert sig["aligned"] is True
    assert sig["pdf_value"] == pytest.approx(0.02)
    assert sig["production_value"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# T16-T20 — Constant audit
# ---------------------------------------------------------------------------

def test_t16_constant_audit_sigma_p_misaligned(p0_constant_audit):
    sp = [c for c in p0_constant_audit["constants_audited"] if c["constant"] == "sigma_P"]
    assert len(sp) == 1
    assert sp[0]["status"] == "MISALIGNED"


def test_t17_constant_audit_required_change(p0_constant_audit):
    sp = [c for c in p0_constant_audit["constants_audited"] if c["constant"] == "sigma_P"]
    assert "0.005" in sp[0]["required_change"]


def test_t18_constant_audit_sigma_l_aligned(p0_constant_audit):
    sl = [c for c in p0_constant_audit["constants_audited"] if c["constant"] == "sigma_L"]
    assert len(sl) == 1
    assert sl[0]["status"] == "ALIGNED"


def test_t19_critical_misalignments_sigma_p(p0_constant_audit):
    assert len(p0_constant_audit["critical_misalignments"]) >= 1
    assert p0_constant_audit["critical_misalignments"][0]["constant"] == "sigma_P"


def test_t20_no_fake_pdf_update(p0_constant_audit):
    assert p0_constant_audit["PDF_UPDATE_POSSIBLE"] is False
    assert p0_constant_audit["PDF_SOURCE_EDITABLE"] is False


# ---------------------------------------------------------------------------
# T21-T25 — Three-way alignment
# ---------------------------------------------------------------------------

def test_t21_three_way_verdict_misaligned(p0_alignment):
    assert p0_alignment["THREE_WAY_ALIGNMENT_VERDICT"] == "MISALIGNED"


def test_t22_alignment_source_not_found(p0_alignment):
    assert p0_alignment["three_way_alignment"]["editable_manuscript_source"]["status"] == "SOURCE_NOT_FOUND"


def test_t23_alignment_production_005(p0_alignment):
    prod = p0_alignment["three_way_alignment"]["production_code"]
    assert "0.005" in prod["value"]


def test_t24_alignment_pdf_value_0_05(p0_alignment):
    pdf = p0_alignment["three_way_alignment"]["published_pdf"]
    assert abs(float(pdf["value"].split("=")[1].strip())) - 0.05 < 1e-9 if "=" in pdf["value"] else "0.05" in pdf["value"]


def test_t25_alignment_blocker_is_source(p0_alignment):
    assert "SOURCE_NOT_FOUND" in p0_alignment["BLOCKER"]


# ---------------------------------------------------------------------------
# T26-T30 — Verdict file
# ---------------------------------------------------------------------------

def test_t26_p0_complete_is_false(p0_verdict):
    assert p0_verdict["P0_COMPLETE"] is False


def test_t27_source_decision_not_found(p0_verdict):
    assert p0_verdict["SOURCE_DECISION"] == "SOURCE_NOT_FOUND"


def test_t28_rl_gate_blocked(p0_verdict):
    assert p0_verdict["RL_GATE"] == "BLOCKED"
    assert p0_verdict["RL_STATUS"] == "NOT_STARTED"


def test_t29_next_stage_is_user_provide(p0_verdict):
    assert p0_verdict["NEXT_STAGE"] == "USER_PROVIDE_MANUSCRIPT_SOURCE"


def test_t30_p0b_audit_complete(p0_verdict):
    assert p0_verdict["verdicts"]["P0B_PDF_CONTENT_AUDIT"] == "COMPLETE"


# ---------------------------------------------------------------------------
# T31-T34 — Production code alignment
# ---------------------------------------------------------------------------

def test_t31_oracle_has_005_default():
    with open(SRC_ORACLE) as f:
        src = f.read()
    assert "sigma_predict=0.005" in src, "oracle.py must have sigma_predict=0.005 (post-D2-D)"


def test_t32_oracle_no_active_003():
    with open(SRC_ORACLE) as f:
        src = f.read()
    assert "sigma_predict=0.03," not in src, "oracle.py must not have active sigma_predict=0.03"


def test_t33_production_value_in_verdict(p0_verdict):
    assert p0_verdict["pdf_audit"]["sigma_P_in_production"] == pytest.approx(0.005)


def test_t34_pdf_value_in_verdict(p0_verdict):
    assert p0_verdict["pdf_audit"]["sigma_P_in_pdf"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# T35-T40 — SOURCE_NOT_FOUND path checks
# ---------------------------------------------------------------------------

def test_t35_source_blocker_exists():
    blocker = os.path.join(P0_DIR, "P0_source_blocker.md")
    assert os.path.exists(blocker)


def test_t36_source_blocker_has_exact_pdf_path():
    with open(os.path.join(P0_DIR, "P0_source_blocker.md")) as f:
        content = f.read()
    assert "PR_DRL_DMOEA_UAV_New_Design_2026-09-14.pdf" in content


def test_t37_source_blocker_has_pdf_hash():
    with open(os.path.join(P0_DIR, "P0_source_blocker.md")) as f:
        content = f.read()
    assert PDF_SHA256_EXPECTED in content


def test_t38_source_blocker_has_page_22():
    with open(os.path.join(P0_DIR, "P0_source_blocker.md")) as f:
        content = f.read()
    assert "22" in content
    assert "0.05" in content
    assert "0.005" in content


def test_t39_no_fabricated_tex_or_md_source():
    fabricated = [
        os.path.join(REPO_ROOT, "paper", "main.tex"),
        os.path.join(REPO_ROOT, "paper", "PR_DRL_DMOEA.tex"),
        os.path.join(REPO_ROOT, "docs", "PR_DRL_DMOEA.md"),
    ]
    for f in fabricated:
        assert not os.path.exists(f), (
            f"Fabricated manuscript source found: {f}. Must not create fake authoritative sources."
        )


def test_t40_rl_not_started():
    # RL must not be started while P0 is incomplete
    rl_files = [
        os.path.join(REPO_ROOT, "src", "prdrl", "rl"),
        os.path.join(REPO_ROOT, "src", "prdrl", "drl"),
        os.path.join(REPO_ROOT, "results", "R0_drl_training"),
        os.path.join(REPO_ROOT, "results", "R1_drl_evaluation"),
    ]
    for path in rl_files:
        # Allow directory to exist if pre-existing, but no training results
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.endswith(".json") and "training" in f.lower()]
            assert len(files) == 0, f"RL training results found at {path} — RL must not start before P0 closes"
