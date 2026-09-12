"""
5.3A — Deterministic replay of the 420 frozen SA-DRL evaluation runs to
recover per-change diagnostic data that was NOT persisted in 5.1
(only summary MIGD/FE fields were written to final_benchmark_runs.jsonl).

Uses the SAME frozen checkpoints, same seeds, same config, eps=0,
training=False. Every replay is verified against the frozen benchmark's
summary metrics (migd_end, migd_response, fes_used) before its per-change
log is accepted. Writes to a NEW file only; never touches
final_benchmark_runs.jsonl or checkpoints.
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import dynamic_runner as dr  # noqa: E402
import train  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
RAW_PATH = os.path.join(RESULTS, "final_benchmark_runs.jsonl")
MODELS_PATH = os.path.join(RESULTS, "final_benchmark_models.jsonl")
CKPT_DIR = os.path.join(RESULTS, "checkpoints")
OUT_PATH = os.path.join(RESULTS, "policy_diagnostic_per_change.csv")
MISMATCH_PATH = os.path.join(RESULTS, "policy_diagnostic_replay_mismatches.jsonl")

DFS = [f"DF{i}" for i in range(1, 15)]
TEST_SEEDS = list(range(100000, 100030))
TOL = 1e-9


def hsd(sd):
    return hashlib.md5(b"".join(v.cpu().numpy().tobytes()
                                for v in sd.values())).hexdigest()


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    sha_before = sha256_of(RAW_PATH)
    print("Raw benchmark SHA256 before replay:", sha_before)

    orig_rows = {(r["problem"], r["seed"]): r for r in
                (json.loads(l) for l in open(RAW_PATH)) if r["method"] == "SA-DRL-frozen"}
    models = {m["problem"]: m for m in
             (json.loads(l) for l in open(MODELS_PATH))}
    assert len(orig_rows) == 420, len(orig_rows)
    assert len(models) == 14

    seg = train.make_segments(train.CFG["D"])
    fieldnames = ["DF", "seed", "change_index", "time", "gate",
                 "seg1_action", "seg2_action", "segment_action_active",
                 "c1", "c2", "dispersion", "hv_drop", "d_mem", "has_memory",
                 "reward", "IGD_pre", "IGD_response", "IGD_end",
                 "delta_response", "delta_recovery", "delta_total",
                 "HV_pre", "HV_response", "HV_end"]

    n_rows_written = 0
    n_mismatch = 0
    t0 = time.time()

    with open(OUT_PATH, "w", newline="") as fout, \
         open(MISMATCH_PATH, "w") as fmis:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for name in DFS:
            model_row = models[name]
            ckpt = torch.load(os.path.join(CKPT_DIR, f"{name}.pt"), weights_only=True)
            frozen_online, frozen_target = ckpt["online"], ckpt["target"]
            assert hsd(frozen_online) == model_row["final_online_hash"], \
                f"{name}: checkpoint hash mismatch vs manifest"

            N, ref = train.problem_setup(name)
            pf = train.ALL_PROBLEMS[name]

            # instrument dispersion/hv_drop (not in return dict)
            disp_capture, hvdrop_capture = [], []
            orig_disp = dr.compute_dispersion
            orig_hvdrop = dr.compute_hv_drop

            def disp_spy(pop, xl, xu):
                v = orig_disp(pop, xl, xu)
                disp_capture.append(v)
                return v

            def hvdrop_spy(fb, fa, rp):
                v = orig_hvdrop(fb, fa, rp)
                hvdrop_capture.append(v)
                return v

            for seed in TEST_SEEDS:
                orig = orig_rows[(name, seed)]

                agent = train.new_agent(seg, seed)
                agent.online.load_state_dict(frozen_online)
                agent.target.load_state_dict(frozen_target)
                agent.eps = 0.0

                disp_capture.clear()
                hvdrop_capture.clear()
                dr.compute_dispersion = disp_spy
                dr.compute_hv_drop = hvdrop_spy
                try:
                    res = dr.run_sa_drl(
                        pf, n_t=train.CFG["n_t"], tau_t=train.CFG["tau_t"],
                        N=N, D=train.CFG["D"], n_changes=train.EVAL_CHANGES,
                        warm_up=train.CFG["warm_up"], agent=agent,
                        segments=seg, seed=seed, training=False,
                        ref_point=ref, n_elite=train.CFG["n_elite"])
                finally:
                    dr.compute_dispersion = orig_disp
                    dr.compute_hv_drop = orig_hvdrop

                ok = (abs(res["migd_end"] - orig["migd_end"]) < TOL
                     and abs(res["migd_response"] - orig["migd_response"]) < TOL
                     and res["fes_used"] == orig["fes_used"])
                if not ok:
                    n_mismatch += 1
                    fmis.write(json.dumps(dict(
                        DF=name, seed=seed,
                        replay_migd_end=res["migd_end"], orig_migd_end=orig["migd_end"],
                        replay_migd_response=res["migd_response"],
                        orig_migd_response=orig["migd_response"],
                        replay_fes=res["fes_used"], orig_fes=orig["fes_used"])) + "\n")
                    print(f"  !!! MISMATCH {name} seed={seed} -- log NOT used")
                    continue

                K = train.EVAL_CHANGES
                tl = res["timeline"]
                assert len(tl) == K == len(disp_capture) == len(hvdrop_capture) \
                    == len(res["gate_action"]) == len(res["reward_history"])

                for i in range(K):
                    gate = res["gate_action"][i]
                    seg_vals = res["seg_actions"][i]
                    active = (gate == 0)
                    igd_pre = res["igd_pre_history"][i]
                    igd_resp = res["igd_response_history"][i]
                    igd_end = res["igd_end_history"][i]
                    writer.writerow(dict(
                        DF=name, seed=seed, change_index=tl[i]["change_index"],
                        time=tl[i]["t_new"], gate=gate,
                        seg1_action=(seg_vals[0] if active else "NA"),
                        seg2_action=(seg_vals[1] if active else "NA"),
                        segment_action_active=active,
                        c1=res["normalized_change"][i][0],
                        c2=res["normalized_change"][i][1],
                        dispersion=disp_capture[i], hv_drop=hvdrop_capture[i],
                        d_mem=res["d_mem"][i], has_memory=res["has_memory"][i],
                        reward=res["reward_history"][i],
                        IGD_pre=igd_pre, IGD_response=igd_resp, IGD_end=igd_end,
                        delta_response=(igd_pre - igd_resp) if None not in (igd_pre, igd_resp) else "",
                        delta_recovery=(igd_resp - igd_end) if None not in (igd_resp, igd_end) else "",
                        delta_total=(igd_pre - igd_end) if None not in (igd_pre, igd_end) else "",
                        HV_pre=res["hv_pre_history"][i], HV_response=res["hv_response_history"][i],
                        HV_end=res["hv_end_history"][i],
                    ))
                    n_rows_written += 1

            print(f"  {name}: 30 seeds replayed  ({time.time()-t0:.0f}s elapsed)")

    sha_after = sha256_of(RAW_PATH)
    print(f"\nRows written: {n_rows_written}  (expect 42000)")
    print(f"Mismatches: {n_mismatch}  (expect 0)")
    print("Raw benchmark SHA256 after replay:", sha_after)
    print("Raw SHA unchanged:", sha_after == sha_before)
    assert sha_after == sha_before, "RAW FILE MODIFIED DURING REPLAY"

    with open(os.path.join(RESULTS, "policy_diagnostic_replay_manifest.json"), "w") as f:
        json.dump(dict(sha256_before=sha_before, sha256_after=sha_after,
                       n_rows_written=n_rows_written, n_mismatch=n_mismatch,
                       n_replays=420), f, indent=2)


if __name__ == "__main__":
    main()
