import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from votekit.pref_profile import PreferenceProfile
from src.edouard.MeekGraph_new import MeekGraph


def main():
    profile = PreferenceProfile.from_pickle("data/portland/Portland_D4.pkl")

    MeekGraph(
        profile=profile,
        m=3,
        auditable_margins_per_deg=[2000, 2000, 2000],
        autosave_path="data/meek_autosave/D4_profile.json",
        use_numerical_labels=False,
        autosave_layers=True,
        autosave_interval=50000,
        resume=True,
        show_progress=True,
        save_layers_json=True,
        full_graph_path="data/meek_autosave/D4_profile_full.json",
        safe_mode=True,
    )


if __name__ == "__main__":
    main()
