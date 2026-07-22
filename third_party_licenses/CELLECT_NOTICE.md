# CELLECT attribution

The experimental CELLECT integration uses the upstream project:

- Project: **CELLECT (Contrastive Embedding Learning for Large-scale Efficient Cell Tracking)**
- Source: <https://github.com/zzz333za/CELLECT>
- Pinned revision: `3586070926f7f1fd5d8df37456861d22bdc63236`
- License: GNU General Public License v2.0 only
- Local source: `external/CELLECT` (Git submodule)
- License copy: `third_party_licenses/CELLECT_GPL-2.0.txt`
- Paper: <https://www.nature.com/articles/s41592-025-02886-x>

Local modifications and integration code are kept outside the pristine pinned
submodule. Files that derive from or bind directly to CELLECT identify
themselves as GPL-2.0-only. The production cell-tracking baseline does not load
this experiment unless explicitly requested.
