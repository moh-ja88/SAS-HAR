# Folder Reorganization Plan — PhD Yasmeen (BASS-HAR)

**Current state:** 50+ files flat in root + 7 subdirectories with unclear purpose  
**Goal:** Clean, professional structure matching PhD-level research organization  
**Date:** Sep 17, 2026

---

## Current Problems

1. **Flat root** — 40+ files dumped in one folder (proposals, lit reviews, code, data, random temp files)
2. **Duplicate/legacy files** — `HAR Proposal 2025.docx` vs `HAR Proposal 2025 (3).docx`, `New proposal.md` vs `.docx` vs `.txt`
3. **Stale artifacts** — `nul` (empty file), `mdpdf.log`, `test.txt`, `comments_docx.txt`, `env_text_docx.txt`
4. **No separation** — proposals, lit reviews, code, experiments, reference papers all mixed together
5. **Unclear subdirs** — `temp/`, `temp_har/`, `temp_pulmovec/`, `-p/` (dash-p) — what are these?
6. **No README** explaining folder purpose or file relationships

---

## Proposed Structure

```
C:\Work\OpenCode\PhDYasmeen\
│
├── README.md                          ← Project overview (NEW — replaces old README)
├── PROGRESS.md                        ← Progress log (existing, keep)
│
├── 01_Proposal/
│   ├── New_propal.md                  ← Current BASS-HAR V3 (SSL-first)
│   ├── New_proposal.pdf              ← Compiled PDF
│   ├── PROPOSAL_REVISIONS_v4.md      ← Supervisor feedback + revisions
│   ├── PROPOSAL_REVISIONS_v4.pdf
│   └── archive/
│       ├── HAR_Proposal_2025.docx    ← Original proposal (Mar 2026)
│       ├── HAR_Proposal_2025.pdf
│       ├── New_proposal.docx         ← Earlier export
│       ├── New_proposal.txt
│       ├── har-proposal-novelty-analysis.md
│       ├── har-proposal-complete-novelty-analysis.md
│       ├── har-proposal-complete-novelty-analysis.pdf
│       └── proposal_text.txt
│
├── 02_Literature_Review/
│   ├── ssl_boundary_detection_har_literature_review.md
│   ├── ssl_boundary_detection_har_literature_review.pdf
│   ├── ssl_har_literature_review.md
│   ├── ssl_har_literature_review.pdf
│   ├── bass_har_research_ideas.md   ← PhD extension ideas (22KB, 27 refs)
│   ├── BASS_HAR_Literature_Review_30_Papers.xlsx
│   └── archive/
│       └── Literature_Review_Contrastive_Predictive_ssl.docx  ← Earlier draft
│
├── 03_Reference_Papers/
│   ├── Noor_2017_Adaptive_sliding_window.pdf      ← Supervisor's paper
│   ├── Deep_similarity_segmentation_sensor.pdf
│   └── Similarity_Segmentation_Approach.pdf
│
├── 04_Experiments/
│   ├── WISDM_Baseline/
│   │   ├── 01_WISDM_Window50_FIXED.ipynb          ← Fixed notebook (subject-wise split)
│   │   └── 01_WISDM_Window50_original.ipynb       ← Original (with leakage, for reference)
│   ├── PAMAP2/                                     ← Future experiments
│   ├── Opportunity/                                ← Future experiments
│   └── Pilot_SSL_boundary/                         ← De-risk pilot (planned)
│
├── 05_Code/
│   ├── PhD-HAR-Segmentation/                       ← Existing project (has .gitignore, LICENSE)
│   ├── SWL-Adapt/                                  ← Existing adaptive window code
│   ├── adwin/                                      ← ADWIN drift detection
│   ├── OPTWIN/                                     ← OPTWIN code
│   └── utils/
│       ├── convert_md_to_pdf_fpdf.py
│       └── create_pdf_simple.py
│
├── 06_Notes/
│   ├── Mohammed_1.docx                             ← Student summary
│   └── Enhanced_deep_learning_model_HAR.txt        ← Notes from paper
│
└── _archive/                                       ← Everything we're unsure about
    ├── temp_har/                                   ← Appears to be a HAR toolkit (Docker, run_*.py)
    ├── temp_pulmovec/                              ← Unknown purpose
    ├── temp/                                       ← Empty temp
    ├── SWL-Adapt.jpg                               ← (moved to 05_Code/SWL-Adapt/)
    ├── comments_docx.txt                           ← Extracted DOCX comments
    ├── env_text_docx.txt                           ← Environment info
    ├── extract_comments.py                         ← Utility script
    ├── mdpdf.log                                   ← PDF compilation log
    ├── nul                                         ← Empty file (Windows artifact)
    ├── test.txt                                    ← Test file
    ├── requirements.txt                            ← Old requirements
    ├── README.md (old)                             ← Old readme
    └── -p/                                         ← Unknown directory
```

---

## File Mapping (Current → New)

### Root → `01_Proposal/`
| Current File | New Location | Notes |
|---|---|---|
| `New proposal.md` | `01_Proposal/New_proposal.md` | Rename: spaces→underscores |
| `New proposal.pdf` | `01_Proposal/New_proposal.pdf` | |
| `PROPOSAL_REVISIONS_v4.md` | `01_Proposal/PROPOSAL_REVISIONS_v4.md` | |
| `PROPOSAL_REVISIONS_v4.pdf` | `01_Proposal/PROPOSAL_REVISIONS_v4.pdf` | |
| `HAR Proposal 2025.docx` | `01_Proposal/archive/HAR_Proposal_2025.docx` | Legacy |
| `HAR Proposal 2025 (3).docx` | `01_Proposal/archive/HAR_Proposal_2025_v3.docx` | Legacy |
| `HAR Proposal 2025.pdf` | `01_Proposal/archive/HAR_Proposal_2025.pdf` | Legacy |
| `New proposal.docx` | `01_Proposal/archive/New_proposal.docx` | Legacy export |
| `New proposal.txt` | `01_Proposal/archive/New_proposal.txt` | Legacy export |
| `har-proposal-novelty-analysis.md` | `01_Proposal/archive/` | Legacy |
| `har-proposal-complete-novelty-analysis.md` | `01_Proposal/archive/` | Legacy |
| `har-proposal-complete-novelty-analysis.pdf` | `01_Proposal/archive/` | Legacy |
| `proposal_text.txt` | `01_Proposal/archive/` | Legacy |

### Root → `02_Literature_Review/`
| Current File | New Location | Notes |
|---|---|---|
| `ssl_boundary_detection_har_literature_review.md` | `02_Literature_Review/` | |
| `ssl_boundary_detection_har_literature_review.pdf` | `02_Literature_Review/` | |
| `ssl_har_literature_review.md` | `02_Literature_Review/` | |
| `ssl_har_literature_review.pdf` | `02_Literature_Review/` | |
| `bass_har_research_ideas.md` | `02_Literature_Review/` | PhD ideas report |
| `BASS_HAR_Literature_Review_30_Papers.xlsx` | `02_Literature_Review/` | |
| `Literature_Review_Contrastive_Predictive_ssl.docx` | `02_Literature_Review/archive/` | Earlier draft |

### Root → `03_Reference_Papers/`
| Current File | New Location | Notes |
|---|---|---|
| `Noor et al. - 2017 - Adaptive sliding window...pdf` | `03_Reference_Papers/Noor_2017_Adaptive_sliding_window.pdf` | Supervisor's paper |
| `Deep similarity segmentation model for sensor-based.pdf` | `03_Reference_Papers/` | |
| `Similarity_Segmentation_Approach...pdf` | `03_Reference_Papers/` | |

### Root → `04_Experiments/`
| Current File | New Location | Notes |
|---|---|---|
| `01_WISDM_Window50_FIXED.ipynb` | `04_Experiments/WISDM_Baseline/` | Fixed notebook |

### Root → `05_Code/utils/`
| Current File | New Location | Notes |
|---|---|---|
| `convert_md_to_pdf_fpdf.py` | `05_Code/utils/` | |
| `create_pdf_simple.py` | `05_Code/utils/` | |
| `extract_comments.py` | `05_Code/utils/` | |

### Root → `06_Notes/`
| Current File | New Location | Notes |
|---| ideas report                 |
| `Mohammed 1.docx` | `06_Notes/` | Student summary |
| `Enhanced deep learning model for HAR.txt` | `06_Notes/` | Paper notes |

### Root → `_archive/` (delete or review)
| Current File | Action | Notes |
|---|---|---|
| `nul` | **DELETE** | Windows artifact (92 bytes) |
| `mdpdf.log` | **DELETE** | 398KB log file |
| `test.txt` | **DELETE** | Test file |
| `comments_docx.txt` | MOVE to `_archive/` | Extracted comments |
| `env_text_docx.txt` | MOVE to `_archive/` | Environment info |
| `requirements.txt` | MOVE to `_archive/` | Old requirements |
| `README.md` (old) | MOVE to `_archive/` | Will be replaced |
| `-p/` | MOVE to `_archive/` | Unknown purpose |
| `temp/` | MOVE to `_archive/` | |
| `temp_har/` | MOVE to `_archive/` | HAR toolkit (Docker, run_*.py) |
| `temp_pulmovec/` | MOVE to `_archive/` | Unknown |

### Subdirs (already in right place)
| Current Folder | New Location | Notes |
|---|---|---|
| `PhD-HAR-Segmentation/` | `05_Code/PhD-HAR-Segmentation/` | Keep structure |
| `SWL-Adapt/` | `05_Code/SWL-Adapt/` | Keep structure |
| `adwin/` | `05_Code/adwin/` | |
| `OPTWIN/` | `05_Code/OPTWIN/` | |

---

## Execution Steps (after approval)

1. Create new directory structure
2. Move files according to mapping above
3. Delete stale files (`nul`, `mdpdf.log`, `test.txt`)
4. Write new `README.md` with project overview
5. Verify no broken paths in PROGRESS.md or other docs
6. Verify no git-tracked files are lost

---

## Notes

- **No files will be deleted** except 3 clearly stale ones (`nul`, `mdpdf.log`, `test.txt`)
- Everything else is moved to `_archive/` for safety
- All code subdirectories (PhD-HAR-Segmentation, SWL-Adapt, etc.) keep their internal structure
- The `_archive/` folder can be reviewed and cleaned later
- Spaces in filenames replaced with underscores for CLI compatibility
