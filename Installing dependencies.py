# ── Core dependencies ────────────────────────────────────────────────────
!pip install inflect==5.6.0 librosa Unidecode pesq pystoi scipy -q

# ── Clone NVIDIA Tacotron2 ───────────────────────────────────────────────
%cd /kaggle/working
!git clone https://github.com/NVIDIA/tacotron2.git
%cd /kaggle/working/tacotron2
!git submodule init
!git submodule update

# ── Clone and install speechmetrics ─────────────────────────────────────
%cd /kaggle/working
!git clone https://github.com/aliutkus/speechmetrics.git
%cd /kaggle/working/speechmetrics
!pip install -e . -q

# ── Return to working dir ────────────────────────────────────────────────
%cd /kaggle/working
print("All dependencies installed.")
print(f"Working dir: {os.getcwd()}")
