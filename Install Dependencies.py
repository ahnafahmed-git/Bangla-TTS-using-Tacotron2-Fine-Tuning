!pip install numpy==1.23.5 -q   # NVIDIA repo needs older numpy
!git clone https://github.com/NVIDIA/tacotron2.git
%cd tacotron2
!git submodule init
!git submodule update
!pip install inflect==5.6.0 librosa Unidecode -q
!pip install pesq pystoi scipy -q
!git clone https://github.com/aliutkus/speechmetrics.git
%cd speechmetrics && pip install -e . -q && %cd ..
