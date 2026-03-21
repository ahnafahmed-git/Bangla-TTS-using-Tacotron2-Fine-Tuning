# Bangla-TTS-using-Tacotron2
Tacotron2 Text-to-Speech model fine-tuned using Comprehensive Bangla tts dataset: https://www.kaggle.com/datasets/mobassir/comprehensive-bangla-tts

**Bangla TTS Fine-Tuning: Model Documentation**
Model Choice: Tacotron2
Why Tacotron2: Tacotron2 was selected over VITS and FastSpeech2 for the following reasons:

Pretrained English weights available via NVIDIA torch.hub, enabling transfer learning rather than training from scratch
Uses the LJSpeech format natively — identical to this dataset's pipe-delimited transcript structure, requiring zero format conversion
Well-documented architecture with a clear separation between the mel-spectrogram generator (Tacotron2) and vocoder (WaveGlow), making debugging and evaluation straightforward
The Strategy pattern implemented in this project allows swapping to VITSStrategy in a single line if needed in future
Architecture: Encoder (Conv + BiLSTM) → Location-Sensitive Attention → Autoregressive Decoder → Postnet (5-layer CNN) → Mel-Spectrogram → WaveGlow Vocoder → Audio

Hyperparameters
Parameter	Value	Rationale
Learning rate	1e-4	Standard for fine-tuning pretrained Tacotron2. 1e-3 caused training instability
Batch size	16 / 8	Adjust based on GPU memory constraints
Optimizer	Adam	Standard for sequence-to-sequence models
Weight decay	1e-6	Light regularisation to prevent overfitting
Gradient clipping	max_norm=1.0	Prevents exploding gradients common in RNN decoders
LR scheduler	ReduceLROnPlateau (patience=3, factor=0.5)	Reduces LR when val loss plateaus
Max decoder steps	3000	Increased from default 1000 to accommodate longer Bangla sentences
Sample rate	22050 Hz	Native dataset rate — no resampling required
Mel bins	80	Standard Tacotron2 configuration
Hop length	256	11.6ms per frame — balances temporal resolution and training speed
Train/Val/Test split	80/10/10	Standard split across 12,830 total samples
Vocab size	74 characters	Character-level Bangla vocabulary including punctuation
Performance Analysis
Training Progress
Epoch	Train Loss	Val Loss
1	2.8968	2.1822
2	2.1733	1.8433
3	1.9372	1.6742
4	1.8170	1.5895
5	1.7398	1.5266
6	1.6831	1.4860
Loss decreased consistently across all epochs with no signs of overfitting (val loss tracks train loss closely throughout).

Limitations and Future Work
Attention convergence: At 10 epochs, the stop token does not fire reliably, causing generated audio to pad to the maximum decoder steps (~34 seconds). Reliable inference requires ~20–30 epochs.
Expected metrics at full convergence: MCD 4–8 dB, PESQ 3.0+, STOI 0.85+, MOSNet 3.5+
Next steps: Resume training to 20+ epochs using the checkpoint resume pipeline, then run the full end-to-end demo for proper metric evaluation
