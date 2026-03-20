# WaveGlow is NVIDIA's vocoder that pairs with Tacotron2
# Converts mel to audio
print("Loading WaveGlow vocoder...")
waveglow = torch.hub.load(
    'NVIDIA/DeepLearningExamples:torchhub',
    'nvidia_waveglow',
    model_math='fp32',
    pretrained=True,
    verbose=False,
)
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to(device)
waveglow.eval()
print("WaveGlow loaded.")
