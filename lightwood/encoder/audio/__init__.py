# This encoder is optional since it's underlying dependency (librosa) needs system dependencies
try:
    from lightwood.encoder.audio.mfcc import MFCCEncoder
except Exception:
    MFCCEncoder = None

__all__ = ['MFCCEncoder']
