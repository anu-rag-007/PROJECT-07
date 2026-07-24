
# ══════════════════════════════════════════════════════════
# REAL MUSE S INTEGRATION
# Replace EEGStreamSimulator with this when device arrives
# ══════════════════════════════════════════════════════════

# 1. Install dependencies:
# pip install muselsl pylsl

# 2. Start Muse stream (in terminal):
# muselsl stream

# 3. In your Python script:
from pylsl import StreamInlet, resolve_stream
import numpy as np

class MuseSStream:
    """Real-time EEG from Muse S via LSL protocol."""

    def __init__(self, channel='AF7'):
        """
        channel: 'TP9', 'AF7', 'AF8', or 'TP10'
        AF7 is closest to Fpz (your training channel)
        """
        print("Looking for Muse S stream...")
        streams     = resolve_stream('type', 'EEG')
        self.inlet  = StreamInlet(streams[0])
        self.info   = self.inlet.info()
        self.fs     = int(self.info.nominal_srate())

        # Muse S channel order: TP9, AF7, AF8, TP10, Right AUX
        channel_map = {'TP9':0, 'AF7':1, 'AF8':2, 'TP10':3}
        self.ch_idx = channel_map.get(channel, 1)

        print(f"  Muse S connected")
        print(f"   Sample rate: {self.fs} Hz")
        print(f"   Channel:     {channel} (idx {self.ch_idx})")

    def get_sample(self):
        """Get one EEG sample — same interface as simulator."""
        sample, timestamp = self.inlet.pull_sample()
        return np.array([sample[self.ch_idx]])

    def start(self): pass
    def stop(self):  pass


# Replace:
# stream = EEGStreamSimulator(fs=256)
# With:
# stream = MuseSStream(channel='AF7')

# Everything else in the pipeline stays IDENTICAL.
# That's the value of the abstraction.
