import streamlit as st
import torch
import librosa
import numpy as np
import pretty_midi
import torch.nn as nn
import torch.nn.functional as F
import os
from scipy.io import wavfile
import io
import fluidsynth

# --- 1. Model Architecture ---
class PianoTranscriptionNet(nn.Module):
    def __init__(self, input_bins=88, lstm_units=256):
        super(PianoTranscriptionNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        
        cnn_out_features = 64 * input_bins
        
        self.onset_lstm = nn.LSTM(cnn_out_features, lstm_units, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(lstm_units * 2, input_bins)
        
        self.frame_lstm = nn.LSTM(cnn_out_features + input_bins, lstm_units, batch_first=True, bidirectional=True)
        self.frame_fc = nn.Linear(lstm_units * 2, input_bins)

    def forward(self, x):
        batch_size, channels, bins, time_frames = x.size()
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(batch_size, time_frames, -1)
        
        onset_out, _ = self.onset_lstm(out)
        onset_logits = self.onset_fc(onset_out)
        onset_probs = torch.sigmoid(onset_logits)
        
        frame_input = torch.cat([out, onset_probs], dim=-1)
        
        frame_out, _ = self.frame_lstm(frame_input)
        frame_logits = self.frame_fc(frame_out)
        frame_probs = torch.sigmoid(frame_logits)
        
        return frame_probs.permute(0, 2, 1), onset_probs.permute(0, 2, 1)

# --- 2. Utility Functions ---

@st.cache_resource
def load_model(model_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PianoTranscriptionNet(input_bins=88, lstm_units=256)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def transcribe(audio_bytes, model, device):
    # Constants
    TARGET_SR = 16000
    HOP_LENGTH = 256
    N_BINS = 88
    FMIN = librosa.note_to_hz('C1')
    FRAME_TIME = HOP_LENGTH / TARGET_SR
    
    # Load audio from bytes
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
    
    # Preprocess
    C = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=N_BINS)
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    
    input_tensor = torch.tensor(C_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        pred_frames, pred_onsets = model(input_tensor)
        
    pred_frames = pred_frames.squeeze(0).cpu().numpy()
    pred_onsets = pred_onsets.squeeze(0).cpu().numpy()
    
    total_frames = pred_frames.shape[1]
    
    # Decode
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    # --- Advanced Peak-Detection State Machine ---
    ONSET_THRESHOLD = 0.35  
    FRAME_THRESHOLD = 0.20  
    
    # Track active notes: stores a dictionary with 'start_time' and 'last_onset_prob'
    # if active, or None if inactive
    active_notes = {key_idx: None for key_idx in range(88)}
    
    for frame_idx in range(total_frames):
        current_time = frame_idx * FRAME_TIME
        
        for key_idx in range(88):
            midi_note_number = key_idx + 21
            
            onset_prob = pred_onsets[key_idx, frame_idx]
            frame_prob = pred_frames[key_idx, frame_idx]
            
            # Look ahead and behind to verify if the current frame is a local peak in time
            is_peak = False
            if 0 < frame_idx < total_frames - 1:
                prev_onset = pred_onsets[key_idx, frame_idx - 1]
                next_onset = pred_onsets[key_idx, frame_idx + 1]
                # It's a peak if it's higher than its immediate neighbors and crosses the threshold
                if onset_prob > ONSET_THRESHOLD and onset_prob >= prev_onset and onset_prob > next_onset:
                    is_peak = True

            # CASE 1: Note is currently INACTIVE
            if active_notes[key_idx] is None:
                if is_peak or (onset_prob > ONSET_THRESHOLD and frame_idx == 0):
                    active_notes[key_idx] = {
                        'start_time': current_time,
                        'last_onset_frame': frame_idx
                    }
            
            # CASE 2: Note is currently ACTIVE (Tracking holds and repeated presses)
            else:
                # Sub-case A: A new continuous strike is detected at the same pitch
                # Prevent re-striking on the exact same frame or the immediate next frame (de-bounce)
                if is_peak and (frame_idx - active_notes[key_idx]['last_onset_frame'] > 2):
                    start_time = active_notes[key_idx]['start_time']
                    end_time = current_time - (FRAME_TIME / 2) # Cut off slightly early for separation
                    
                    if end_time - start_time > 0.03:
                        piano.notes.append(pretty_midi.Note(
                            velocity=100, pitch=midi_note_number, start=start_time, end=end_time
                        ))
                    
                    # Instantly start the next note cycle
                    active_notes[key_idx] = {
                        'start_time': current_time,
                        'last_onset_frame': frame_idx
                    }
                
                # Sub-case B: The note naturally decays and turns off
                elif frame_prob < FRAME_THRESHOLD:
                    start_time = active_notes[key_idx]['start_time']
                    end_time = current_time
                    
                    if end_time - start_time > 0.03:
                        piano.notes.append(pretty_midi.Note(
                            velocity=100, pitch=midi_note_number, start=start_time, end=end_time
                        ))
                    
                    active_notes[key_idx] = None
                    
    pm.instruments.append(piano)
    return pm

def synthesize_midi(pm, sr=44100):
    sf2_path = os.path.join(os.path.dirname(__file__), "Steinway Grand Piano 1.2.sf2")
    if os.path.exists(sf2_path):
        try:
            synth = fluidsynth.Synth()
            synth.start(driver="coreaudio")

            sfid = synth.sfload(sf2_path)
            if sfid == -1:
                raise RuntimeError("Failed to load SoundFont")

            synth.program_select(0, sfid, 0, 0)

            audio = pm.fluidsynth(fs=sr, synthesizer=synth)

            synth.delete()

            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            return (audio * 32767).astype(np.int16)

        except Exception as e:
            print(f"FluidSynth failed: {e}")
    
    # Fallback synthesizer (sine wave)
    total_time = pm.get_end_time()
    audio = np.zeros(int(total_time * sr) + sr)
    for instrument in pm.instruments:
        for note in instrument.notes:
            start_sample = int(note.start * sr)
            end_sample = int(note.end * sr)
            duration = end_sample - start_sample
            if duration <= 0: continue
            freq = librosa.midi_to_hz(note.pitch)
            t = np.linspace(0, note.end - note.start, duration, endpoint=False)
            wave = np.sin(2 * np.pi * freq * t)
            # Simple envelope
            attack = int(0.01 * sr)
            envelope = np.ones(duration)
            if attack > 0 and attack < duration:
                envelope[:attack] = np.linspace(0, 1, attack)
                envelope[attack:] = np.linspace(1, 0, duration - attack)
            wave *= envelope
            audio[start_sample:end_sample] += wave * (note.velocity / 127.0) * 0.2
    
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return (audio * 32767).astype(np.int16)

# --- 3. Streamlit UI ---

st.set_page_config(page_title="Piano Transcriber", page_icon="🎹", layout="centered")

st.title("Piano Transcription")
st.markdown("""
Upload an audio file of a piano performance, and this AI will convert it into a MIDI file. 
You can play the transcribed version directly in your browser!
""")

model_path = "best_piano_transcriber.pth"

if not os.path.exists(model_path):
    st.error(f"Model file `{model_path}` not found! Please make sure it's in the project directory.")
else:
    model, device = load_model(model_path)
    
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a", "ogg"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Transcribe"):
            with st.spinner("Transcribing... this might take a minute depending on the audio length."):
                audio_bytes = uploaded_file.read()
                pm = transcribe(audio_bytes, model, device)
                
                # Success!
                st.success("Transcription complete!")
                
                # MIDI Download
                midi_data = io.BytesIO()
                pm.write(midi_data)
                st.download_button(
                    label="Download MIDI",
                    data=midi_data.getvalue(),
                    file_name="transcribed.mid",
                    mime="audio/midi"
                )
                
                # Playback
                st.subheader("Play Transcribed Output")
                audio_wav = synthesize_midi(pm)
                
                wav_io = io.BytesIO()
                wavfile.write(wav_io, 44100, audio_wav)
                st.audio(wav_io.getvalue(), format='audio/wav')