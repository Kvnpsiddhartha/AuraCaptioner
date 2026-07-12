import streamlit as st
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
import json

# Force loading .env file before importing our packages
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v

from src import ingest, sampling, audio, ocr, grounding, verification, styling, judge
from src.config import load_settings
from src.schemas import Task, GroundingFacts, Style

# Set page configuration with a premium dark-ish layout
st.set_page_config(
    page_title="AuraCaptioner - Gemma 4 Video Captioning Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS
st.markdown("""
<style>
    .reportview-container {
        background: #0f1116;
    }
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8585 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #8a99ad;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }
    .caption-card {
        background-color: #1a1e27;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2e3646;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .caption-style {
        color: #ff4b4b;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.9rem;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .caption-text {
        font-size: 1.1rem;
        color: #f1f5f9;
        line-height: 1.5;
    }
    .step-header {
        font-weight: 600;
        color: #ff8585;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🎬 AuraCaptioner</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Next-generation video captioning powered by Google Gemma 4 models on Fireworks AI.</p>", unsafe_allow_html=True)

# Define Demo Videos
DEMO_VIDEOS = {
    "Select a Demo Video...": "",
    "Bicycle & Car Detection (Sports/Traffic)": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
    "Car Traffic Flow (Street Scene)": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
    "Short Testing Clip (Fast Download)": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/free-h264-video-for-testing.mp4"
}

# Sidebar configuration
st.sidebar.header("🔧 Settings & Models")

try:
    settings = load_settings()
    st.sidebar.success("Successfully loaded Fireworks AI settings.")
except Exception as e:
    st.sidebar.error(f"Failed to load settings: {e}. Please configure FIREWORKS_API_KEY.")
    settings = None

# Custom Model Override sliders/inputs
if settings:
    st.sidebar.markdown("### Active Gemma Models")
    st.sidebar.info(f"**Grounding:** \n`{settings.grounding_model.split('/')[-1]}`")
    st.sidebar.info(f"**Styling:** \n`{settings.styling_model.split('/')[-1]}`")
    st.sidebar.info(f"**Judge:** \n`{settings.judge_model.split('/')[-1]}`")
    
    st.sidebar.markdown("### Config Overrides")
    frames_range = st.sidebar.slider(
        "Frames range (min/max)", 
        min_value=4, 
        max_value=30, 
        value=(settings.frames_min, settings.frames_max)
    )
    settings.frames_min, settings.frames_max = frames_range[0], frames_range[1]
    
    max_retries = st.sidebar.slider(
        "Self-Judge Max Retries",
        min_value=0,
        max_value=4,
        value=settings.max_self_judge_retries
    )
    settings.max_self_judge_retries = max_retries

# Selected Styles choice
selected_styles = st.sidebar.multiselect(
    "Choose Caption Styles",
    options=["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
    default=["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
)

# App UI Logic
st.markdown("### 1. Provide a Video URL")

# Demo link dropdown
demo_choice = st.selectbox("Quick Start: Choose a Demo Video", list(DEMO_VIDEOS.keys()))
default_url = DEMO_VIDEOS[demo_choice] if demo_choice != "Select a Demo Video..." else ""

# Video URL input
video_url = st.text_input("Or enter any public direct MP4 URL:", value=default_url)

if video_url:
    st.video(video_url)

# Button to trigger the pipeline
run_clicked = st.button("🚀 Generate Captions with Gemma 4", disabled=(settings is None or not video_url or len(selected_styles) == 0))

async def run_streamlit_pipeline(url, styles, config):
    workdir = Path(tempfile.mkdtemp(prefix="st_vcap_"))
    
    try:
        # Step 1: Ingest
        status_box = st.empty()
        status_box.info("📥 Stage 1: Downloading & Downscaling Video...")
        
        video_path = await asyncio.to_thread(ingest.download_video, url, workdir)
        duration = await asyncio.to_thread(ingest.probe_duration, video_path)
        video_path = await asyncio.to_thread(ingest.downscale, video_path)
        
        status_box.info("🎞️ Stage 2: Extracting Keyframes & Audio...")
        
        # Step 2: Sampling
        n_frames = sampling.frame_count_for_duration(duration, config.frames_min, config.frames_max)
        frames = await asyncio.to_thread(sampling.sample_keyframes, video_path, n_frames, workdir / "frames")
        if not frames:
            frames = await asyncio.to_thread(sampling.sample_uniform, video_path, n_frames, workdir / "frames")
            
        # Display sampled frames
        if frames:
            with st.expander("🖼️ View Sampled Keyframes", expanded=False):
                cols = st.columns(min(len(frames), 5))
                for idx, frame_path in enumerate(frames):
                    cols[idx % 5].image(str(frame_path), caption=f"Frame {idx+1}")
        
        # Step 3: Transcription & OCR
        transcript = await asyncio.to_thread(audio.transcribe, video_path)
        ocr_text = await asyncio.to_thread(ocr.extract_text, frames)
        
        with st.expander("🗣️ / 🔍 Speech Transcription & OCR text", expanded=False):
            st.markdown(f"**Speech Transcript:** {transcript or 'None detected.'}")
            st.markdown(f"**Detected On-Screen Text (OCR):** {ocr_text or 'None detected.'}")
            
        # Step 4: Grounding
        status_box.info("🧠 Stage 3: Running Multimodal Factual Grounding (Gemma 4 31B IT)...")
        facts = await grounding.ground_video(frames, transcript, ocr_text, config)
        
        with st.expander("📄 Raw Grounding Facts", expanded=False):
            st.json(facts.model_dump_json())
            
        # Step 5: Verification (CoVe)
        status_box.info("🔍 Stage 4: Verifying facts using Chain-of-Verification (CoVe)...")
        verification_result = await verification.verify_facts(facts, frames, config)
        cleaned_facts = verification_result.cleaned_facts
        
        with st.expander("🛡️ Chain-of-Verification (CoVe) Details", expanded=False):
            st.markdown("**Verification Questions Generated:**")
            for q in verification_result.verification_questions:
                st.write(f"- {q}")
            st.markdown("**Answers Obtained from Visual Verification:**")
            for a in verification_result.verification_answers:
                st.write(f"- {a}")
            if verification_result.dropped_claims:
                st.warning(f"**Dropped Claims (Unsupported Factual Violations):** {verification_result.dropped_claims}")
            else:
                st.success("No facts were dropped during verification.")
                
        # Step 6: Styling
        status_box.info("✍️ Stage 5: Styling Captions per rubric definitions...")
        captions = await styling.style_all(cleaned_facts, styles, config)
        
        # Step 7: Self-Judging
        status_box.info("⚖️ Stage 6: Running Evaluator Judge (Gemma 4 26B A4B IT)...")
        
        # We perform the judging and get scores to present in the UI
        final_captions = {}
        judging_reports = {}
        
        for style in styles:
            cap_text = captions.get(style, config.fallback_caption)
            styled_cap = styling.StyledCaption(style=style, text=cap_text)
            
            # Run judge
            score = await judge.judge_caption(cleaned_facts, styled_cap, config)
            
            # If weak and retries > 0, run judge_and_regenerate logic
            if (score.accuracy < judge.JUDGE_REGENERATE_THRESHOLD or score.style_match < judge.JUDGE_REGENERATE_THRESHOLD) and config.max_self_judge_retries > 0:
                status_box.info(f"⚖️ Regenerating weak caption for style '{style}'...")
                improved_map = await judge.judge_and_regenerate(
                    facts=cleaned_facts,
                    captions={style: cap_text},
                    settings=config,
                    regenerate_fn=lambda f, s, cfg: styling.style_caption(f, s, cfg),
                    max_retries=config.max_self_judge_retries
                )
                cap_text = improved_map[style]
                styled_cap = styling.StyledCaption(style=style, text=cap_text)
                score = await judge.judge_caption(cleaned_facts, styled_cap, config)
                
            final_captions[style] = cap_text
            judging_reports[style] = score

        status_box.success("🎉 Captioning pipeline completed successfully!")
        
        # Display Final Caption Cards
        st.markdown("### 2. Generated Captions")
        
        # Grid of captions
        cols = st.columns(2)
        for idx, style in enumerate(styles):
            col = cols[idx % 2]
            with col:
                st.markdown(f"""
                <div class="caption-card">
                    <div class="caption-style">{style.replace('_', ' ')}</div>
                    <div class="caption-text">"{final_captions[style]}"</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show judge scores
                j_score = judging_reports[style]
                col.markdown(f"**Evaluator Score:** Accuracy: `{j_score.accuracy:.2f}`, Tone Match: `{j_score.style_match:.2f}`")
                if j_score.notes:
                    col.caption(f"📝 *Judge notes:* {j_score.notes}")
                    
    except Exception as e:
        st.error(f"Error during execution: {e}")
        st.exception(e)
    finally:
        # Cleanup temp directory
        await asyncio.to_thread(shutil.rmtree, workdir, ignore_errors=True)

# Run pipeline when clicked
if run_clicked:
    asyncio.run(run_streamlit_pipeline(video_url, selected_styles, settings))
