import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RAG.RAG import RAG
from Classifier.classification import classify_text

rag = RAG()

documents = [
    "The Nikon D3500 is a super approachable DSLR for beginners. It's lightweight, easy to handle, and takes beautiful 24MP photos without overwhelming you with too many buttons. Even if you're just snapping vacation shots or portraits, the D3500 gives you crisp images and decent Full HD video. The battery lasts ages, so you won't be scrambling for a charger mid-trip. Perfect for anyone starting out in photography.",
    "Imagine capturing every detail of a bustling street market or a serene sunset in stunning clarity—that's what the Canon EOS 90D delivers. With its 32.5MP sensor, every shot feels alive, every color vibrant. Whether you're chasing wildlife or vlogging your travels, its fast autofocus and 4K video make sure you never miss the moment. Holding it feels solid yet comfortable, like it was made for adventures that last all day.",
    "The Sony Alpha a6400 features a 24.2MP APS-C Exmor CMOS sensor paired with a BIONZ X image processor, delivering high-resolution images with low noise even in challenging lighting. It offers real-time Eye AF, 425 phase-detection points, and 11 fps continuous shooting for action photography. Video enthusiasts will appreciate 4K recording without pixel binning, S-Log3 gamma, and a tiltable 3-inch LCD for vlogging or high-angle shots. Connectivity includes Wi-Fi and Bluetooth for rapid file sharing.",
    "Looking for a camera that's perfect for Instagram, TikTok, or YouTube? The Canon PowerShot G7 X Mark III has you covered! Its pocket-sized body packs a 20MP sensor and a bright f/1.8–2.8 lens, perfect for low-light selfies or cafe photography. Flip the screen, hit record in 4K, and start live streaming instantly. Lightweight, sleek, and stylish—this little powerhouse is ready for your next viral post.",
    "Elevate your photography with the Fujifilm X-T4, the mirrorless camera that combines cutting-edge performance with stunning design. From breathtaking landscapes to cinematic video, its 26.1MP sensor and 5-axis in-body stabilization ensure every shot is sharp and vibrant. Shoot at up to 15 fps or record 4K/60fps video with ease. Stylish, rugged, and versatile, the X-T4 empowers both creatives and professionals to capture their vision flawlessly.",
    "Meet the GoPro HERO11 Black — the tiny camera that refuses to sit still. Rain, snow, mud, or your cousin doing backflips into a pool? No problem. This little action hero shoots 5.3K video, 27MP photos, and keeps everything buttery smooth with HyperSmooth 5.0 stabilization. It's waterproof, rugged, and basically indestructible… though we don't recommend testing that theory. Perfect for adventurers, thrill-seekers, or anyone who wants to make their friends jealous with epic videos.",
    "The Nikon Z50 is a compact mirrorless camera with a 20.9MP APS-C sensor that delivers clean, detailed images. It shoots 4K video, has fast hybrid autofocus, and a tilting touchscreen for flexible framing. Lightweight and easy to carry, it's perfect for everyday photography without any unnecessary bells and whistles."
]

for doc in documents:
    rag.add_document(doc)

st.set_page_config(layout="wide") 
st.title("LLM Manipulation")

RESET_TEXT = "For those who live and breathe photography, the Fujifilm X-T3 is a dream come true. Its 26.1MP X-Trans sensor captures every nuance of light and color, while the blazing-fast autofocus and 11 fps burst mode let you freeze fleeting moments with perfection. With 4K video, weather-sealed construction, and a tactile, retro-inspired body, it's not just a camera—it's an instrument for creating art."

TEXT_KEY = "user_input_area"
OUTPUT_KEY = "llm_output"

if TEXT_KEY not in st.session_state:
    st.session_state[TEXT_KEY] = RESET_TEXT

if OUTPUT_KEY not in st.session_state:
    st.session_state[OUTPUT_KEY] = "Output will appear here after submission."

def handle_reset_click():
    st.session_state[TEXT_KEY] = RESET_TEXT
    st.session_state[OUTPUT_KEY] = "Output will appear here after submission."
    st.session_state["retrieved_flag"] = None

def handle_submit_click(k):
    input_text = st.session_state[TEXT_KEY]

    class_0_prob, class_1_prob = classify_text(input_text)

    st.session_state["class_0_prob"] = class_0_prob
    st.session_state["class_1_prob"] = class_1_prob

    if class_1_prob < 0.5:
        rag.add_document(input_text)
    
    output = rag.query(prompt, model=st.session_state["model"], k=k)
    st.session_state[OUTPUT_KEY] = output
    
    retrieved_docs = rag.retrieve_documents(prompt, k=k)
    
    if input_text in retrieved_docs:
        st.session_state["retrieved_flag"] = True
        st.session_state["retrieved_rank"] = retrieved_docs.index(input_text) + 1  # Get rank (1-based index)
    else:
        st.session_state["retrieved_flag"] = False
        st.session_state["retrieved_rank"] = "N/A"  # Not retrieved
    
    if class_1_prob < 0.42:
        rag.remove_document(input_text)

left_col, middle_col, right_col = st.columns(3)

with left_col:
    st.header("Documents")
    with st.container(height=500):
        for doc in documents:
            st.write(doc)
            st.write("---")

with middle_col:
    st.header("Interaction")
    prompt = st.text_input("Prompt", value="Give me some camera recommendation for beginner", disabled=False)
    
    col1, col2 = st.columns(2)
    with col1:
        k = st.slider("Number of Documents to Retrieve", min_value=1, max_value=10, value=5)
    with col2:
        col_21, col_22 = st.columns([1, 3])
        with col_22:
            st.session_state["model"] = st.radio(
                "Select one:",
                options=["gemini", "mistral"],
                index=0
            )
    
    st.text_area(
        "New Document",
        key=TEXT_KEY,
        height=250
    )

    _, _, _, col4, col5= st.columns(5)
    with col4:
        st.button("Reset", on_click=handle_reset_click)
    with col5:
        st.button("Submit", on_click=handle_submit_click, type="primary", args=(k,))
        

with right_col:
    st.subheader("LLM Output")
    st.text_area(
        "Output", 
        value=st.session_state[OUTPUT_KEY], 
        height=300, 
        disabled=True, 
        label_visibility="collapsed"
    )
    
    retrieved_flag = st.session_state.get("retrieved_flag", None)
    retrieved_rank = st.session_state.get("retrieved_rank", "N/A")
    
    if retrieved_flag is True:
        st.markdown("**New Document Retrieved:** ✅")
        st.markdown(f"**Rank in Retrieved Documents:** {retrieved_rank}")
    elif retrieved_flag is False:
        st.markdown("**New Document Retrieved:** ❌")
        st.markdown(f"**Rank in Retrieved Documents:** {retrieved_rank}")
    else:
        st.markdown("**New Document Retrieved:** Not Checked Yet")
        st.markdown(f"**Rank in Retrieved Documents:** {retrieved_rank}")
    
    class_0_prob = st.session_state.get("class_0_prob", "N/A")
    class_1_prob = st.session_state.get("class_1_prob", "N/A")
    
    st.markdown(f"**Malicious Text Probability:** {class_1_prob}")