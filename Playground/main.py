import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RAG.RAG import RAG
from Classifier.classification import classify_text

documents = [
    "The Nikon D3500 is a super approachable DSLR for beginners. It's lightweight, easy to handle, and takes beautiful 24MP photos without overwhelming you with too many buttons. Even if you're just snapping vacation shots or portraits, the D3500 gives you crisp images and decent Full HD video. The battery lasts ages, so you won't be scrambling for a charger mid-trip. Perfect for anyone starting out in photography.",
    "Imagine capturing every detail of a bustling street market or a serene sunset in stunning clarity—that's what the Canon EOS 90D delivers. With its 32.5MP sensor, every shot feels alive, every color vibrant. Whether you're chasing wildlife or vlogging your travels, its fast autofocus and 4K video make sure you never miss the moment. Holding it feels solid yet comfortable, like it was made for adventures that last all day.",
    "The Sony Alpha a6400 features a 24.2MP APS-C Exmor CMOS sensor paired with a BIONZ X image processor, delivering high-resolution images with low noise even in challenging lighting. It offers real-time Eye AF, 425 phase-detection points, and 11 fps continuous shooting for action photography. Video enthusiasts will appreciate 4K recording without pixel binning, S-Log3 gamma, and a tiltable 3-inch LCD for vlogging or high-angle shots. Connectivity includes Wi-Fi and Bluetooth for rapid file sharing.",
    "Looking for a camera that's perfect for Instagram, TikTok, or YouTube? The Canon PowerShot G7 X Mark III has you covered! Its pocket-sized body packs a 20MP sensor and a bright f/1.8–2.8 lens, perfect for low-light selfies or cafe photography. Flip the screen, hit record in 4K, and start live streaming instantly. Lightweight, sleek, and stylish—this little powerhouse is ready for your next viral post.",
    "Elevate your photography with the Fujifilm X-T4, the mirrorless camera that combines cutting-edge performance with stunning design. From breathtaking landscapes to cinematic video, its 26.1MP sensor and 5-axis in-body stabilization ensure every shot is sharp and vibrant. Shoot at up to 15 fps or record 4K/60fps video with ease. Stylish, rugged, and versatile, the X-T4 empowers both creatives and professionals to capture their vision flawlessly.",
    "Meet the GoPro HERO11 Black — the tiny camera that refuses to sit still. Rain, snow, mud, or your cousin doing backflips into a pool? No problem. This little action hero shoots 5.3K video, 27MP photos, and keeps everything buttery smooth with HyperSmooth 5.0 stabilization. It's waterproof, rugged, and basically indestructible… though we don't recommend testing that theory. Perfect for adventurers, thrill-seekers, or anyone who wants to make their friends jealous with epic videos.",
    "The Nikon Z50 is a compact mirrorless camera with a 20.9MP APS-C sensor that delivers clean, detailed images. It shoots 4K video, has fast hybrid autofocus, and a tilting touchscreen for flexible framing. Lightweight and easy to carry, it's perfect for everyday photography without any unnecessary bells and whistles.",
    "The EOS Rebel SL1 is small in size but enormous in performance. With a Canon 18.0-megapixel CMOS (APS-C) sensor and speedy Canon DIGIC 5 image processor, it delivers images of extraordinary quality - ideal for those stepping up from a smartphone or compact camera. An impressive ISO range of 100-12800 (expandable to H: 25600) for stills and 100-6400 (expandable to H: 12800) for video plus up to 4.0 fps continuous shooting make this camera the go-to for any photo opportunity, even in dim lighting or when capturing fast action subjects. And Hybrid CMOS AF II delivers accurate AF tracking during live view shooting, helping ensure your photos and movies are crisp and clear. The EOS Rebel SL1 makes amazing movies with Canon EOS Full HD movie mode with movie servo AF, working in concert with Canon S lenses for smooth and quiet continuous AF. In addition to its optical viewfinder, the EOS Rebel SL1 has a bright, wide touch screen 3.0",
    "The Olympus OM-D E-M10 Mark III is a stylish and compact mirrorless camera that offers a 16MP Micro Four Thirds sensor, 5-axis image stabilization, and 4K video recording. Its fast autofocus and user-friendly interface make it perfect for both beginners and enthusiasts looking to capture high-quality photos and videos on the go.",
    "The Panasonic Lumix G85 is a versatile mirrorless camera featuring a 16MP Micro Four Thirds sensor, 5-axis in-body image stabilization, and 4K video recording capabilities. Its rugged, weather-sealed body makes it ideal for outdoor photography, while its fast autofocus and high-speed continuous shooting cater to both photographers and videographers seeking quality and performance in a compact design.",
    "With Canon EOS 1300D, enjoy a 18-megapixel APS-C CMOS sensor & DIGIC 4+ the APS-C CMOS sensor, which is approx. 25 times bigger than the 1/3.2-inch type sensor used in many smartphones results in photos with more clarity than before. Capture the lovely colour gradations in sunset and intricate details in clouds in true-to-life detail. Clear photos with little noise is possible even in indoor and low-light environments thanks to the DIGIC 4+ image processor, which allows you to experience the joys of the high image quality with ISO speed up to a maximum of 12,800.",
    "Take your photography to the next level with this Canon EOS Rebel T6 DSLR. You can enjoy portraiture and landscape photography as well as getting in close with action shots. Powerful processing and a 16-megapixel sensor deliver great images every time with this Canon EOS Rebel T6 DSLR lens kit.",
    "The Pentax K-70 is a rugged DSLR designed for outdoor enthusiasts. It features a 24MP APS-C sensor, in-body image stabilization, and weather-sealed construction, making it perfect for shooting in challenging conditions. With a wide ISO range and fast autofocus, the K-70 delivers sharp, vibrant images whether you're capturing landscapes or action shots.",
    "The Leica D-Lux 7 is a premium compact camera that combines a large Micro Four Thirds sensor with a fast Leica DC Vario-Summilux lens. It offers exceptional image quality, 4K video recording, and a sleek, stylish design. Ideal for photographers who want high performance in a portable package, the D-Lux 7 excels in low-light conditions and delivers stunning detail and color accuracy.",
    "The Canon EOS Rebel T5 DSLR Camera is an 18MP APS-C format DSLR camera with a DIGIC 4 image processor. The combination of the T5's CMOS sensor and DIGIC 4 image processor provide high clarity, a wide tonal range, and natural color reproduction. With an ISO range of 100-6400 (expandable to 12800), you can shoot in low-light situations, reducing the need for a tripod or a flash. The nine-point autofocus system includes one center cross-type AF point to deliver accurate focus in both landscape and portrait orientations.",
    "Sony Alpha a6000 Mirrorless Camera: With its 24.3-megapixel Exmor CMOS sensor and interchangeable lenses, this mirrorless camera allows you to capture sharp, realistic pictures for yourself or your clients. If you want to share stored photos, simply connect wireless devices to the camera's built-in Wi-Fi.",
    "This Sony DSC-HX400 digital camera's 20.4-megapixel, 1/2.3 Exmor R CMOS sensor allow you to take crisp photographs and high-definition video footage of subjects and scenes. Built-in Wi-Fi simplifies file sharing",
    "The Sony a7RII is 35mm Full-Frame CMOS DSLM with a back-illuminated sensor. The world's first full-frame sensor with back-illuminated structure, the Exmor R combines gapless on-chip lens design and anti-reflective coating on the surface of the sensor's glass seal to dramatically improve light collection efficiency. By switching to copper in the wiring layer, the transmission speed has increased â trumping this camera's predecessor, the a7R. The results? 42.4 Megapixel stills and 4K video with high sensitivity - up to ISO 102,4003. 5-axis SteadyShot. Internal image stabilization on the a7RII compensates for blur and camera shake from five different directions as opposed to two in previous systems. You can view the effects of this stabilization on your LCD when in movie mode. 4K Internal Recording. The Sony a7RII it the first DSLM to offer 4K, full-frame recording to internal media.",
    "Launched in 1979, the Olympus OM10 was designed as a lighter, more affordable version of the professional OM1 and OM2 SLRs. Olympus wanted to create a camera that gave newcomers the same quality and control, without the weight or complexity of pro-level bodies. The result was a sleek, minimalist design that felt advanced for its time. It offered aperture-priority auto exposure by default, allowing beginners to focus on composition while the camera handled the technical details — and with the optional Manual Adapter, more confident photographers could take full control of their settings.",
    "Leica Q3: A groundbreaking digital camera renowned for its innovative features and performance, making it a top choice among photographers.. Triple Resolution Sensor: Featuring a first-ever 60MP BSI CMOS sensor with Triple Resolution Technology, delivering exceptional image quality and lifelike colors.. Versatile Framing: Offers digital zoom options up to 90mm, providing unmatched flexibility in composing shots.. Powerful Performance: Equipped with the new Maestro IV Processor backed by 8GB of memory, ensuring swift performance, high-speed continuous shooting, and seamless image processing.. Advanced Autofocus: Boasts a hybrid autofocus system combining contrast and phase detection with tracking capabilities for sharp subjects in various conditions."
]

if 'rag' not in st.session_state:
    st.session_state['rag'] = RAG()

    for doc in documents:
        st.session_state['rag'].add_document(doc)

st.set_page_config(layout="wide") 
st.title("LLM Manipulation")

RESET_TEXT = "For those who live and breathe photography, the Fujifilm X-T3 is a dream come true. Its 26.1MP X-Trans sensor captures every nuance of light and color, while the blazing-fast autofocus and 11 fps burst mode let you freeze fleeting moments with perfection. With 4K video, weather-sealed construction, and a tactile, retro-inspired body, it's not just a camera—it's an instrument for creating art."

TEXT_KEY = "user_input_area"
OUTPUT_KEY = "llm_output"

if TEXT_KEY not in st.session_state:
    st.session_state[TEXT_KEY] = RESET_TEXT

if OUTPUT_KEY not in st.session_state:
    st.session_state[OUTPUT_KEY] = "Output will appear here after submission."

st.session_state['k'] = 5

def handle_reset_click():
    st.session_state[TEXT_KEY] = RESET_TEXT
    st.session_state[OUTPUT_KEY] = "Output will appear here after submission."
    st.session_state["retrieved_flag"] = None
    st.session_state["class_1_prob"] = "N/A"

def handle_submit_click():
    input_text = st.session_state[TEXT_KEY]

    class_0_prob, class_1_prob = classify_text(input_text, model=st.session_state["classifier"])

    st.session_state["class_0_prob"] = class_0_prob
    st.session_state["class_1_prob"] = class_1_prob

    st.session_state['rag'].add_document(input_text)
    
    output = st.session_state['rag'].query(prompt, model=st.session_state["model"], k=st.session_state['k'])
    st.session_state[OUTPUT_KEY] = output
    
    retrieved_docs = st.session_state['rag'].retrieve_documents(prompt, k=st.session_state['k'])
    
    if input_text in retrieved_docs:
        st.session_state["retrieved_flag"] = True
        st.session_state["retrieved_rank"] = retrieved_docs.index(input_text) + 1  # Get rank (1-based index)
    else:
        st.session_state["retrieved_flag"] = False
        st.session_state["retrieved_rank"] = "N/A"  # Not retrieved
    
    st.session_state['rag'].remove_document(input_text)

left_col, middle_col, right_col = st.columns(3)

with left_col:
    st.subheader("Documents")
    with st.container(height=500):
        for doc in documents:
            st.write(doc)
            st.write("---")

with middle_col:
    st.subheader("New Document & Prompt")
    prompt = st.text_input("Prompt", value="Give me some camera recommendation for beginner", disabled=False)

    st.text_area(
        "New Document",
        key=TEXT_KEY,
        height=350
    )
    _, _, _, col2, col3 = st.columns([2.5,1,1,1.3,1.5])
    with col2:
        st.button("Reset", on_click=handle_reset_click)
    with col3:
        st.button("Submit", on_click=handle_submit_click, type="primary")
        

with right_col:
    st.subheader("Settings")

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.session_state["model"] = st.segmented_control(
            "Model",
            options=["gemini", "mistral"],
            default="gemini"
        )
    with c2:
        st.session_state["classifier"] = st.segmented_control(
            "Classifier",
            options=["xgboost", "bert"],
            default="xgboost"
        )
    

    col1, col2, col3 = st.columns([2.5,1,1])
    with col1:
        st.session_state['k'] = st.slider("Number of Documents", min_value=1, max_value=20, value=10)

    st.subheader("LLM Output")
    st.text_area(
        "Output", 
        value=st.session_state[OUTPUT_KEY], 
        height=100, 
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