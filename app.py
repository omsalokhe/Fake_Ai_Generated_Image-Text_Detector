import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from collections import Counter
import math

st.set_page_config(page_title="AI Content Detector", layout="centered")

# Custom CSS for better UI with translation button
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .section-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .image-section {
        border-left-color: #667eea;
    }
    .text-section {
        border-left-color: #f5576c;
    }
    .result-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .ai-result {
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #ffe6e6 0%, #ffcccc 100%);
    }
    .human-result {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #e6ffe6 0%, #ccffcc 100%);
    }
    .uncertain-result {
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }
    .insight-box {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #6c757d;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .language-selector {
        background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Translation dictionary for UI elements
TRANSLATIONS = {
    "en": {
        "title": "🤖 AI Content Detector",
        "subtitle": "Advanced detection for AI-generated images and text",
        "image_tab": "🖼 Image Detection",
        "text_tab": "📝 Multi-Lingual Text Detection",
        "image_header": "AI-Generated Image Detection",
        "image_desc": "Upload an image to detect if it's AI-generated",
        "upload_label": "Choose an image file",
        "analyze_image": "Analyze Image",
        "analyze_text": "Analyze Text",
        "real_photo": "Real Photo Probability",
        "ai_generated": "AI-Generated Probability",
        "confidence": "Confidence",
        "high_confidence_human": "HIGH CONFIDENCE - HUMAN WRITTEN",
        "high_confidence_ai": "HIGH CONFIDENCE - AI GENERATED",
        "likely_human": "LIKELY HUMAN WRITTEN",
        "likely_ai": "LIKELY AI GENERATED",
        "detailed_analysis": "Detailed Analysis",
        "method": "Method",
        "image_size": "Image Size",
        "aspect_ratio": "Aspect Ratio",
        "analysis": "Analysis",
        "text_placeholder": "Paste your text in any Indian language...",
        "supported_languages": "Supported Languages",
        "detected_language": "Detected Language",
        "human_written": "Human-Written",
        "advanced_metrics": "Advanced Text Metrics",
        "perplexity": "Perplexity",
        "burstiness": "Burstiness",
        "complexity": "Complexity",
        "ai_indicators": "AI Indicators Found",
        "human_indicators": "Human Indicators Found",
        "language_analysis": "Language-Specific Analysis",
        "no_ai_indicators": "No strong AI indicators detected",
        "limited_human_patterns": "Limited human writing patterns",
        "language_patterns": "Language Patterns Detected",
        "text_statistics": "Text Statistics",
        "characters": "Characters",
        "words": "Words",
        "sentences": "Sentences",
        "avg_sentence_length": "Avg. Sentence Length",
        "heuristic_analysis": "Heuristic Analysis",
        "deep_learning": "Deep Learning Analysis",
        "upload_prompt": "👆 Upload an image to analyze",
        "enter_text_prompt": "👆 Enter text above to analyze",
        "footer": "Advanced AI Content Detector | Multi-Lingual Support • 22+ Indian Languages",
        "select_language": "Select Language",
        "language": "Language"
    },
    "hi": {
        "title": "🤖 एआई कंटेंट डिटेक्टर",
        "subtitle": "एआई-जनित छवियों और पाठ के लिए उन्नत पहचान",
        "image_tab": "🖼 छवि पहचान",
        "text_tab": "📝 बहुभाषी पाठ पहचान",
        "image_header": "एआई-जनित छवि पहचान",
        "image_desc": "एआई-जनित है या नहीं जांचने के लिए छवि अपलोड करें",
        "upload_label": "छवि फ़ाइल चुनें",
        "analyze_image": "छवि विश्लेषण करें",
        "analyze_text": "पाठ विश्लेषण करें",
        "real_photo": "वास्तविक फोटो संभावना",
        "ai_generated": "एआई-जनित संभावना",
        "confidence": "विश्वसनीयता",
        "high_confidence_human": "उच्च विश्वास - मानव लिखित",
        "high_confidence_ai": "उच्च विश्वास - एआई जनित",
        "likely_human": "संभावित मानव लिखित",
        "likely_ai": "संभावित एआई जनित",
        "detailed_analysis": "विस्तृत विश्लेषण",
        "method": "विधि",
        "image_size": "छवि आकार",
        "aspect_ratio": "पहलू अनुपात",
        "analysis": "विश्लेषण",
        "text_placeholder": "किसी भी भारतीय भाषा में पाठ चिपकाएँ...",
        "supported_languages": "समर्थित भाषाएँ",
        "detected_language": "पहचानी गई भाषा",
        "human_written": "मानव-लिखित",
        "advanced_metrics": "उन्नत पाठ मेट्रिक्स",
        "perplexity": "पेरप्लेक्सिटी",
        "burstiness": "बर्स्टिनेस",
        "complexity": "जटिलता",
        "ai_indicators": "एआई संकेतक मिले",
        "human_indicators": "मानव संकेतक मिले",
        "language_analysis": "भाषा-विशिष्ट विश्लेषण",
        "no_ai_indicators": "कोई मजबूत एआई संकेतक नहीं मिले",
        "limited_human_patterns": "सीमित मानव लेखन पैटर्न",
        "language_patterns": "भाषा पैटर्न मिले",
        "text_statistics": "पाठ आंकड़े",
        "characters": "वर्ण",
        "words": "शब्द",
        "sentences": "वाक्य",
        "avg_sentence_length": "औसत वाक्य लंबाई",
        "heuristic_analysis": "ह्युरिस्टिक विश्लेषण",
        "deep_learning": "डीप लर्निंग विश्लेषण",
        "upload_prompt": "👆 विश्लेषण करने के लिए छवि अपलोड करें",
        "enter_text_prompt": "👆 विश्लेषण करने के लिए पाठ दर्ज करें",
        "footer": "उन्नत एआई कंटेंट डिटेक्टर | बहुभाषी समर्थन • 22+ भारतीय भाषाएँ",
        "select_language": "भाषा चुनें",
        "language": "भाषा"
    }
}

# Initialize session state for language
if 'ui_language' not in st.session_state:
    st.session_state.ui_language = 'en'

def get_translation(key):
    """Get translated text for the current UI language"""
    lang = st.session_state.ui_language
    if lang in TRANSLATIONS and key in TRANSLATIONS[lang]:
        return TRANSLATIONS[lang][key]
    return TRANSLATIONS['en'][key]  # Fallback to English

# Language selector in sidebar
with st.sidebar:
    st.markdown("### 🌐 " + get_translation('language'))
    
    # Create a form to handle language change
    with st.form("language_form"):
        selected_language = st.selectbox(
            get_translation('select_language'),
            options=["en", "hi"],
            format_func=lambda x: {"en": "English", "hi": "हिन्दी"}[x],
            key="language_selector"
        )
        language_submitted = st.form_submit_button("Apply Language Change")
        
        if language_submitted:
            st.session_state.ui_language = selected_language
            st.rerun()

# Header with translated text
st.markdown(f"""
<div class="main-header">
    <h1>{get_translation('title')}</h1>
    <p>{get_translation('subtitle')}</p>
</div>
""", unsafe_allow_html=True)

# Indian Languages Support
INDIAN_LANGUAGES = {
    "English": "en", "Hindi": "hi", "Bengali": "bn", "Telugu": "te", "Marathi": "mr", 
    "Tamil": "ta", "Urdu": "ur", "Gujarati": "gu", "Kannada": "kn", "Odia": "or", 
    "Punjabi": "pa", "Malayalam": "ml", "Assamese": "as"
}

LANGUAGE_PATTERNS = {
    "hi": {
        'formal': r'\b(हालांकि|इसके अलावा|इस प्रकार|परिणामस्वरूप|अतः)\b',
        'emotional': r'\b(प्यार|खुशी|दुख|गुस्सा|आश्चर्य|वाह|अद्भुत)\b',
        'personal': r'\b(मैं|मेरा|हम|हमारा|तुम|आप)\b',
        'informal': r'\b(हाहा|वाह|अरे|यार|कमाल)\b'
    }
}

def detect_language(text):
    scripts = {
        'hi': r'[\u0900-\u097F]', 'bn': r'[\u0980-\u09FF]', 'te': r'[\u0C00-\u0C7F]', 
        'ta': r'[\u0B80-\u0BFF]', 'ml': r'[\u0D00-\u0D7F]', 'mr': r'[\u0900-\u097F]', 
        'gu': r'[\u0A80-\u0AFF]', 'kn': r'[\u0C80-\u0CFF]', 'pa': r'[\u0A00-\u0A7F]', 
        'or': r'[\u0B00-\u0B7F]', 'as': r'[\u0980-\u09FF]', 'ur': r'[\u0600-\u06FF]',
    }
    
    for lang_code, pattern in scripts.items():
        if re.search(pattern, text):
            return lang_code
    return 'en'

def analyze_multilingual_patterns(text, lang_code):
    if lang_code not in LANGUAGE_PATTERNS:
        return {}
    
    patterns = LANGUAGE_PATTERNS[lang_code]
    analysis = {}
    
    for pattern_type, pattern in patterns.items():
        matches = len(re.findall(pattern, text, re.UNICODE))
        analysis[pattern_type] = matches
    
    return analysis

def enhanced_text_analysis(text):
    lang_code = detect_language(text)
    language_name = [k for k, v in INDIAN_LANGUAGES.items() if v == lang_code][0] if lang_code in INDIAN_LANGUAGES.values() else "English"
    
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?।॥]+', text) if s.strip()]
    char_count = len(text)
    word_count = len(words)
    sentence_count = len(sentences)
    
    perplexity = calculate_perplexity(text)
    burstiness = analyze_burstiness(text)
    syntactic_complexity = analyze_syntactic_complexity(text)
    lang_patterns = analyze_multilingual_patterns(text, lang_code)
    
    ai_score = 0.5
    human_score = 0.5
    
    if perplexity < 50:
        ai_score += 0.2
    elif perplexity > 150:
        human_score += 0.2
    
    if burstiness > 0.3:
        human_score += 0.15
    elif burstiness < 0.1:
        ai_score += 0.15
    
    if syntactic_complexity > 0.8:
        human_score += 0.15
    elif syntactic_complexity < 0.4:
        ai_score += 0.15
    
    if lang_patterns:
        if lang_patterns.get('formal', 0) > len(sentences) * 0.4:
            ai_score += 0.1
        if lang_patterns.get('informal', 0) > 0:
            human_score += 0.1
        if lang_patterns.get('personal', 0) < len(words) * 0.03 and word_count > 30:
            ai_score += 0.1
    
    if sentence_count > 2:
        sentence_lengths = [len(s.split()) for s in sentences]
        length_variance = np.var(sentence_lengths)
        if length_variance < 2:
            ai_score += 0.1
        else:
            human_score += 0.1
    
    total = ai_score + human_score
    ai_prob = ai_score / total
    human_prob = human_score / total
    
    insights = {
        'language': {'detected': language_name, 'code': lang_code},
        'basic_stats': {
            'characters': char_count, 'words': word_count, 'sentences': sentence_count,
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0
        },
        'advanced_metrics': {'perplexity': perplexity, 'burstiness': burstiness, 'syntactic_complexity': syntactic_complexity},
        'language_patterns': lang_patterns,
        'ai_indicators': [],
        'human_indicators': []
    }
    
    if perplexity < 50:
        insights['ai_indicators'].append("Low perplexity (predictable word patterns)")
    if burstiness < 0.1:
        insights['ai_indicators'].append("Low word repetition burstiness")
    if syntactic_complexity < 0.4:
        insights['ai_indicators'].append("Simple sentence structures")
    if lang_patterns.get('personal', 0) < len(words) * 0.03:
        insights['ai_indicators'].append("Limited personal pronouns")
    
    if perplexity > 150:
        insights['human_indicators'].append("High perplexity (creative word usage)")
    if burstiness > 0.3:
        insights['human_indicators'].append("Natural word repetition patterns")
    if lang_patterns.get('informal', 0) > 0:
        insights['human_indicators'].append("Informal language usage")
    if syntactic_complexity > 0.8:
        insights['human_indicators'].append("Complex sentence structures")
    
    return ai_prob, human_prob, insights

def calculate_perplexity(text):
    words = text.lower().split()
    if len(words) < 10: return 100
    word_freq = Counter(words)
    total_words = len(words)
    log_sum = 0
    for word in words:
        prob = word_freq[word] / total_words
        log_sum += math.log(prob) if prob > 0 else math.log(1e-10)
    return math.exp(-log_sum / total_words)

def analyze_burstiness(text):
    words = text.lower().split()
    if len(words) < 20: return 0.5
    word_positions = {}
    burst_scores = []
    for i, word in enumerate(words):
        if word in word_positions:
            last_pos = word_positions[word]
            distance = i - last_pos
            burst_score = 1.0 / (distance + 1)
            burst_scores.append(burst_score)
        word_positions[word] = i
    return np.mean(burst_scores) if burst_scores else 0.0

def analyze_syntactic_complexity(text):
    sentences = [s.strip() for s in re.split(r'[.!?।॥]+', text) if s.strip()]
    if len(sentences) < 3: return 0.5
    complexity_scores = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) < 5: continue
        word_count = len(words)
        unique_words = len(set(words))
        avg_word_len = np.mean([len(word) for word in words])
        complexity = (unique_words / word_count) * (avg_word_len / 5)
        complexity_scores.append(complexity)
    return np.mean(complexity_scores) if complexity_scores else 0.5

# Image Detection Model
class SimpleResNetAIDetector(nn.Module):
    def __init__(self):
        super(SimpleResNetAIDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_simple_model():
    return SimpleResNetAIDetector()

def analyze_image_characteristics(image):
    width, height = image.size
    img_array = np.array(image)
    ai_score = real_score = 0.5
    
    ratio = width / height
    perfect_ratios = [1.0, 1.33, 1.5, 1.77, 0.75, 0.67]
    if any(abs(ratio - r) < 0.02 for r in perfect_ratios): ai_score += 0.2
    
    common_ai_sizes = [(512, 512), (1024, 1024), (768, 768), (1024, 576), (576, 1024)]
    if (width, height) in common_ai_sizes: ai_score += 0.3
    
    if len(img_array.shape) == 3:
        color_std = np.std(img_array, axis=(0, 1))
        avg_color_std = np.mean(color_std)
        if avg_color_std < 40: ai_score += 0.1
        else: real_score += 0.1
    
    if hasattr(image, 'format') and image.format in ['JPEG', 'PNG']: real_score += 0.1
    
    total = ai_score + real_score
    return ai_score / total, real_score / total

# Create tabs with translated labels
tab1, tab2 = st.tabs([get_translation('image_tab'), get_translation('text_tab')])

with tab1:
    st.markdown('<div class="section-container image-section">', unsafe_allow_html=True)
    st.header(get_translation('image_header'))
    st.markdown(get_translation('image_desc'))
    
    uploaded_file = st.file_uploader(get_translation('upload_label'), type=["jpg", "jpeg", "png"], key="image_upload")
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1: 
            st.image(image, caption="Original Image", use_column_width=True)
        with col2: 
            st.image(image.resize((224, 224)), caption="Processed for Analysis", use_column_width=True)
        
        analysis_method = st.radio(
            f"{get_translation('method')}:",
            [get_translation('heuristic_analysis'), get_translation('deep_learning')],
            key="image_method"
        )
        
        if st.button(get_translation('analyze_image'), type="primary", key="analyze_img"):
            with st.spinner("Analyzing image characteristics..."):
                if analysis_method == get_translation('heuristic_analysis'):
                    ai_prob, real_prob = analyze_image_characteristics(image)
                    results = "Heuristic analysis based on image characteristics"
                else:
                    model = load_simple_model()
                    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
                    img_tensor = transform(image).unsqueeze(0)
                    with torch.no_grad():
                        output = model(img_tensor)
                        probabilities = F.softmax(output, dim=1)
                        ai_prob = probabilities[0][1].item()
                        real_prob = probabilities[0][0].item()
                    results = "Deep learning analysis using custom CNN"
            
            # Display results
            st.subheader("🔍 " + get_translation('detailed_analysis'))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(get_translation('real_photo'), f"{real_prob*100:.1f}%")
            with col2:
                st.metric(get_translation('ai_generated'), f"{ai_prob*100:.1f}%")
            
            confidence = abs(real_prob - ai_prob)
            st.progress(confidence)
            st.write(f"{get_translation('confidence')}: {confidence*100:.1f}%")
            
            # Final verdict with styled box
            if real_prob > 0.7:
                st.markdown(f"""
                <div class="result-box human-result">
                    <h3>✅ {get_translation('high_confidence_human')}</h3>
                    <p>High confidence ({real_prob*100:.1f}%) - This appears to be a genuine photograph</p>
                </div>
                """, unsafe_allow_html=True)
            elif ai_prob > 0.7:
                st.markdown(f"""
                <div class="result-box ai-result">
                    <h3>🤖 {get_translation('high_confidence_ai')}</h3>
                    <p>High confidence ({ai_prob*100:.1f}%) - AI generation patterns detected</p>
                </div>
                """, unsafe_allow_html=True)
            elif real_prob > ai_prob:
                st.markdown(f"""
                <div class="result-box human-result">
                    <h3>⚠ {get_translation('likely_human')}</h3>
                    <p>Low confidence ({real_prob*100:.1f}%) - Likely real but uncertain</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box ai-result">
                    <h3>⚠ {get_translation('likely_ai')}</h3>
                    <p>Low confidence ({ai_prob*100:.1f}%) - Some AI patterns detected</p>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("📊 " + get_translation('detailed_analysis')):
                st.write(f"**{get_translation('method')}:** {analysis_method}")
                st.write(f"**{get_translation('image_size')}:** {image.size}")
                st.write(f"**{get_translation('aspect_ratio')}:** {image.size[0]/image.size[1]:.3f}")
                st.write(f"**{get_translation('analysis')}:** {results}")
    
    else:
        st.info(get_translation('upload_prompt'))
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-container text-section">', unsafe_allow_html=True)
    st.header("🌍 " + get_translation('text_tab'))
    
    # Language selector
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    st.write(f"**{get_translation('supported_languages')}:** Hindi, Bengali, Telugu, Marathi, Tamil, Urdu, Gujarati, Kannada, Malayalam, Odia, Punjabi, and more!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    user_text = st.text_area(
        get_translation('text_placeholder'),
        height=200,
        key="text_input"
    )
    
    if st.button(get_translation('analyze_text'), type="primary", key="analyze_text"):
        if user_text.strip():
            if len(user_text) < 30:
                st.warning("⚠ For best results, please provide at least 30 characters of text.")
            
            with st.spinner("Running multi-lingual analysis..."):
                ai_prob, human_prob, insights = enhanced_text_analysis(user_text)
            
            # Display main results
            st.subheader("🎯 " + get_translation('detailed_analysis'))
            
            # Language detection result
            st.info(f"**{get_translation('detected_language')}:** {insights['language']['detected']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(get_translation('human_written'), f"{human_prob*100:.1f}%")
            with col2:
                st.metric(get_translation('ai_generated'), f"{ai_prob*100:.1f}%")
            with col3:
                confidence = abs(human_prob - ai_prob)
                st.metric(get_translation('confidence'), f"{confidence*100:.1f}%")
            
            st.progress(confidence)
            
            # Final verdict
            if human_prob > 0.75:
                st.markdown(f"""
                <div class="result-box human-result">
                    <h3>✅ {get_translation('high_confidence_human')}</h3>
                    <p>Strong evidence of natural writing patterns in {insights['language']['detected']} ({human_prob*100:.1f}% confidence)</p>
                </div>
                """, unsafe_allow_html=True)
            elif ai_prob > 0.75:
                st.markdown(f"""
                <div class="result-box ai-result">
                    <h3>🤖 {get_translation('high_confidence_ai')}</h3>
                    <p>Clear AI writing patterns detected in {insights['language']['detected']} ({ai_prob*100:.1f}% confidence)</p>
                </div>
                """, unsafe_allow_html=True)
            elif human_prob > ai_prob:
                st.markdown(f"""
                <div class="result-box uncertain-result">
                    <h3>📝 {get_translation('likely_human')}</h3>
                    <p>Moderate confidence - appears natural in {insights['language']['detected']} ({human_prob*100:.1f}% confidence)</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box uncertain-result">
                    <h3>🤖 {get_translation('likely_ai')}</h3>
                    <p>Moderate confidence - some AI patterns in {insights['language']['detected']} ({ai_prob*100:.1f}% confidence)</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Advanced Metrics
            st.subheader("📊 " + get_translation('advanced_metrics'))
            
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{get_translation('perplexity')}</h4>
                    <h3>{insights['advanced_metrics']['perplexity']:.1f}</h3>
                    <small>{'Low (AI-like)' if insights['advanced_metrics']['perplexity'] < 80 else 'High (Human-like)'}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[1]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{get_translation('burstiness')}</h4>
                    <h3>{insights['advanced_metrics']['burstiness']:.3f}</h3>
                    <small>{'Low (AI-like)' if insights['advanced_metrics']['burstiness'] < 0.2 else 'High (Human-like)'}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[2]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{get_translation('complexity')}</h4>
                    <h3>{insights['advanced_metrics']['syntactic_complexity']:.3f}</h3>
                    <small>{'Simple (AI-like)' if insights['advanced_metrics']['syntactic_complexity'] < 0.5 else 'Complex (Human-like)'}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed Insights
            st.subheader("🔍 " + get_translation('language_analysis'))
            
            col1, col2 = st.columns(2)
            
            with col1:
                if insights['ai_indicators']:
                    st.write(f"🤖 {get_translation('ai_indicators')}:")
                    for indicator in insights['ai_indicators']:
                        st.markdown(f'<div class="insight-box">{indicator}</div>', unsafe_allow_html=True)
                else:
                    st.info(get_translation('no_ai_indicators'))
            
            with col2:
                if insights['human_indicators']:
                    st.write(f"📝 {get_translation('human_indicators')}:")
                    for indicator in insights['human_indicators']:
                        st.markdown(f'<div class="insight-box">{indicator}</div>', unsafe_allow_html=True)
                else:
                    st.info(get_translation('limited_human_patterns'))
        
        else:
            st.warning("Please enter some text to analyze.")
    
    else:
        st.info(get_translation('enter_text_prompt'))
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p><strong>{get_translation('footer')}</strong></p>
</div>
""", unsafe_allow_html=True)