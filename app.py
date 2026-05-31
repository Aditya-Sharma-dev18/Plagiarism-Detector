# ============================================
# AI PLAGIARISM DETECTOR - ACCURATE VERSION
# N-gram Overlap + Synonym Matching + Chunk Detection
# ============================================

from flask import Flask, request, render_template, send_file
import os
import io
import re
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# ============================================
# SYNONYM DICTIONARY
# ============================================
synonym_dict = {
    "learn": ["study", "understand", "grasp"],
    "develop": ["build", "create", "construct", "make", "design"],
    "use": ["utilize", "employ", "apply", "leverage"],
    "help": ["assist", "aid", "support", "facilitate"],
    "important": ["crucial", "essential", "vital", "significant", "critical"],
    "popular": ["widely used", "common", "prevalent", "famous"],
    "powerful": ["strong", "robust", "potent", "effective"],
    "complex": ["complicated", "intricate", "difficult", "challenging"],
    "solve": ["resolve", "address", "tackle", "handle"],
    "enable": ["allow", "permit", "let"],
    "fundamental": ["basic", "core", "essential", "foundational"],
    "efficient": ["optimized", "fast", "effective"],
    "deploy": ["launch", "release", "implement"],
    "field": ["branch", "area", "domain", "sector"],
    "programming": ["coding", "development"],
    "algorithm": ["method", "procedure", "technique"],
    "framework": ["library", "toolkit", "platform"],
    "artificial intelligence": ["ai", "machine intelligence"],
    "machine learning": ["ml", "statistical learning"],
    "large": ["big", "huge", "extensive", "massive"],
    "many": ["numerous", "several", "various", "multiple"],
    "good": ["great", "excellent", "fine", "superior"],
    "easy": ["simple", "straightforward", "effortless"],
}


# ============================================
# TEXT EXTRACTION
# ============================================

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)


def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return ""


# ============================================
# TEXT PROCESSING
# ============================================

def clean_text(text):
    """Remove special characters, extra spaces"""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def get_ngrams(words, n):
    """Generate n-grams from word list"""
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ============================================
# ACCURATE SIMILARITY CALCULATION
# ============================================

def calculate_ngram_overlap(text1, text2, n=4):
    """
    N-gram overlap - Best for detecting exact copy
    Uses 4-word sequences (tetragrams)
    """
    words1 = clean_text(text1).split()
    words2 = clean_text(text2).split()
    
    if len(words1) < n or len(words2) < n:
        return 0
    
    ngrams1 = set(get_ngrams(words1, n))
    ngrams2 = set(get_ngrams(words2, n))
    
    if not ngrams1 or not ngrams2:
        return 0
    
    overlap = ngrams1.intersection(ngrams2)
    return round(len(overlap) / len(ngrams1.union(ngrams2)) * 100, 2)


def calculate_jaccard_similarity(text1, text2):
    """Jaccard similarity on words"""
    words1 = set(clean_text(text1).split())
    words2 = set(clean_text(text2).split())
    
    if not words1 or not words2:
        return 0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return round(len(intersection) / len(union) * 100, 2)


def normalize_text(text):
    """Replace synonyms with base words"""
    text_lower = text.lower()
    for base_word, synonyms in synonym_dict.items():
        for syn in synonyms:
            if syn in text_lower:
                text_lower = text_lower.replace(syn, base_word)
    return text_lower


def calculate_semantic_similarity(text1, text2):
    """Semantic similarity using synonym normalization + Jaccard"""
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    return calculate_jaccard_similarity(norm1, norm2)


def calculate_hybrid_similarity(text1, text2):
    """
    Hybrid Score:
    - 50% N-gram overlap (direct copy detection)
    - 30% Jaccard (word overlap)
    - 20% Semantic (synonym normalized)
    """
    ngram_score = calculate_ngram_overlap(text1, text2)
    jaccard_score = calculate_jaccard_similarity(text1, text2)
    semantic_score = calculate_semantic_similarity(text1, text2)
    
    hybrid = round(ngram_score * 0.5 + jaccard_score * 0.3 + semantic_score * 0.2, 2)
    
    return {
        "ngram": ngram_score,
        "jaccard": jaccard_score,
        "semantic": semantic_score,
        "hybrid": hybrid
    }


def find_matching_sentences(sentences1, sentences2, threshold=25):
    """Find matching sentences between two documents"""
    matches = []
    
    for i, s1 in enumerate(sentences1):
        for j, s2 in enumerate(sentences2):
            # Direct n-gram overlap
            direct_score = calculate_ngram_overlap(s1, s2, n=3)
            
            # Semantic (synonym-based)
            semantic_score = calculate_semantic_similarity(s1, s2)
            
            best_score = max(direct_score, semantic_score)
            
            if best_score >= threshold:
                # Determine type
                if direct_score >= semantic_score:
                    match_type = "Direct Copy"
                    match_score = direct_score
                else:
                    match_type = "Paraphrased"
                    match_score = semantic_score
                
                matches.append({
                    "sent1_index": i,
                    "sent2_index": j,
                    "score": match_score,
                    "direct_score": direct_score,
                    "semantic_score": semantic_score,
                    "type": match_type,
                    "text1": s1[:200] + "..." if len(s1) > 200 else s1,
                    "text2": s2[:200] + "..." if len(s2) > 200 else s2,
                })
    
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:10]


# ============================================
# PDF REPORT
# ============================================

def generate_pdf_report(result):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=20,
                                   textColor=colors.HexColor('#6c5ce7'), alignment=1, spaceAfter=20)
    heading_style = ParagraphStyle('H', parent=styles['Heading2'], fontSize=14,
                                     textColor=colors.HexColor('#333'), spaceBefore=15, spaceAfter=8)
    normal_style = ParagraphStyle('N', parent=styles['Normal'], fontSize=9, leading=13)
    match_style = ParagraphStyle('M', parent=styles['Normal'], fontSize=9, leading=13,
                                   textColor=colors.HexColor('#c62828'))

    elements = []

    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("AI Plagiarism Detection Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", normal_style))
    elements.append(Spacer(1, 0.3*inch))

    elements.append(Paragraph(f"<b>Document 1:</b> {result.get('doc1_name', 'N/A')}", normal_style))
    elements.append(Paragraph(f"<b>Document 2:</b> {result.get('doc2_name', 'N/A')}", normal_style))
    elements.append(Spacer(1, 0.2*inch))

    score_data = [
        [Paragraph('<b>Detection Type</b>', normal_style), Paragraph('<b>Score</b>', normal_style)],
        [Paragraph('Overall Similarity', normal_style), Paragraph(f"{result.get('hybrid_similarity', 0)}%", normal_style)],
        [Paragraph('N-gram Overlap (Exact Copy)', normal_style), Paragraph(f"{result.get('ngram_similarity', 0)}%", normal_style)],
        [Paragraph('Word Overlap (Jaccard)', normal_style), Paragraph(f"{result.get('jaccard_similarity', 0)}%", normal_style)],
        [Paragraph('Semantic Match (Meaning)', normal_style), Paragraph(f"{result.get('semantic_similarity', 0)}%", normal_style)],
        [Paragraph('Direct Copy Detected', normal_style), Paragraph(f"{result.get('direct_match_percent', 0)}%", normal_style)],
        [Paragraph('Paraphrase Detected', normal_style), Paragraph(f"{result.get('paraphrase_percent', 0)}%", normal_style)],
    ]

    score_table = Table(score_data, colWidths=[3.5*inch, 1.5*inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6c5ce7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ddd')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(score_table)
    elements.append(Spacer(1, 0.3*inch))

    if result.get('matches'):
        elements.append(Paragraph("Top Matching Sections", heading_style))
        for i, match in enumerate(result['matches'][:5], 1):
            elements.append(Paragraph(f"<b>#{i} [{match['type']}] — {match['score']:.1f}%</b>", match_style))
            elements.append(Paragraph(f"Doc1: {match['text1'][:150]}...", normal_style))
            elements.append(Paragraph(f"Doc2: {match['text2'][:150]}...", normal_style))
            elements.append(Spacer(1, 0.1*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ============================================
# FLASK ROUTES
# ============================================

@app.route("/")
def home():
    return render_template("index.html", result=None)


@app.route('/check', methods=['POST'])
def check_plagiarism():

    file1 = request.files.get('document1')
    file2 = request.files.get('document2')

    if not file1 or not file2:
        return render_template("index.html", result={"error": "Please upload both documents"})

    path1 = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
    file1.save(path1)
    file2.save(path2)

    text1 = extract_text(path1)
    text2 = extract_text(path2)

    if not text1.strip() or not text2.strip():
        return render_template("index.html", result={"error": "Could not extract text"})

    # Overall similarity
    scores = calculate_hybrid_similarity(text1, text2)

    # Sentence-level matching
    sentences1 = split_into_sentences(text1)
    sentences2 = split_into_sentences(text2)
    matches = find_matching_sentences(sentences1, sentences2)

    # Calculate direct vs paraphrase percentages
    direct_match_percent = 0
    paraphrase_percent = 0

    if matches:
        direct_count = sum(1 for m in matches if m['type'] == 'Direct Copy')
        paraphrase_count = len(matches) - direct_count
        total_matches = len(matches)
        
        if total_matches > 0:
            direct_match_percent = round((direct_count / total_matches) * 100, 2)
            paraphrase_percent = round((paraphrase_count / total_matches) * 100, 2)

    # Verdict
    hybrid_score = scores['hybrid']
    if hybrid_score > 60:
        verdict = "🔴 High Plagiarism Detected"
        verdict_class = "danger"
    elif hybrid_score > 30:
        verdict = "🟡 Moderate Similarity"
        verdict_class = "warning"
    else:
        verdict = "🟢 Mostly Original Content"
        verdict_class = "success"

    result = {
        "doc1_name": file1.filename,
        "doc2_name": file2.filename,
        "hybrid_similarity": hybrid_score,
        "ngram_similarity": scores['ngram'],
        "jaccard_similarity": scores['jaccard'],
        "semantic_similarity": scores['semantic'],
        "direct_match_percent": direct_match_percent,
        "paraphrase_percent": paraphrase_percent,
        "verdict": verdict,
        "verdict_class": verdict_class,
        "matches": matches,
        "matched_count": len(matches),
    }

    app.config['LAST_RESULT'] = result

    return render_template("index.html", result=result)


@app.route('/download-report')
def download_report():
    last_result = app.config.get('LAST_RESULT')
    if not last_result:
        return "Run analysis first", 400

    pdf_buffer = generate_pdf_report(last_result)

    return send_file(pdf_buffer,
                     download_name=f"Plagiarism_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                     mimetype='application/pdf', as_attachment=True)


# ============================================
# START
# ============================================

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("✅ AI Plagiarism Detector running!")
    print("🌐 Open: http://127.0.0.1:5000")
    app.run(debug=True)