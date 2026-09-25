import pymupdf
import ollama
import json
import re
import random
import os
from tqdm import tqdm
from itertools import combinations

# ==============================================================================
# CONFIGURATION (Tweak these for your research)
# ==============================================================================
PDF_PATH = "my_textbook.pdf"
OUTPUT_FILE = "dataset.json"
OLLAMA_MODEL = "llama3.1:8b"  # or "mistral", "phi3", "qwen2"
SINGLE_CHAPTER_QA = 20   # Number of simple Q&A pairs
CROSS_CHAPTER_QA = 15    # Number of complex synthesis Q&A pairs
CACHE_DIR = ".cache"     # Intermediate results saved here

os.makedirs(CACHE_DIR, exist_ok=True)

# ==============================================================================
# 1. ROBUST PDF EXTRACTION (PyMuPDF)
# ==============================================================================
def extract_text_per_page(pdf_path):
    """Extracts text page-by-page using PyMuPDF. Handles CFF fonts natively."""
    print(f"📖 Extracting text from {pdf_path}...")
    doc = pymupdf.open(pdf_path)  # Changed from fitz.open
    pages = []
    for page_num in tqdm(range(len(doc)), desc="  Extracting pages"):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    print(f"✅ Extracted {len(pages)} non-empty pages.")
    return pages    

def split_into_chapters(pages):
    """Detects chapter boundaries using multiple regex patterns."""
    # Remove (?i) from individual patterns - we'll apply it globally
    patterns = [
        r'^chapter\s+\d+',
        r'^CHAPTER\s+\d+',
        r'^\d+\.\s+[A-Z][a-zA-Z\s]{3,40}$',  # "1. Introduction"
        r'^part\s+[IVX\d]+',
    ]
    combined_pattern = '|'.join(f'(?:{p})' for p in patterns)  # Use non-capturing groups
    
    chapters = {}
    current_chapter = "Preamble"
    current_text = ""
    
    for page_data in pages:
        lines = page_data["text"].split('\n')
        for line in lines:
            line_stripped = line.strip()
            # Apply IGNORECASE flag as a parameter, not inline
            if re.match(combined_pattern, line_stripped, re.IGNORECASE) and len(line_stripped) < 80:
                # Save previous chapter
                if len(current_text) > 200:
                    chapters[current_chapter] = current_text.strip()
                current_chapter = line_stripped
                current_text = ""
            else:
                current_text += line + "\n"
    
    # Don't forget the last chapter
    if len(current_text) > 200:
        chapters[current_chapter] = current_text.strip()
    
    # Fallback: if no chapters detected, split by page count
    if len(chapters) <= 1:
        print("⚠️  No chapter headings detected. Splitting by page blocks...")
        chapters = {}
        chunk_size = max(1, len(pages) // 10)
        for i in range(0, len(pages), chunk_size):
            block = pages[i:i + chunk_size]
            text = "\n".join(p["text"] for p in block)
            chapters[f"Section {i // chunk_size + 1}"] = text
    
    print(f"✅ Identified {len(chapters)} chapters/sections.")
    return chapters

# ==============================================================================
# 2. ROBUST OLLAMA CALLER (With retries)
# ==============================================================================
def call_ollama(prompt, max_retries=3):
    """Calls local Ollama with retry logic and timeout handling."""
    for attempt in range(max_retries):
        try:
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                options={"temperature": 0.4, "num_predict": 512}
            )
            return response['response'].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    ⚠️  Ollama call failed (attempt {attempt+1}), retrying...")
            else:
                print(f"    ❌ Ollama call failed after {max_retries} attempts: {e}")
                return ""

def extract_json_from_text(text):
    """Robustly extracts JSON from LLM output, handling markdown and garbage."""
    # Strip markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Find the first { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            # Try fixing trailing commas
            fixed = re.sub(r',\s*}', '}', match.group())
            fixed = re.sub(r',\s*]', ']', fixed)
            try:
                return json.loads(fixed)
            except:
                pass
    return None

# ==============================================================================
# 3. CACHED SUMMARIZATION
# ==============================================================================
def generate_summaries(chapters):
    """Summarizes each chapter with caching to survive crashes."""
    cache_file = os.path.join(CACHE_DIR, "summaries.json")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            summaries = json.load(f)
        print(f"📋 Loaded {len(summaries)} cached summaries.")
        return summaries
    
    summaries = {}
    print("\n🧠 Generating chapter summaries...")
    for heading, text in tqdm(chapters.items(), desc="  Summarizing"):
        excerpt = text[:2000]
        prompt = f"Summarize the core concepts, rules, and principles of this text in exactly 2-3 sentences. Be specific about terminology.\n\nText:\n{excerpt}"
        summary = call_ollama(prompt)
        if summary:
            summaries[heading] = summary
    
    with open(cache_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"✅ Summaries cached to {cache_file}")
    return summaries

# ==============================================================================
# 4. SINGLE-CHAPTER Q&A (Foundational knowledge)
# ==============================================================================
def generate_single_chapter_qa(chapters, num_examples):
    """Generates straightforward Q&A from individual chapters."""
    training_data = []
    chapter_keys = list(chapters.keys())
    
    print(f"\n📝 Generating {num_examples} single-chapter Q&A pairs...")
    for i in tqdm(range(num_examples), desc="  Single-chapter Q&A"):
        ch = random.choice(chapter_keys)
        excerpt = chapters[ch][random.randint(0, max(0, len(chapters[ch]) - 2000)):][:2000]
        
        prompt = f"""Based ONLY on the following textbook excerpt, generate ONE challenging exam question and its detailed answer.

Excerpt from "{ch}":
{excerpt}

Output strictly as JSON:
{{"instruction": "The question", "input": "", "output": "The detailed answer citing specific concepts from the text"}}"""
        
        result = extract_json_from_text(call_ollama(prompt))
        if result and "instruction" in result and "output" in result:
            result["input"] = ""
            training_data.append(result)
    
    print(f"✅ Generated {len(training_data)} single-chapter Q&A pairs.")
    return training_data

# ==============================================================================
# 5. CROSS-CHAPTER SYNTHESIS Q&A (Complex reasoning)
# ==============================================================================
def generate_cross_chapter_qa(chapters, summaries, num_examples):
    """Generates complex Q&A requiring knowledge from TWO chapters."""
    training_data = []
    chapter_keys = list(chapters.keys())
    
    if len(chapter_keys) < 2:
        print("⚠️  Need at least 2 chapters for cross-chapter synthesis.")
        return training_data
    
    # Generate all unique pairs and shuffle (prevents duplicates)
    all_pairs = list(combinations(chapter_keys, 2))
    random.shuffle(all_pairs)
    selected_pairs = all_pairs[:num_examples]
    
    print(f"\n🔀 Generating {num_examples} cross-chapter synthesis Q&A pairs...")
    for ch1, ch2 in tqdm(selected_pairs, desc="  Cross-chapter Q&A"):
        excerpt1 = chapters[ch1][:1500]
        excerpt2 = chapters[ch2][:1500]
        
        prompt = f"""You are an expert exam writer. Create a complex scenario-based question that CANNOT be answered using only one chapter. The student MUST combine principles from both chapters.

Chapter A ("{ch1}") Summary: {summaries.get(ch1, 'N/A')}
Chapter A Excerpt: {excerpt1}

Chapter B ("{ch2}") Summary: {summaries.get(ch2, 'N/A')}
Chapter B Excerpt: {excerpt2}

Output strictly as JSON:
{{"instruction": "A novel scenario requiring both chapters' knowledge", "input": "", "output": "Step 1: From [Chapter A concept]... Step 2: Combining with [Chapter B concept]... Conclusion:..."}}"""
        
        result = extract_json_from_text(call_ollama(prompt))
        if result and "instruction" in result and "output" in result:
            result["input"] = ""
            training_data.append(result)
    
    print(f"✅ Generated {len(training_data)} cross-chapter Q&A pairs.")
    return training_data

# ==============================================================================
# 6. MAIN PIPELINE
# ==============================================================================
def build_dataset():
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found: {PDF_PATH}")
        print(f"   Place your textbook PDF in this directory and update PDF_PATH.")
        return
    
    # Step 1: Extract
    pages = extract_text_per_page(PDF_PATH)
    chapters = split_into_chapters(pages)
    
    # Step 2: Summarize (cached)
    summaries = generate_summaries(chapters)
    
    # Step 3: Generate Q&A
    single_qa = generate_single_chapter_qa(chapters, SINGLE_CHAPTER_QA)
    cross_qa = generate_cross_chapter_qa(chapters, summaries, CROSS_CHAPTER_QA)
    
    # Step 4: Combine and save
    full_dataset = single_qa + cross_qa
    random.shuffle(full_dataset)  # Mix simple and complex
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(full_dataset, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"🎉 DATASET COMPLETE")
    print(f"   Total Q&A pairs: {len(full_dataset)}")
    print(f"   Single-chapter:  {len(single_qa)}")
    print(f"   Cross-chapter:   {len(cross_qa)}")
    print(f"   Saved to:        {OUTPUT_FILE}")
    print(f"{'='*50}")

if __name__ == "__main__":
    build_dataset()