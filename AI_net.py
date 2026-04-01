# Author : P.P. Chanchal
# AI_net_enhanced.py
# Enhanced version of your script:
# - Multithreaded extraction (I/O parallel)
# - Lightweight webpage summarization
# - Auto-detection of audio (audio tags, audio links, YouTube)
# - Reduced memory usage (lazy model load + streaming downloads)
# - Failure recovery (retries + per-step try/except)
# - Save all extracted text to timestamped .txt for offline use
#
# Keep the original behavior otherwise.

# Suppress tensorflow, keras, pyarrow, and torchvision to avoid DLL/dependency errors
import sys
sys.modules['tensorflow'] = None
sys.modules['keras'] = None
sys.modules['tensorflow'] = None
sys.modules['keras'] = None

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import tempfile
import shutil
import requests
import asyncio
import yt_dlp
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from PyPDF2 import PdfReader
import pandas as pd
from bs4 import BeautifulSoup
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin
import re
from datetime import datetime
import heapq

# Small stopword set to keep summarizer lightweight
_STOPWORDS = {
    "the","a","an","and","or","is","it","of","to","in","that","this","on","for","with","as","are",
    "was","were","be","by","at","from","I","you","he","she","they","we","but","not","have","has",
    "had","which","will","can","would","could","should","about"
}

def now_ts():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

class AutonomousInternetQuerier:
    def __init__(self, device='cpu', max_workers=3, offline_dir="offline_data"):
        self.device = device
        self._whisper_processor = None
        self._whisper_model = None
        self.max_workers = max_workers
        self.offline_dir = offline_dir
        os.makedirs(self.offline_dir, exist_ok=True)

    # Lazy-load whisper to reduce memory when not transcribing
    def _ensure_whisper_loaded(self):
        if self._whisper_processor is None or self._whisper_model is None:
            print("Loading Whisper model (lazy load) ...")
            self._whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self._whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(self.device)

    def web_search(self, query):
        """Robust web search using googlesearch library with DuckDuckGo fallback"""
        print(f"Searching web for: {query}")
        results = []
        # Method 1: Try googlesearch library
        try:
            import googlesearch
            for url in googlesearch.search(query, num_results=3):
                results.append({'title': url, 'link': url})
        except Exception as e:
            # not fatal - fallback next
            # print(f"Google library search failed: {e}")
            pass

        # Method 2: Fallback to DuckDuckGo HTML scraping if Google yielded no results
        if not results:
            try:
                url = "https://html.duckduckgo.com/html/"
                data = {'q': query}
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                resp = requests.post(url, data=data, headers=headers, timeout=15)
                soup = BeautifulSoup(resp.text, 'html.parser')

                for res in soup.find_all('div', class_='result'):
                    a = res.find('a', class_='result__a')
                    snippet = res.find('a', class_='result__snippet')
                    if a:
                        link = a['href']
                        results.append({
                            'title': a.get_text(strip=True),
                            'link': link,
                            'snippet': snippet.get_text(strip=True) if snippet else ''
                        })
                    if len(results) >= 3:
                        break
            except Exception as e:
                print(f"DuckDuckGo search failed: {e}")

        # Sanitize links (basic)
        cleaned = []
        for r in results:
            link = r.get('link') or ''
            if not link:
                continue
            # if relative or redirect-like, ignore or try to fix
            if link.startswith('/l/?'):
                # DDG redirect pattern (skip)
                continue
            cleaned.append({'title': r.get('title', ''), 'link': link, 'snippet': r.get('snippet', '')})
            if len(cleaned) >= 3:
                break

        return cleaned

    # --- Lightweight summarizer (frequency-based) ---
    def summarize_text(self, text, num_sentences=3):
        """Return a short extractive summary using sentence scoring by word frequency."""
        if not text or len(text.split()) < 30:
            return text  # too short to summarize
        # Split into sentences (simple split on punctuation)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Build word frequency
        freq = {}
        for sent in sentences:
            for word in re.findall(r'\w+', sent.lower()):
                if word in _STOPWORDS: continue
                if len(word) <= 1: continue
                freq[word] = freq.get(word, 0) + 1
        if not freq:
            return " ".join(sentences[:num_sentences])
        # Normalize frequencies
        maxf = max(freq.values())
        for k in freq:
            freq[k] = freq[k] / maxf
        # Score sentences
        scores = []
        for i, sent in enumerate(sentences):
            s = 0.0
            for w in re.findall(r'\w+', sent.lower()):
                s += freq.get(w, 0.0)
            # penalize extremely long sentences slightly
            s = s / (len(sent.split())**0.1 + 1e-6)
            scores.append((s, i, sent))
        top = heapq.nlargest(num_sentences, scores, key=lambda x: x[0])
        top_sorted = sorted(top, key=lambda x: x[1])
        summary = " ".join([t[2].strip() for t in top_sorted])
        return summary

    # --- Small helper: robust requests with retries, streaming ---
    def _robust_get(self, url, stream=False, timeout=15, max_retries=3):
        last_exc = None
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        for attempt in range(max_retries):
            try:
                r = requests.get(url, stream=stream, timeout=timeout, headers=headers)
                r.raise_for_status()
                return r
            except Exception as e:
                last_exc = e
                time.sleep(1 + attempt * 2)
        raise last_exc

    # --- Transcription (uses whisper lazily) ---
    def _transcribe_audio(self, audio_path):
        try:
            self._ensure_whisper_loaded()
            audio_input, sr = librosa.load(audio_path, sr=16000)
            inputs = self._whisper_processor(audio_input, sampling_rate=sr, return_tensors="pt").to(self.device)
            with torch.no_grad():
                predicted_ids = self._whisper_model.generate(inputs.input_features, attention_mask=inputs.attention_mask, language="en")
            transcription = self._whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
        except Exception as e:
            return f"Error in transcription: {e}"

    async def extract_text_from_youtube_audio(self, youtube_url):
        # kept async interface - will run downloader in thread
        try:
            tempdir = tempfile.mkdtemp()
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(tempdir, 'audio.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            def download():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
            await asyncio.to_thread(download)
            files = os.listdir(tempdir)
            if not files:
                shutil.rmtree(tempdir)
                return "Error: No audio downloaded."
            audio_path = os.path.join(tempdir, files[0])
            transcription = await asyncio.to_thread(self._transcribe_audio, audio_path)
            # cleanup
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception:
                pass
            shutil.rmtree(tempdir)
            return transcription
        except Exception as e:
            return f"Error processing YouTube audio: {e}"

    # --- File download & extractors with streaming and retries ---
    def _download_and_extract_pdf_text(self, file_url):
        tmpfile = None
        try:
            r = self._robust_get(file_url, stream=True)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmpfile = tmp.name
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
            text_content = ''
            try:
                with open(tmpfile, 'rb') as f:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        text_content += page.extract_text() or ''
            except Exception:
                text_content = ''
            return text_content
        except Exception as e:
            return ""
        finally:
            if tmpfile and os.path.exists(tmpfile):
                try:
                    os.unlink(tmpfile)
                except Exception:
                    pass

    def _download_and_extract_csv(self, file_url):
        tmpfile = None
        try:
            r = self._robust_get(file_url, stream=True)
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmpfile = tmp.name
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
            try:
                df = pd.read_csv(tmpfile)
                text = df.to_csv(index=False)
            except Exception:
                text = ''
            return text
        except Exception:
            return ''
        finally:
            if tmpfile and os.path.exists(tmpfile):
                try:
                    os.unlink(tmpfile)
                except Exception:
                    pass

    def _download_and_extract_excel(self, file_url):
        tmpfile = None
        try:
            r = self._robust_get(file_url, stream=True)
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                tmpfile = tmp.name
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
            try:
                df = pd.read_excel(tmpfile)
                text = df.to_csv(index=False)
            except Exception:
                text = ''
            return text
        except Exception:
            return ''
        finally:
            if tmpfile and os.path.exists(tmpfile):
                try:
                    os.unlink(tmpfile)
                except Exception:
                    pass

    # --- Helper: detects audio on an arbitrary html page (audio tags, audio file links, og:audio, youtube iframes) ---
    def _detect_audio_on_page(self, base_url, soup):
        found = []
        # <audio> tags
        for a in soup.find_all('audio'):
            src = a.get('src')
            if src:
                found.append(urljoin(base_url, src))
            for s in a.find_all('source'):
                if s.get('src'):
                    found.append(urljoin(base_url, s.get('src')))
        # <a> links with common audio extensions
        for a in soup.find_all('a', href=True):
            href = a['href']
            if re.search(r'\.(mp3|wav|m4a|ogg|flac)(?:$|\?)', href, re.I):
                found.append(urljoin(base_url, href))
        # meta tags like og:audio
        for meta in soup.find_all('meta', attrs={'property':'og:audio'}) + soup.find_all('meta', attrs={'name':'og:audio'}):
            if meta.get('content'):
                found.append(urljoin(base_url, meta.get('content')))
        # iframes -> check for youtube links
        for iframe in soup.find_all('iframe', src=True):
            src = iframe['src']
            if 'youtube' in src or 'youtu.be' in src:
                # canonicalize to youtube watch url if possible
                m = re.search(r'(youtube\.com/embed/|youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{6,})', src)
                if m:
                    vid = m.group(2)
                    found.append(f"https://www.youtube.com/watch?v={vid}")
        # dedupe and return
        unique = []
        for f in found:
            if f not in unique:
                unique.append(f)
        return unique

    # --- Process a single search result (download or scrape) ---
    def _process_result(self, res):
        link = res.get('link', '')
        title = res.get('title', link)
        try:
            # Basic file-type checks
            if link.lower().endswith('.pdf'):
                text = self._download_and_extract_pdf_text(link)
                return {'link': link, 'title': title, 'text': text, 'summary': self.summarize_text(text, 2)}
            elif link.lower().endswith('.csv'):
                text = self._download_and_extract_csv(link)
                return {'link': link, 'title': title, 'text': text, 'summary': self.summarize_text(text, 2)}
            elif link.lower().endswith(('.xlsx', '.xls')):
                text = self._download_and_extract_excel(link)
                return {'link': link, 'title': title, 'text': text, 'summary': self.summarize_text(text, 2)}
            else:
                # Scrape page
                try:
                    r = self._robust_get(link, stream=False)
                    soup = BeautifulSoup(r.content, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    if len(text) > 12000:
                        text = text[:12000]  # limit
                except Exception as e:
                    text = ''
                    soup = BeautifulSoup('', 'html.parser')
                # Auto-detect audio and, if found, prioritize transcribing first audio source (YouTube or audio links)
                audio_urls = self._detect_audio_on_page(link, soup)
                audio_transcript = ''
                for au in audio_urls:
                    try:
                        if 'youtube.com' in au or 'youtu.be' in au:
                            # Transcribe YouTube audio (sync via asyncio.run for this thread)
                            audio_transcript = asyncio.run(self.extract_text_from_youtube_audio(au))
                        else:
                            # Direct audio file: download to temp and transcribe
                            tmpfile = None
                            try:
                                r2 = self._robust_get(au, stream=True)
                                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(au)[1] or '.mp3', delete=False) as tmp:
                                    tmpfile = tmp.name
                                    for chunk in r2.iter_content(chunk_size=8192):
                                        if chunk:
                                            tmp.write(chunk)
                                audio_transcript = self._transcribe_audio(tmpfile)
                            finally:
                                if tmpfile and os.path.exists(tmpfile):
                                    try:
                                        os.unlink(tmpfile)
                                    except Exception:
                                        pass
                        if audio_transcript:
                            break
                    except Exception:
                        continue
                # Combine text + audio transcription if any
                if audio_transcript:
                    combined = f"[Audio transcript from page]\n{audio_transcript}\n\n[Page text]\n{text}"
                else:
                    combined = text
                summary = self.summarize_text(combined, 2)
                return {'link': link, 'title': title, 'text': combined, 'summary': summary}
        except Exception as e:
            return {'link': link, 'title': title, 'text': '', 'summary': '', 'error': str(e)}

    async def autonomous_query(self, query=None, save_files=True):
        if not query:
            print("AutonomousInternetQuerier: No query provided. Standing by.")
            return ""

        print(f"Querying web: {query}")
        results = await asyncio.to_thread(self.web_search, query)
        print(f"Found {len(results)} results.")

        # Process top results concurrently (I/O bound) but keep model/transcription tasks handled safely
        extracted_texts = []
        per_result_meta = []
        # Use a small thread pool for I/O bound scraping
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._process_result, res): res for res in results[:self.max_workers]}
            for future in as_completed(futures):
                res = futures[future]
                try:
                    out = future.result()
                    per_result_meta.append(out)
                    if out.get('text'):
                        extracted_texts.append(out.get('text', ''))
                        print(f"  -> Extracted {len(out.get('text',''))} chars from {res.get('link')}")
                    else:
                        print(f"  -> No text extracted from {res.get('link')}")
                except Exception as e:
                    print(f"  -> Error processing {res.get('link')}: {e}")

        combined_text = " ".join(extracted_texts)
        if not combined_text:
            print("Warning: No text content could be extracted from the search results.")

        # Save combined and per-result files for offline use
        if save_files:
            ts = now_ts()
            base_name = re.sub(r'\W+', '_', query)[:60] or "query"
            combined_path = os.path.join(self.offline_dir, f"{base_name}_{ts}_combined.txt")
            try:
                with open(combined_path, "w", encoding="utf-8") as f:
                    f.write(f"Query: {query}\nTimestamp(UTC): {ts}\n\n")
                    for meta in per_result_meta:
                        f.write("="*80 + "\n")
                        f.write(f"Title: {meta.get('title')}\nLink: {meta.get('link')}\n\n")
                        f.write("Summary:\n")
                        f.write(meta.get('summary','') + "\n\n")
                        f.write("Full Extracted Text:\n")
                        f.write((meta.get('text') or '') + "\n\n")
                print(f"Saved combined extracted text to: {combined_path}")
            except Exception as e:
                print(f"Error saving combined file: {e}")

            # Also save per-result short files
            for meta in per_result_meta:
                try:
                    safe_title = re.sub(r'\W+', '_', (meta.get('title') or meta.get('link')))[:60] or "result"
                    pf = os.path.join(self.offline_dir, f"{safe_title}_{ts}.txt")
                    with open(pf, "w", encoding="utf-8") as f:
                        f.write(f"Title: {meta.get('title')}\nLink: {meta.get('link')}\n\n")
                        f.write("Summary:\n")
                        f.write(meta.get('summary','') + "\n\n")
                        f.write("Full Extracted Text:\n")
                        f.write((meta.get('text') or '') + "\n\n")
                except Exception:
                    pass

        return combined_text

# Example usage (keeps same CLI behavior)
async def main():
    ai_query = AutonomousInternetQuerier()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        text_data = await ai_query.autonomous_query(query)
        print("\n=== Extracted Text Snippet ===\n", text_data[:1000])
    else:
        print("AI_net_enhanced.py: Ready for autonomous query. Pass a query string as an argument.")

if __name__ == "__main__":
    asyncio.run(main())
