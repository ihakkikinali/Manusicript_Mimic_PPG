"""Fetch selected patient folders from PhysioNet using download_urls.txt.

Requires PhysioNet credentialed access. Set credentials via environment:
    PHYSIONET_USER=... PHYSIONET_PASS=... python src/download_patients.py --dest ./data
"""
import os, argparse, base64, time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote
import requests
from bs4 import BeautifulSoup

BASE = "https://physionet.org/files/mimic-iii-ext-ppg/1.1.0/"


def session():
    s = requests.Session()
    tok = base64.b64encode(f"{os.environ['PHYSIONET_USER']}:{os.environ['PHYSIONET_PASS']}".encode()).decode()
    s.headers.update({"Authorization": f"Basic {tok}", "User-Agent": "Wget/1.21"})
    return s


def files_in(s, url):
    r = s.get(url, timeout=60); r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for a in soup.find_all("a"):
        h = a.get("href", "")
        if not h or h.startswith("..") or h.endswith("/"):
            continue
        full = urljoin(url, h).split("?")[0]
        if full.startswith(BASE) and not Path(full).name.startswith("index"):
            out.append(full)
    return out


def main(urls_file, dest):
    s = session(); dest = Path(dest)
    urls = [u.strip() for u in Path(urls_file).read_text().splitlines() if u.startswith("http")]
    for j, u in enumerate(urls, 1):
        u = u.rstrip("/") + "/"
        rel = unquote(urlparse(u).path).split("1.1.0/")[1]
        (dest / rel).mkdir(parents=True, exist_ok=True)
        for f in files_in(s, u):
            name = Path(unquote(urlparse(f).path)).name
            p = dest / rel / name
            if p.exists():
                continue
            r = s.get(f, stream=True, timeout=180); r.raise_for_status()
            with open(p, "wb") as fh:
                for c in r.iter_content(1 << 20):
                    fh.write(c)
            time.sleep(0.1)
        print(f"[{j}/{len(urls)}] {rel}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default="outputs/download_urls.txt")
    ap.add_argument("--dest", required=True)
    a = ap.parse_args(); main(a.urls, a.dest)
