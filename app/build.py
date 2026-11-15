"""Inline the pipeline output into the web app: app/template.html -> app/index.html"""
from pathlib import Path

root = Path(__file__).resolve().parent
data = (root.parent / "output" / "firepulse_data.json").read_text()
html = (root / "template.html").read_text().replace("/*__DATA__*/", data.replace("</", "<\\/"))
(root / "index.html").write_text(html)
print(f"wrote {root / 'index.html'} ({len(html) / 1e6:.1f} MB)")
