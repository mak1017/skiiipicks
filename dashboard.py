"""Inject dashboard_data.json into the HTML template -> one self-contained file.

    python -m skiiipicks.dashboard dashboard_data.json dashboard.html
"""
import json
import pathlib
import sys

TEMPLATE = pathlib.Path(__file__).with_name("dashboard_template.html")


def make(data_path, out_path):
    data = json.loads(pathlib.Path(data_path).read_text())
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    pathlib.Path(out_path).write_text(html)
    print("wrote", out_path)


if __name__ == "__main__":
    make(*(sys.argv[1:3] if len(sys.argv) > 2 else ("dashboard_data.json", "dashboard.html")))
