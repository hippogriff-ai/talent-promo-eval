from pathlib import Path

from tpe.job_fetch import html_to_text, job_text

HTML = """<html><head><style>p{color:red}</style><script>var x=1;</script></head>
<body><h1>Staff Engineer</h1><p>Build   agent infrastructure.</p></body></html>"""


def test_html_to_text_strips_script_style_tags():
    text = html_to_text(HTML)
    assert "Staff Engineer" in text
    assert "Build agent infrastructure." in text
    assert "var x" not in text
    assert "color:red" not in text
    assert "<p>" not in text


def test_job_text_passthrough_for_raw_text():
    assert job_text("Just a plain JD body") == "Just a plain JD body"


def test_job_text_reads_file(tmp_path: Path):
    f = tmp_path / "jd.txt"
    f.write_text("JD from file")
    assert job_text(str(f)) == "JD from file"
