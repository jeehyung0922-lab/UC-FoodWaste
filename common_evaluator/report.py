"""Dependency-free SVG output. Curve axes show the low-FPR operating range."""
from html import escape


def document(path, title, elements, height=410):
    with path.open("x",encoding="utf-8") as stream:
        stream.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 {height}" font-family="sans-serif">\n'
                     f'<rect width="860" height="{height}" fill="white"/>\n'
                     f'<text x="38" y="35" font-size="20" font-weight="bold">{escape(title)}</text>\n'
                     + "\n".join(elements) + "\n</svg>\n")


def bars(path,title,metrics):
    elements = []
    for i,(label,value) in enumerate(metrics):
        y=76+38*i
        elements.append(f'<text x="38" y="{y}" font-size="14">{escape(label)}</text>')
        elements.append(f'<rect x="345" y="{y-15}" width="420" height="17" fill="#e9eff4"/>')
        if value is not None:
            elements.append(f'<rect x="345" y="{y-15}" width="{420*value:.2f}" height="17" fill="#286381"/>')
        text="N/A" if value is None else f"{value:.2%}"
        elements.append(f'<text x="778" y="{y}" font-size="14">{text}</text>')
    document(path,title,elements,110+38*len(metrics))


def curve(path,title,name,points,selected):
    def x(a): return 85+a/.05*700
    def y(b): return 320-b*240
    elements=['<path d="M85 80 V320 H785" fill="none" stroke="#243844"/>',
              '<text x="415" y="380" font-size="15">Pixel FPR</text>',
              f'<text x="85" y="62" font-size="15">{escape(name.replace("_"," ").upper())}</text>']
    for value in (0,.01,.02,.03,.04,.05):
        elements.append(f'<text x="{x(value)-7:.2f}" y="345" font-size="13">{value:.0%}</text>')
    for value in (0,.25,.5,.75,1):
        elements.append(f'<text x="36" y="{y(value)+4:.2f}" font-size="13">{value:.0%}</text>')
    points=list(reversed(points))
    coords=[(a,b) for a,b in points if a is not None and b is not None and a<=.05]
    line=" ".join(f"{x(a):.2f},{y(b):.2f}" for a,b in coords)
    elements.append(f'<polyline points="{line}" fill="none" stroke="#286381" stroke-width="2"/>')
    elements.append(f'<line x1="{x(.02)}" y1="80" x2="{x(.02)}" y2="320" stroke="#b55542" stroke-dasharray="6 4"/>')
    elements.append(f'<circle cx="{x(selected[0]):.2f}" cy="{y(selected[1]):.2f}" r="5" fill="#327962"/>')
    document(path,title,elements)
