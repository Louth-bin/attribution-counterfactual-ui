"""Build the inline participant-versus-model diagnostic visualization."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "qualtrics" / "v1.0_model_vs_participant_statistics.csv"
OUTPUT = Path(tempfile.gettempdir()) / "participant-model-diagnostics-v10.html"

with INPUT.open(encoding="utf-8-sig", newline="") as source:
    rows = list(csv.DictReader(source))

keep = []
for row in rows:
    item = {"participant": row["participant"], "xai": row["xai"], "family": row["selected model family"]}
    for metric in ("validity", "boundary distance", "plausibility"):
        for source in ("observed", "model", "baseline"):
            item[f"{source} {metric}"] = round(float(row[f"{source} {metric}"]), 6)
    keep.append(item)

data = json.dumps(keep, separators=(",", ":"))
fragment = f'''<div id="participant-model-diagnostics">
  <h2>Participant statistics versus fitted predictions</h2>
  <div class="viz-row legend" aria-label="Legend">
    <button type="button" class="btn btn-ghost source-toggle" data-source="model" aria-pressed="true"><span class="circle-symbol">●</span> Fitted model</button>
    <button type="button" class="btn btn-ghost source-toggle" data-source="baseline" aria-pressed="true"><span class="cross-symbol">×</span> Random-edit baseline</button>
    <span class="condition-label none">None</span><span class="condition-label attribution">Attribution</span><span class="condition-label counterfactual">Counterfactual</span>
  </div>
  <div class="plots"></div>
  <div class="tooltip" role="tooltip"></div>
</div>
<style>
#participant-model-diagnostics {{ width:100%; color:var(--foreground); }}
#participant-model-diagnostics h2 {{ font-weight:500; margin:0 0 8px; white-space:normal; overflow-wrap:anywhere; max-width:100%; }}
#participant-model-diagnostics .legend {{ gap:12px; margin-bottom:8px; }}
#participant-model-diagnostics .source-toggle[aria-pressed="false"] {{ opacity:.4; }}
#participant-model-diagnostics .circle-symbol, #participant-model-diagnostics .cross-symbol {{ color:var(--foreground); }}
#participant-model-diagnostics .condition-label {{ font-size:12px; }}
#participant-model-diagnostics .condition-label.none {{ color:var(--viz-series-1); }}
#participant-model-diagnostics .condition-label.attribution {{ color:var(--viz-series-2); }}
#participant-model-diagnostics .condition-label.counterfactual {{ color:var(--viz-series-3); }}
#participant-model-diagnostics .plots {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
#participant-model-diagnostics .panel {{ min-width:0; }}
#participant-model-diagnostics .panel-title {{ font-weight:500; margin:0 0 3px; }}
#participant-model-diagnostics svg {{ width:100%; display:block; }}
#participant-model-diagnostics .axis path, #participant-model-diagnostics .axis line {{ stroke:var(--border); }}
#participant-model-diagnostics .axis text, #participant-model-diagnostics .axis-title, #participant-model-diagnostics .identity-label {{ fill:var(--foreground); font-size:12px; }}
#participant-model-diagnostics .grid line {{ stroke:var(--border); opacity:.45; }}
#participant-model-diagnostics .identity {{ stroke:var(--muted-foreground); stroke-width:1.5; stroke-dasharray:5 4; }}
#participant-model-diagnostics .tooltip {{ position:absolute; pointer-events:none; opacity:0; background:var(--popover); color:var(--popover-foreground); padding:7px 9px; border-radius:6px; font-size:12px; z-index:3; }}
@media(max-width:700px) {{ #participant-model-diagnostics .plots {{ grid-template-columns:1fr; }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root=document.getElementById('participant-model-diagnostics');
  const data={data};
  const specs=[
    {{key:'validity',title:'Validity rate',format:d3.format('.2f')}},
    {{key:'boundary distance',title:'Distance to boundary',format:d3.format('.3f')}},
    {{key:'plausibility',title:'Plausibility',format:d3.format('.3f')}}
  ];
  const colors={{none:'var(--viz-series-1)',attribution:'var(--viz-series-2)',counterfactual:'var(--viz-series-3)'}};
  const visible={{model:true,baseline:true}};
  const tooltip=root.querySelector('.tooltip');
  const plots=d3.select(root).select('.plots');
  const panels=specs.map(spec=>{{
    const panel=plots.append('section').attr('class','panel');
    panel.append('div').attr('class','panel-title').text(spec.title);
    const svg=panel.append('svg').attr('role','img').attr('aria-label',`${{spec.title}}: observed participant mean on x-axis and predicted mean on y-axis`);
    return {{spec,panel:panel.node(),svg}};
  }});
  function draw(item) {{
    const width=Math.max(300,item.panel.getBoundingClientRect().width), height=330;
    const margin={{top:12,right:34,bottom:52,left:60}};
    item.svg.attr('viewBox',`0 0 ${{width}} ${{height}}`); item.svg.selectAll('*').remove();
    const values=[];
    data.forEach(d=>['model','baseline'].forEach(s=>{{values.push(+d[`observed ${{item.spec.key}}`],+d[`${{s}} ${{item.spec.key}}`]);}}));
    let extent=d3.extent(values); let pad=Math.max((extent[1]-extent[0])*.08,.005); extent=[extent[0]-pad,extent[1]+pad];
    if(item.spec.key==='validity') extent=[Math.max(0,extent[0]),Math.min(1,extent[1])];
    const x=d3.scaleLinear().domain(extent).range([margin.left,width-margin.right]);
    const y=d3.scaleLinear().domain(extent).range([height-margin.bottom,margin.top]);
    const ticks=width<380?4:5;
    item.svg.append('g').attr('class','grid').attr('transform',`translate(0,${{height-margin.bottom}})`).call(d3.axisBottom(x).ticks(ticks).tickSize(-(height-margin.top-margin.bottom)).tickFormat('')).call(g=>g.select('.domain').remove());
    item.svg.append('line').attr('class','identity').attr('x1',x(extent[0])).attr('y1',y(extent[0])).attr('x2',x(extent[1])).attr('y2',y(extent[1]));
    item.svg.append('text').attr('class','identity-label').attr('x',x(extent[1])-4).attr('y',y(extent[1])+13).attr('text-anchor','end').text('perfect agreement');
    item.svg.append('g').attr('class','axis').attr('transform',`translate(0,${{height-margin.bottom}})`).call(d3.axisBottom(x).ticks(ticks).tickFormat(item.spec.format));
    item.svg.append('g').attr('class','axis').attr('transform',`translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(item.spec.format));
    item.svg.append('text').attr('class','axis-title').attr('data-axis','x').attr('x',(margin.left+width-margin.right)/2).attr('y',height-9).attr('text-anchor','middle').text('Observed participant mean');
    item.svg.append('text').attr('class','axis-title').attr('data-axis','y').attr('transform','rotate(-90)').attr('x',-(margin.top+height-margin.bottom)/2).attr('y',16).attr('text-anchor','middle').text('Predicted mean');
    ['baseline','model'].forEach(source=>{{
      const group=item.svg.append('g').attr('class',`series-${{source}}`).style('display',visible[source]?null:'none');
      data.forEach(d=>{{
        const cx=x(+d[`observed ${{item.spec.key}}`]), cy=y(+d[`${{source}} ${{item.spec.key}}`]);
        const mark=source==='model'
          ? group.append('circle').attr('cx',cx).attr('cy',cy).attr('r',4.2).attr('fill',colors[d.xai]).attr('opacity',.78)
          : group.append('path').attr('d',`M${{cx-4}},${{cy-4}}L${{cx+4}},${{cy+4}}M${{cx+4}},${{cy-4}}L${{cx-4}},${{cy+4}}`).attr('fill','none').attr('stroke',colors[d.xai]).attr('stroke-width',1.5).attr('opacity',.62);
        mark.attr('tabindex',0).on('mouseenter focus',(event)=>{{
          tooltip.style.opacity=1; tooltip.innerHTML=`${{d.participant}} · ${{d.xai}}<br>${{source==='model'?d.family:'random-edit baseline'}}<br>Observed: ${{item.spec.format(+d[`observed ${{item.spec.key}}`])}}<br>Predicted: ${{item.spec.format(+d[`${{source}} ${{item.spec.key}}`])}}`;
          const box=root.getBoundingClientRect(); tooltip.style.left=`${{event.clientX-box.left+10}}px`; tooltip.style.top=`${{event.clientY-box.top+10}}px`;
        }}).on('mouseleave blur',()=>tooltip.style.opacity=0);
      }});
    }});
  }}
  panels.forEach(draw);
  root.querySelectorAll('.source-toggle').forEach(button=>button.addEventListener('click',()=>{{
    const source=button.dataset.source; visible[source]=!visible[source]; button.setAttribute('aria-pressed',String(visible[source])); panels.forEach(draw);
  }}));
  new ResizeObserver(()=>panels.forEach(draw)).observe(root);
}})();
</script>'''

OUTPUT.write_text(fragment, encoding="utf-8")
print(OUTPUT)
