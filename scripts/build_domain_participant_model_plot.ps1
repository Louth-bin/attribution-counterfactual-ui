param(
    [string]$InputCsv = "qualtrics\v1.4_all_old_domains_model_vs_participant_statistics.csv",
    [string]$OutputHtml = "$env:TEMP\participant-model-by-domain.html"
)

$rows = Import-Csv -LiteralPath $InputCsv | ForEach-Object {
    [ordered]@{
        participant = $_.participant
        domain = $_.domain
        xai = $_.xai
        observedValidity = [double]$_.'observed validity'
        modelValidity = [double]$_.'model validity'
        observedBoundary = [double]$_.'observed boundary distance'
        modelBoundary = [double]$_.'model boundary distance'
        observedPlausibility = [double]$_.'observed plausibility'
        modelPlausibility = [double]$_.'model plausibility'
    }
}
$json = $rows | ConvertTo-Json -Compress

$fragment = @'
<div id="participant-model-by-domain">
  <h2>Observed participants vs simulated models</h2>
  <div class="viz-row legend" aria-label="Explanation condition legend"></div>
  <div class="domains"></div>
  <div class="tooltip" role="tooltip" hidden></div>
</div>

<style>
#participant-model-by-domain { width: 100%; color: var(--foreground); }
#participant-model-by-domain h2 { margin: 0 0 8px; font-weight: 500; }
#participant-model-by-domain h3 { margin: 20px 0 6px; font-weight: 500; }
#participant-model-by-domain h4 { margin: 0; font-weight: 500; }
#participant-model-by-domain .legend { margin-bottom: 4px; gap: 14px; }
#participant-model-by-domain .legend button { background: transparent; border: 0; color: var(--foreground); padding: 5px 2px; }
#participant-model-by-domain .legend button[aria-pressed="false"] { opacity: .38; }
#participant-model-by-domain .swatch { display: inline-block; width: 10px; height: 10px; margin-right: 6px; background: var(--swatch); }
#participant-model-by-domain .plots { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; }
#participant-model-by-domain .panel { min-width: 0; }
#participant-model-by-domain .panel-meta { color: var(--muted-foreground); margin-top: 2px; }
#participant-model-by-domain .chart { width: 100%; min-height: 250px; }
#participant-model-by-domain svg { display: block; width: 100%; height: auto; overflow: visible; }
#participant-model-by-domain .axis path,
#participant-model-by-domain .axis line { stroke: var(--border); }
#participant-model-by-domain .axis text,
#participant-model-by-domain svg text { fill: var(--foreground); font-size: 12px; }
#participant-model-by-domain .grid line { stroke: var(--border); stroke-opacity: .45; }
#participant-model-by-domain .identity { stroke: var(--muted-foreground); stroke-width: 1.5; stroke-dasharray: 5 4; }
#participant-model-by-domain rect[data-chart-frame] { fill: none; stroke: var(--border); }
#participant-model-by-domain .point { stroke: var(--background); stroke-width: 1; }
#participant-model-by-domain .axis-title { font-size: 12px; fill: var(--foreground); }
#participant-model-by-domain .tooltip { position: absolute; pointer-events: none; z-index: 20; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 7px 9px; max-width: 240px; }
@media (max-width: 700px) {
  #participant-model-by-domain .plots { grid-template-columns: 1fr; gap: 20px; }
  #participant-model-by-domain .chart { min-height: 270px; }
}
</style>

<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
  const root = document.getElementById('participant-model-by-domain');
  const data = __DATA__;
  const domains = [
    { key: 'diabetes', label: 'Diabetes' },
    { key: 'housing', label: 'Housing' },
    { key: 'safelimit', label: 'SafeLimit' }
  ];
  const metrics = [
    { key: 'Validity', label: 'Validity', observed: 'observedValidity', model: 'modelValidity', format: d3.format('.2f') },
    { key: 'Boundary', label: 'Boundary distance', observed: 'observedBoundary', model: 'modelBoundary', format: d3.format('.3f') },
    { key: 'Plausibility', label: 'Plausibility', observed: 'observedPlausibility', model: 'modelPlausibility', format: d3.format('.3f') }
  ];
  const conditions = [
    { key: 'none', label: 'None', color: 'var(--viz-series-1)', symbol: d3.symbolCircle },
    { key: 'attribution', label: 'Attribution', color: 'var(--viz-series-2)', symbol: d3.symbolSquare },
    { key: 'counterfactual', label: 'Counterfactual', color: 'var(--viz-series-3)', symbol: d3.symbolTriangle }
  ];
  const active = new Set(conditions.map(d => d.key));
  const tooltip = root.querySelector('.tooltip');

  const legend = d3.select(root.querySelector('.legend'));
  legend.selectAll('button').data(conditions).join('button')
    .attr('type', 'button')
    .attr('aria-pressed', 'true')
    .html(d => `<span class="swatch" style="--swatch:${d.color}"></span>${d.label}`)
    .on('click', function(event, d) {
      if (active.has(d.key)) active.delete(d.key); else active.add(d.key);
      d3.select(this).attr('aria-pressed', active.has(d.key) ? 'true' : 'false');
      drawAll();
    });

  const domainNodes = d3.select(root.querySelector('.domains')).selectAll('section')
    .data(domains).join('section').attr('class', 'domain');
  domainNodes.append('h3').text(d => `${d.label} | n=${data.filter(row => row.domain === d.key).length}`);
  const plotRows = domainNodes.append('div').attr('class', 'plots');
  plotRows.each(function(domain) {
    const panels = d3.select(this).selectAll('.panel').data(metrics).join('div').attr('class', 'panel');
    panels.append('h4').text(metric => metric.label);
    panels.append('div').attr('class', 'panel-meta text-small tabular-nums');
    panels.append('div').attr('class', 'chart');
  });

  function correlation(values, xKey, yKey) {
    if (values.length < 2) return NaN;
    const mx = d3.mean(values, d => d[xKey]);
    const my = d3.mean(values, d => d[yKey]);
    const numerator = d3.sum(values, d => (d[xKey] - mx) * (d[yKey] - my));
    const dx = Math.sqrt(d3.sum(values, d => (d[xKey] - mx) ** 2));
    const dy = Math.sqrt(d3.sum(values, d => (d[yKey] - my) ** 2));
    return numerator / (dx * dy);
  }

  function drawPanel(container, domain, metric) {
    const visible = data.filter(row => row.domain === domain.key && active.has(row.xai));
    const allDomain = data.filter(row => row.domain === domain.key);
    const r = correlation(allDomain, metric.observed, metric.model);
    const mae = d3.mean(allDomain, d => Math.abs(d[metric.observed] - d[metric.model]));
    d3.select(container.parentNode).select('.panel-meta').text(`r=${d3.format('.2f')(r)} | MAE=${metric.format(mae)}`);
    const host = d3.select(container);
    host.selectAll('*').remove();
    const width = Math.max(280, container.getBoundingClientRect().width || 300);
    const height = width < 380 ? 280 : 260;
    const margin = { top: 12, right: 14, bottom: 50, left: 62 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const extent = d3.extent(allDomain.flatMap(d => [d[metric.observed], d[metric.model]]));
    const span = Math.max(extent[1] - extent[0], metric.key === 'Validity' ? .1 : .01);
    let lo = extent[0] - span * .08;
    let hi = extent[1] + span * .08;
    if (metric.key === 'Validity' || metric.key === 'Plausibility') {
      lo = Math.max(0, lo); hi = Math.min(1, hi);
    }
    const x = d3.scaleLinear().domain([lo, hi]).nice().range([0, innerW]);
    const y = d3.scaleLinear().domain(x.domain()).range([innerH, 0]);
    const svg = host.append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('role', 'img')
      .attr('aria-label', `${domain.label} ${metric.label}: observed participant values versus fitted-model simulation`);
    svg.append('title').text(`${domain.label} ${metric.label}`);
    svg.append('desc').text('Points near the diagonal identity line indicate close participant-level replication.');
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    const tickCount = width < 340 ? 4 : 5;
    g.append('g').attr('class', 'grid').call(d3.axisLeft(y).ticks(tickCount).tickSize(-innerW).tickFormat('')).call(s => s.select('.domain').remove());
    g.append('rect').attr('data-chart-frame', '').attr('width', innerW).attr('height', innerH);
    g.append('line').attr('class', 'identity').attr('x1', x(x.domain()[0])).attr('y1', y(x.domain()[0])).attr('x2', x(x.domain()[1])).attr('y2', y(x.domain()[1]));
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(tickCount).tickFormat(metric.format));
    g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(tickCount).tickFormat(metric.format));
    svg.append('text').attr('class', 'axis-title').attr('data-axis', 'x').attr('x', margin.left + innerW / 2).attr('y', height - 7).attr('text-anchor', 'middle').text('Observed participant');
    svg.append('text').attr('class', 'axis-title').attr('data-axis', 'y').attr('transform', `translate(15,${margin.top + innerH / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('Simulated model');

    const conditionMap = new Map(conditions.map(d => [d.key, d]));
    const points = g.selectAll('.point').data(visible, d => d.participant).join('path')
      .attr('class', 'point')
      .attr('d', d => d3.symbol().type(conditionMap.get(d.xai).symbol).size(65)())
      .attr('transform', d => `translate(${x(d[metric.observed])},${y(d[metric.model])})`)
      .attr('fill', d => conditionMap.get(d.xai).color);

    g.append('rect')
      .attr('data-chart-hit', '')
      .attr('x', 0).attr('y', 0).attr('width', innerW).attr('height', innerH)
      .attr('fill', 'transparent')
      .on('pointermove', function(event) {
        if (!visible.length) return;
        const [px, py] = d3.pointer(event, this);
        const nearest = d3.least(visible, d => Math.hypot(x(d[metric.observed]) - px, y(d[metric.model]) - py));
        const bounds = root.getBoundingClientRect();
        tooltip.hidden = false;
        tooltip.innerHTML = `<strong>${nearest.participant}</strong><br>${conditionMap.get(nearest.xai).label}<br>Observed: ${metric.format(nearest[metric.observed])}<br>Model: ${metric.format(nearest[metric.model])}`;
        tooltip.style.left = `${event.clientX - bounds.left + 12}px`;
        tooltip.style.top = `${event.clientY - bounds.top + 12}px`;
      })
      .on('pointerleave', () => { tooltip.hidden = true; });
  }

  function drawAll() {
    domainNodes.each(function(domain) {
      d3.select(this).selectAll('.panel').each(function(metric) {
        drawPanel(this.querySelector('.chart'), domain, metric);
      });
    });
  }

  drawAll();
  const observer = new ResizeObserver(() => drawAll());
  observer.observe(root);
})();
</script>
'@

$fragment = $fragment.Replace('__DATA__', $json)
[System.IO.File]::WriteAllText(
    [System.IO.Path]::GetFullPath($OutputHtml),
    $fragment,
    [System.Text.UTF8Encoding]::new($false)
)
Write-Output $OutputHtml
