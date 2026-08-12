# Training question HTML

```html
<div id="cf-study-root" data-phase="training" class="cf-study-root">
  <h2 id="cf-case-title">Training case</h2>
  <p>Study the profile, then predict the AI output.</p>
  <iframe id="cf-case-frame" title="Training profile" style="width:100%;height:420px;border:0"></iframe>
  <div id="cf-training-answer" class="cf-training-answer" hidden></div>
  <p id="cf-status" class="cf-status" aria-live="polite"></p>
</div>
<style>
  .cf-study-root { font-size:18px; max-width:1050px; margin:0 auto; }
  .cf-training-answer { display:flex; gap:12px; justify-content:center; margin:18px 0; }
  .cf-answer-button { border:1px solid #243447; border-radius:6px; background:#fff; padding:11px 22px; cursor:pointer; font-size:17px; }
  .cf-answer-button:hover { background:#eef5fb; }
  .cf-answer-button:disabled { cursor:default; opacity:.7; }
  .cf-status { min-height:1.5em; font-weight:600; }
  .cf-correct { color:#217a45; }
  .cf-incorrect { color:#b33a3a; }
</style>
```

# Label-0 testing question HTML

```html
<div id="cf-study-root" data-phase="test" data-test-label="0" class="cf-study-root">
  <h2 id="cf-case-title">Testing case</h2>
  <!-- These prompts are part of Qualtrics, not the iframe. Edit them here. -->
  <p class="cf-test-prompt" data-domain="housing" hidden>
    This house is predicted as <b>Cheap</b>. Use the Changes column to make the
    smallest changes that make the AI predict <b>Expensive</b>.
  </p>
  <p class="cf-test-prompt" data-domain="safelimit" hidden>
    This driver is predicted as <b>Above Limit</b>. Use the Changes column to
    make the smallest changes that make the AI predict <b>Below Limit</b>.
  </p>
  <iframe id="cf-case-frame" title="Testing profile" style="width:100%;height:650px;border:0"></iframe>
  <p id="cf-status" class="cf-status" aria-live="polite"></p>
</div>
<style>
  .cf-study-root { font-size:18px; max-width:1050px; margin:0 auto; }
  .cf-test-prompt { padding:12px 14px; border-left:4px solid #243447; background:#f5f7f9; }
  .cf-status { min-height:1.5em; font-weight:600; }
</style>
```

Duplicate this question for the label-1 testing block, change:

```html
data-test-label="0"
```

to:

```html
data-test-label="1"
```

and reverse the source and target labels in both editable prompt paragraphs.

Use the same JavaScript from `qualtrics-frame.js` for all three looped questions.
