# Qualtrics setup: Housing, Drink Driving, and Diabetes Warning

Start with `Recourse_v10.qsf`. Qualtrics owns the looping and response data, while `iframe.html` is the main case UI.

## Study design encoded in the QSF

- Set the Embedded Data field `appId` to `housing`, `safelimit`, or `diabetes`. The v10 file retains the v09 default (`safelimit`). The simplest workflow is to duplicate the imported project and use one domain in each copy.
- The default explanation condition is `attribution`.
- Training contains exactly 10 fixed cases; their presentation order is randomized for each participant, and no case is selected from a larger pool.
- Testing contains exactly 20 different fixed cases: ten label-0 cases and ten label-1 cases.
- Each testing direction has ten cases from one predicted label.
- In training, every possible pair of the five most important SHAP features occurs exactly once.
- Training answers are followed by the correct AI output and the assigned explanation.
- Testing has two ten-case direction blocks. Qualtrics shows both, randomizes which block comes first, and randomizes the cases within each block.
- Testing requires at least one counterfactual edit before Next becomes available.

The auditable IDs and labels are in `case-manifest.json`.

## Deployment workflow

1. Publish this repository's web assets over HTTPS. The default URL is `https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html`. The deployment must include `iframe.html`, `iframe.js`, `index.css`, `static-model.js`, `static/experiment-data.js`, and any libraries referenced by `iframe.html`.
2. In Qualtrics, create a project and choose **Import a QSF**, then import `Recourse_v10.qsf`.
3. Open **Survey Flow**. Confirm that the first Embedded Data element contains:
   - `ui_base_url`
   - `xaiType`
   - `training_log_json`
   - `testing_log_json`
4. Change `ui_base_url` from `https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html` if the UI is hosted elsewhere.
5. A second Embedded Data element sets `appId`. Use `housing`, `safelimit`, or `diabetes`.
6. Open the training and testing blocks and verify their Loop & Merge rows are `0` through `9`, with Loop & Merge randomization set to **All** for all three blocks.
7. In Survey Flow, verify that the two testing blocks are nested under one Randomizer configured to present **2 of 2** elements. This randomizes their order while showing both directions.
8. Preview each domain after changing `appId`.
9. Run one complete response per domain before publishing. Export the response and verify that `training_log_json` contains ten records and `testing_log_json` contains twenty records.
10. Verify the screening failure path: attempt 2 repeats the Basic Interface and the condition-appropriate AI Explanation directly above the repeated questions.

## Training answers and saved data

The two training answer buttons appear directly below the embedded profile. They are created by `qualtrics-frame.js`, so the participant selects the answer on the Qualtrics page rather than inside the iframe.

Each click immediately updates `training_log_json`. The record contains the domain, case number, instance ID, explanation condition, selected prediction, correct prediction, whether the answer was correct, and response time. Clicking Next saves the same record again as a final safeguard. Records are keyed by instance ID, so revising or revisiting a case updates that case without duplicating it.

Testing edits are saved to `testing_log_json`. Its records include the tested direction, instance ID, original prediction, edited values, feedback, and response time. The two testing blocks can both use Loop & Merge rows `0` through `9` because records are keyed by instance ID rather than loop number.

## Editing the application scenario

The **Domain and Interface Tutorials** block contains an **Application scenario** Descriptive Text question. Its HTML contains complete Housing, Drink-Driving, and Diabetes Warning cards marked with `data-domain="housing"`, `data-domain="safelimit"`, and `data-domain="diabetes"`. Edit their headings, descriptions, or feature tables directly in the Qualtrics HTML editor. The script displays the card matching `appId`; Housing remains visible as a fallback if JavaScript does not run.

The participant-facing introductions and prompts use outcome names such as Cheap/Expensive, Above Limit/Below Limit, and Diabetes/No Diabetes; they do not introduce numeric class codes.

## Tutorials

The same block contains two interface tutorials copied from the full experimental setup:

- **Basic Interface** is shown for every condition. Its iframe receives `tutorialCallouts=basic` and displays markers 1-4 for Attribute, Value, Low/High or Range/Options, and AI prediction.
- **AI Explanation** displays the tutorial matching `xaiType`. Attribution explains the Influence column and colors; counterfactual explains the Counter-example column and changes. Both use markers 1-2 for the explanation graphic and explanatory sentence. The question hides itself when `xaiType=none`.

Both tutorial texts and example instances switch with `appId`, so all three domains use their own attributes, labels, examples, and ranges. `tutorialCallouts` is intentionally used only on these tutorial iframes; ordinary training and testing cases do not display numbered markers.

The first screening item also switches its answer options by domain. Choice 1 remains the correct choice for every domain so the existing Survey Flow scoring branches remain valid.

## Editing the testing question

The testing task text is part of the Qualtrics question HTML, immediately above the iframe. It is not generated inside the instance interface. Each testing question contains Housing, Drink-Driving, and Diabetes Warning paragraphs marked with `data-domain`; the shared JavaScript displays the paragraph matching `appId`.

In Qualtrics, open either testing block, select its Descriptive Text question, open the HTML editor, and edit the paragraphs with class `cf-test-prompt`. Do not remove the `data-domain` attributes unless domains should display the same text.

When the iframe is in testing/simulation mode, it does not generate a task question or display an attribution/counterfactual narrative. Those elements could duplicate the Qualtrics prompt or reveal the counterfactual answer. The iframe shows only the original profile, AI prediction, and editable Changes column. The task wording remains entirely in the Qualtrics question HTML.

## Explanation conditions

The shipped QSF sets `xaiType=attribution`. Supported values are:

- `attribution`
- `counterfactual`
- `none`

To run explanation conditions between subjects, replace the single `xaiType` value with a Qualtrics Randomizer containing three Embedded Data elements. Place that randomizer before the tutorial and retain the same field name.

## If rebuilding manually in Qualtrics

Create one Descriptive Text question inside each looped block:

1. Paste the matching HTML from `question-html.md` into the question's HTML editor.
2. Paste all of `qualtrics-frame.js` into that question's JavaScript editor.
3. Set Loop & Merge field 1 to ten rows numbered `0` to `9` for training and for each testing block. Use **All** for Loop & Merge randomization in all three blocks.
4. Define the four Embedded Data fields listed above before the blocks.

The label-0 testing question uses `data-test-label="0"`; the label-1 testing question uses `data-test-label="1"`. Put both blocks under a Survey Flow Randomizer and set it to present both elements.

The same JavaScript is used for training and testing; the question's `data-phase` attribute selects the behavior.

## Viewing the fixed instances

From the repository root, run:

```powershell
python -m http.server 8767 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8767/qualtrics/instance-browser.html
```

The browser lets you switch domain, training/testing phase, and explanation type; click any row to view that exact instance. It also displays the instance ID, AI label, and top feature pair.

## Collected data

`training_log_json` stores domain, case number, instance ID, assigned explanation, selected label, correct label, correctness, and response time.

`testing_log_json` stores domain, direction, case number within that direction, instance ID, original prediction, the most recent visible edits, raw edited feature values, any feedback/prediction returned by the UI, and response time.

Qualtrics embedded-data values are strings, so parse these two columns as JSON after export.

## Regeneration

After changing models or source data, regenerate the fixed static bundle and QSF:

```powershell
python scripts\generate_static_experiment.py --preserve-training
python scripts\build_recourse_v10.py
```

The generator fails instead of relaxing either the testing 10/10 label balance or the one-case-per-feature-pair training constraint.
