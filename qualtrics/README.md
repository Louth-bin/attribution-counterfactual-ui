# Qualtrics setup: Housing and Drink Driving

Start with `UPLOAD_THIS_Qualtrics_Starter.qsf`. Qualtrics owns the looping and response data, while `iframe.html` is the main case UI.

## Study design encoded in the QSF

- The starter defaults to `appId=housing`. Change that Embedded Data value to `safelimit` for Drink Driving. The simplest workflow is to duplicate the imported project and use one domain in each copy.
- The default explanation condition is `attribution`.
- Training contains exactly 10 fixed cases in the audited manifest order; no case is selected from a larger pool.
- Testing contains exactly 10 different fixed cases: five label-0 cases and five label-1 cases.
- Each phase has five cases from each predicted label.
- In training, every possible pair of the five most important SHAP features occurs exactly once.
- Training answers are followed by the correct AI output and the assigned explanation.
- Testing has two five-case direction blocks. Qualtrics shows both and randomizes which block comes first.
- Testing requires at least one counterfactual edit before Next becomes available.

The auditable IDs and labels are in `case-manifest.json`.

## Deployment workflow

1. Publish this repository's web assets over HTTPS. The default URL is `https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html`. The deployment must include `iframe.html`, `iframe.js`, `index.css`, `static-model.js`, `static/experiment-data.js`, and any libraries referenced by `iframe.html`.
2. In Qualtrics, create a project and choose **Import a QSF**, then import `UPLOAD_THIS_Qualtrics_Starter.qsf`.
3. Open **Survey Flow**. Confirm that the first Embedded Data element contains:
   - `ui_base_url`
   - `xaiType`
   - `training_log_json`
   - `testing_log_json`
4. Change `ui_base_url` from `https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html` if the UI is hosted elsewhere.
5. A second Embedded Data element sets `appId=housing`. Use `safelimit` for Drink Driving.
6. Open the training block and verify its Loop & Merge rows are `0` through `9`. Each testing block should have rows `0` through `4`. Loop & Merge randomization is **None** for all three blocks.
7. In Survey Flow, verify that the two testing blocks are nested under one Randomizer configured to present **2 of 2** elements. This randomizes their order while showing both directions.
8. Preview each domain after changing `appId`.
9. Run one complete response per domain before publishing. Export the response and verify that both JSON log columns contain ten records.

## Training answers and saved data

The two training answer buttons appear directly below the embedded profile. They are created by `qualtrics-frame.js`, so the participant selects the answer on the Qualtrics page rather than inside the iframe.

Each click immediately updates `training_log_json`. The record contains the domain, case number, instance ID, explanation condition, selected prediction, correct prediction, whether the answer was correct, and response time. Clicking Next saves the same record again as a final safeguard. Records are keyed by instance ID, so revising or revisiting a case updates that case without duplicating it.

Testing edits are saved to `testing_log_json`. Its records include the tested direction, instance ID, original prediction, edited values, feedback, and response time. The two testing blocks can both use Loop & Merge rows `0` through `4` because records are keyed by instance ID rather than loop number.

## Editing the application scenario

The **Domain and Basic Interface** block contains an **Application scenario** Descriptive Text question. Its HTML contains complete Housing and Drink-Driving cards marked with `data-domain="housing"` and `data-domain="safelimit"`. Edit their headings, descriptions, feature tables, or task instructions directly in the Qualtrics HTML editor. The script displays the card matching `appId`; Housing remains visible as a fallback if JavaScript does not run.

## Editing the testing question

The testing task text is part of the Qualtrics question HTML, immediately above the iframe. It is not generated inside the instance interface. Each testing question contains one Housing paragraph and one Drink-Driving paragraph marked with `data-domain`; the shared JavaScript displays the paragraph matching `appId`.

In Qualtrics, open either testing block, select its Descriptive Text question, open the HTML editor, and edit the paragraphs with class `cf-test-prompt`. Do not remove `data-domain="housing"` or `data-domain="safelimit"` unless both domains should display the same text.

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
3. Set Loop & Merge field 1 to ten rows numbered `0` to `9` for training and five rows numbered `0` to `4` for each testing block.
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
python scripts\generate_static_experiment.py
python scripts\build_qualtrics_qsf.py --source "C:\path\to\source.qsf"
```

The generator fails instead of relaxing either the 5/5 label balance or the one-case-per-feature-pair training constraint.
