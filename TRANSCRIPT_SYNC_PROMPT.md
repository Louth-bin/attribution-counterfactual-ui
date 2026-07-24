# Transcript and interaction sync prompt

```text
You are aligning participant transcripts with study interaction logs.

Inputs:
- Participant code: [CODE]
- Transcript(s): [PASTE; retain their timestamps]
- Interaction JSON file(s): [PASTE]
- Optional spoken sync-marker timestamp per transcript/session: [TIMESTAMP OR NONE]

There may be more than one transcript or study session for the same participant. Match each transcript to the most plausible session using timestamps, duration, event order, phase/case IDs, and any sync marker. Never merge two sessions silently. Label each matched pair with its sessionId; if the match is uncertain, state the alternatives.

For each matched pair:
1. Treat recordingElapsedMs as milliseconds from the Start-button time zero. If a spoken sync marker is supplied, calculate one offset for that pair and apply it consistently.
2. Preserve every transcript word and original timestamp verbatim.
3. Insert interaction annotations at the nearest transcript timestamp in this compact form:
   [INT HH:MM:SS | eventType | phase | caseId | essential detail]
4. Keep only essential event-specific detail:
   - navigation/screen: destination or screenId
   - answer: selected answer/label and correctness
   - instance/explanation: prediction plus displayed influences or counterfactual changes
   - simulation: changed attribute(s), old -> new displayed value, and normalized old -> new value
5. Omit technical/storage fields (uid, document id, serverAt, clientAtMs, schemaVersion, userAgent, pageUrl), nulls, unchanged values, repeated attribute arrays, visible text already present in the transcript, and exact duplicate events.
6. Do not delete distinct rapid interactions merely because they are similar. If several events are redundant render snapshots with the same semantic content, keep the earliest and note the number omitted.
7. Flag gaps, conflicting timestamps, cross-session overlaps, and ambiguous matches; do not guess.

Output only:
- a short session-matching note,
- the verbatim timestamped transcript with inline [INT ...] annotations,
- a short ambiguity/omission note if needed.
```
