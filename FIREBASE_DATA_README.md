# Study interaction data

## One-time Firebase setup

1. In Firebase Console, create Cloud Firestore in **Production mode**.
2. In **Authentication > Sign-in method**, enable **Anonymous** authentication.
3. In **Firestore Database > Rules**, paste `firestore.rules` and publish it.
4. Serve the project over HTTP(S); do not open the HTML as a `file://` URL.

You may provide a non-identifying participant code in the study URL:

```text
experimental.html?dataset=diabetes&explanation=attribution&participant=P001
```

If `participant` is omitted, the Start button atomically assigns the next database-backed code (`P001`, `P002`, ...). Do not manually lower or delete `studyMetadata/participantCounter` after data collection begins, because that could reuse an existing code. Re-publish `firestore.rules` after any rules change.

The participant first sees the domain introduction with a **Start** button. The button click is Firebase time zero and creates the `recording_started` event. Its position in the external recording/transcript can be aligned separately during analysis.

## Stored data

```text
studySessions/{sessionId}
studySessions/{sessionId}/events/{sequence}
```

Every screen rendered after Start is stored as `screen_viewed`, including its visible text, controls, phase, screen ID, and only the data visible in that condition. Navigation, screening answers, training answers, feedback, and simulation changes are separate events. Instance screens save the prediction, values, and normalized values. Attribution feedback adds only the displayed influences. Counterfactual feedback and simulation events add only attributes that visibly changed, with their old/new values and normalized values. `recordingElapsedMs` is the offset from the Start button.

Firestore does not permit arrays directly nested inside arrays. The logger therefore represents inner arrays (for example, individual feature ranges or attribution matrix rows) as `{ "items": [...] }`. Important instance values are also duplicated as clearly named top-level event fields so they are easy to inspect in Firebase Console.

## Download a participant

The Firebase Console is suitable for inspection, but not a convenient one-session JSON export. Install the Admin SDK once:

```powershell
npm install firebase-admin
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\secure-location\service-account.json"
node scripts/export-study-session.mjs P001 P001-interactions.json
```

Create the service-account JSON from **Firebase Console > Project settings > Service accounts > Generate new private key**. The exporter accepts either a participant code or exact session ID, orders events by sequence, and converts timestamps to ISO-8601.

Alternatively, use Google Cloud's managed Firestore export for a full database backup. Never put a Firebase Admin service-account key in this repository or browser code. Keep it outside the project on the researcher's computer.

## Transcript-agent handoff

Give the next agent the transcript, exported interaction JSON, and this prompt:

```text
Align interactions using event.recordingElapsedMs (milliseconds from transcript time zero). Insert concise annotations at the nearest transcript timestamp, while preserving the transcript verbatim. Include phase, caseId, eventType, answer or changed attribute, displayed/raw/normalized values, prediction, and explanation values when relevant. If the spoken “sync now” marker differs from zero, calculate one offset and apply it consistently. Flag ambiguous matches rather than guessing.
```

Keep participant codes pseudonymous. Do not store names, emails, or transcript text unless this is covered by consent and the study data-management plan.
