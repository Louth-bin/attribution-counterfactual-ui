import { createSign } from "node:crypto";
import { readFile } from "node:fs/promises";

const participants = process.argv.slice(2);
if (!participants.length || !process.env.GOOGLE_APPLICATION_CREDENTIALS) {
    console.error("Usage: GOOGLE_APPLICATION_CREDENTIALS=<service-account.json> node scripts/audit-live-firestore.mjs P002 P003");
    process.exit(1);
}

const credentials = JSON.parse(await readFile(process.env.GOOGLE_APPLICATION_CREDENTIALS, "utf8"));
const encode = (value) => Buffer.from(JSON.stringify(value)).toString("base64url");
const now = Math.floor(Date.now() / 1000);
const unsignedJwt = `${encode({ alg: "RS256", typ: "JWT" })}.${encode({
    iss: credentials.client_email,
    scope: "https://www.googleapis.com/auth/datastore",
    aud: credentials.token_uri,
    iat: now,
    exp: now + 3600,
})}`;
const signer = createSign("RSA-SHA256");
signer.update(unsignedJwt);
const assertion = `${unsignedJwt}.${signer.sign(credentials.private_key, "base64url")}`;
const tokenResponse = await fetch(credentials.token_uri, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
        grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
        assertion,
    }),
});
if (!tokenResponse.ok) throw new Error(`OAuth failed: ${tokenResponse.status} ${await tokenResponse.text()}`);
const { access_token: accessToken } = await tokenResponse.json();
const baseUrl = `https://firestore.googleapis.com/v1/projects/${credentials.project_id}/databases/(default)/documents`;
const headers = { authorization: `Bearer ${accessToken}`, "content-type": "application/json" };

function decode(value) {
    if (!value || typeof value !== "object") return value;
    if ("nullValue" in value) return null;
    if ("stringValue" in value) return value.stringValue;
    if ("booleanValue" in value) return value.booleanValue;
    if ("integerValue" in value) return Number(value.integerValue);
    if ("doubleValue" in value) return Number(value.doubleValue);
    if ("timestampValue" in value) return value.timestampValue;
    if ("arrayValue" in value) return (value.arrayValue.values ?? []).map(decode);
    if ("mapValue" in value) {
        return Object.fromEntries(Object.entries(value.mapValue.fields ?? {}).map(([key, item]) => [key, decode(item)]));
    }
    return value;
}

const decodeDocument = (document) => ({
    id: document.name.split("/").at(-1),
    ...Object.fromEntries(Object.entries(document.fields ?? {}).map(([key, value]) => [key, decode(value)])),
});

async function sessionsFor(participantCode) {
    const response = await fetch(`${baseUrl}:runQuery`, {
        method: "POST",
        headers,
        body: JSON.stringify({
            structuredQuery: {
                from: [{ collectionId: "studySessions" }],
                where: {
                    fieldFilter: {
                        field: { fieldPath: "participantCode" },
                        op: "EQUAL",
                        value: { stringValue: participantCode },
                    },
                },
            },
        }),
    });
    if (!response.ok) throw new Error(`Session query failed: ${response.status} ${await response.text()}`);
    return (await response.json()).flatMap((row) => row.document ? [decodeDocument(row.document)] : []);
}

async function eventsFor(sessionId) {
    const events = [];
    let pageToken = "";
    do {
        const url = new URL(`${baseUrl}/studySessions/${sessionId}/events`);
        url.searchParams.set("pageSize", "1000");
        url.searchParams.set("orderBy", "sequence");
        if (pageToken) url.searchParams.set("pageToken", pageToken);
        const response = await fetch(url, { headers });
        if (!response.ok) throw new Error(`Event query failed: ${response.status} ${await response.text()}`);
        const page = await response.json();
        events.push(...(page.documents ?? []).map(decodeDocument));
        pageToken = page.nextPageToken ?? "";
    } while (pageToken);
    return events;
}

const technicalFields = new Set(["id", "uid", "sessionId", "sequence", "clientAtMs", "elapsedMs", "recordingElapsedMs", "serverAt"]);
for (const participantCode of participants) {
    const sessions = await sessionsFor(participantCode);
    const report = { participantCode, sessionCount: sessions.length, sessions: [] };
    for (const session of sessions.sort((a, b) => String(a.startedAt).localeCompare(String(b.startedAt)))) {
        const events = await eventsFor(session.id);
        const typeCounts = {};
        const fieldBytes = {};
        const fingerprints = new Map();
        let exactRedundantEvents = 0;
        let rapidSimulationEvents = 0;
        let previousSimulation = null;
        for (const event of events) {
            typeCounts[event.eventType] = (typeCounts[event.eventType] ?? 0) + 1;
            const payload = Object.fromEntries(Object.entries(event).filter(([key]) => !technicalFields.has(key)));
            const fingerprint = JSON.stringify(payload);
            if (fingerprints.has(fingerprint)) exactRedundantEvents += 1;
            else fingerprints.set(fingerprint, event.sequence);
            for (const [key, value] of Object.entries(payload)) {
                fieldBytes[key] = (fieldBytes[key] ?? 0) + JSON.stringify(value).length;
            }
            if (event.eventType === "simulation_changed") {
                if (previousSimulation &&
                    event.caseId === previousSimulation.caseId &&
                    event.recordingElapsedMs - previousSimulation.recordingElapsedMs <= 500) {
                    rapidSimulationEvents += 1;
                }
                previousSimulation = event;
            }
        }
        report.sessions.push({
            sessionId: session.id,
            startedAt: session.startedAt,
            completedAt: session.completedAt ?? null,
            eventCount: events.length,
            exactRedundantEvents,
            rapidSimulationEvents,
            typeCounts,
            largestPayloadFields: Object.entries(fieldBytes)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10)
                .map(([field, bytes]) => ({ field, bytes })),
        });
    }
    console.log(JSON.stringify(report, null, 2));
}
