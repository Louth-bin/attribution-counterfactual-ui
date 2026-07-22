import { initializeApp } from "https://www.gstatic.com/firebasejs/12.16.0/firebase-app.js";
import { getAuth, signInAnonymously } from "https://www.gstatic.com/firebasejs/12.16.0/firebase-auth.js";
import { collection, doc, getFirestore, runTransaction, serverTimestamp, setDoc } from "https://www.gstatic.com/firebasejs/12.16.0/firebase-firestore.js";

const firebaseConfig = {
    apiKey: "AIzaSyCEXjSsMzgE1wU1GB0nM3VG7gqG_9jsG70",
    authDomain: "attribution-counterfactual.firebaseapp.com",
    projectId: "attribution-counterfactual",
    storageBucket: "attribution-counterfactual.firebasestorage.app",
    messagingSenderId: "756708259563",
    appId: "1:756708259563:web:9516e2fea27572f03ef0ae",
    measurementId: "G-1T5DFX0K1T",
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);
let sessionRef = null;
let uid = null;
let sequence = 0;
let recordingStartedClientMs = null;
let recordingStartedPerformanceMs = null;
let writeChain = Promise.resolve();
let assignedParticipantCode = null;

function serializable(value, insideArray = false) {
    if (value === undefined || value === null) return null;
    if (typeof value === "number") return Number.isFinite(value) ? value : null;
    if (typeof value !== "object") return value;
    if (Array.isArray(value)) {
        const items = value.map((item) => serializable(item, true));
        // Firestore rejects an array directly nested inside another array.
        return insideArray ? { items } : items;
    }
    return Object.fromEntries(
        Object.entries(value).map(([key, item]) => [key, serializable(item, false)])
    );
}

async function getParticipantCode() {
    const suppliedCode = new URLSearchParams(location.search).get("participant")?.trim();
    if (suppliedCode) return suppliedCode;
    if (assignedParticipantCode) return assignedParticipantCode;

    const counterRef = doc(db, "studyMetadata", "participantCounter");
    assignedParticipantCode = await runTransaction(db, async (transaction) => {
        const snapshot = await transaction.get(counterRef);
        const nextNumber = (snapshot.exists() ? Number(snapshot.data().nextNumber) : 0) + 1;
        transaction.set(counterRef, { nextNumber, updatedAt: serverTimestamp() });
        return `P${String(nextNumber).padStart(3, "0")}`;
    });
    return assignedParticipantCode;
}

async function startSession(metadata = {}) {
    if (sessionRef) return sessionRef.id;
    // Capture time zero at the click, before authentication or network latency.
    recordingStartedClientMs = Date.now();
    recordingStartedPerformanceMs = performance.now();
    const credential = await signInAnonymously(auth);
    uid = credential.user.uid;
    sessionRef = doc(collection(db, "studySessions"));
    const participantCode = await getParticipantCode();
    try {
        await setDoc(sessionRef, {
            uid,
            participantCode,
            ...serializable(metadata),
            sessionId: sessionRef.id,
            startedAt: serverTimestamp(),
            recordingStartedAt: serverTimestamp(),
            recordingStartedClientMs,
            userAgent: navigator.userAgent,
            pageUrl: location.href,
            schemaVersion: 1,
        });
        const syncSaved = await log("recording_started", { syncMarker: true });
        if (!syncSaved) throw new Error("The transcript sync event could not be saved.");
    } catch (error) {
        sessionRef = null;
        throw error;
    }
    return sessionRef.id;
}

function log(eventType, details = {}) {
    if (!sessionRef) return Promise.resolve(false);
    const eventSequence = ++sequence;
    const isSyncEvent = eventType === "recording_started";
    const now = isSyncEvent ? recordingStartedClientMs : Date.now();
    const eventRef = doc(sessionRef, "events", String(eventSequence).padStart(6, "0"));
    const event = {
        uid,
        sessionId: sessionRef.id,
        sequence: eventSequence,
        eventType,
        clientAtMs: now,
        recordingElapsedMs: isSyncEvent ? 0 : now - recordingStartedClientMs,
        elapsedMs: isSyncEvent ? 0 : Math.round(performance.now() - recordingStartedPerformanceMs),
        serverAt: serverTimestamp(),
        ...serializable(details),
    };
    let saved = true;
    writeChain = writeChain.then(() => setDoc(eventRef, event)).catch((error) => {
        saved = false;
        console.error("Study event could not be saved:", error);
        window.dispatchEvent(new CustomEvent("experiment-logging-error", { detail: error }));
    });
    return writeChain.then(() => saved);
}

async function completeSession() {
    if (!sessionRef) return;
    await log("session_completed");
    await setDoc(sessionRef, { completedAt: serverTimestamp(), eventCount: sequence }, { merge: true });
}

window.ExperimentLogger = { startSession, log, completeSession, get sessionId() { return sessionRef?.id ?? null; } };
