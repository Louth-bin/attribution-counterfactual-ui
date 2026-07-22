import { applicationDefault, cert, initializeApp } from "firebase-admin/app";
import { getFirestore } from "firebase-admin/firestore";
import { readFile, writeFile } from "node:fs/promises";

const [identifier, outputPath = `${identifier || "study-session"}-interactions.json`] = process.argv.slice(2);
if (!identifier) {
    console.error("Usage: node scripts/export-study-session.mjs <session-id-or-participant-code> [output.json]");
    process.exit(1);
}

const serviceAccountPath = process.env.GOOGLE_APPLICATION_CREDENTIALS;
const credential = serviceAccountPath
    ? cert(JSON.parse(await readFile(serviceAccountPath, "utf8")))
    : applicationDefault();
initializeApp({ credential, projectId: "attribution-counterfactual" });
const db = getFirestore();

let sessionSnapshot = await db.collection("studySessions").doc(identifier).get();
if (!sessionSnapshot.exists) {
    const matches = await db.collection("studySessions")
        .where("participantCode", "==", identifier)
        .get();
    sessionSnapshot = matches.docs.sort((first, second) =>
        (second.data().startedAt?.toMillis?.() ?? 0) - (first.data().startedAt?.toMillis?.() ?? 0)
    )[0];
}
if (!sessionSnapshot?.exists) {
    throw new Error(`No session found for '${identifier}'.`);
}

const eventSnapshots = await sessionSnapshot.ref.collection("events").orderBy("sequence").get();
const convert = (value) => {
    if (value?.toDate instanceof Function) return value.toDate().toISOString();
    if (Array.isArray(value)) return value.map(convert);
    if (value && typeof value === "object") {
        return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, convert(item)]));
    }
    return value;
};
const output = {
    session: convert({ id: sessionSnapshot.id, ...sessionSnapshot.data() }),
    events: eventSnapshots.docs.map((snapshot) => convert({ id: snapshot.id, ...snapshot.data() })),
};
await writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");
console.log(`Exported ${output.events.length} events to ${outputPath}`);
