// api-server/app.js
const express = require('express');
const cors = require('cors');
const { MongoClient } = require('mongodb');
const { exec } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json());

// ===== MongoDB Configuration =====
const MONGO_URI = process.env.MONGO_URI || "mongodb://mongo:27017";
const DB_NAME = "interior_analysis";

let db;

// Połączenie z MongoDB
async function connectToDB() {
    try {
        const client = new MongoClient(MONGO_URI);
        await client.connect();
        db = client.db(DB_NAME);
        console.log("✅ Connected to MongoDB");
    } catch (err) {
        console.error("❌ MongoDB connection error:", err);
    }
}

// ===== Endpoints =====

// ROOT
app.get('/', (req, res) => {
    res.json({
        message: "Interior Analysis API is running!",
        endpoints: {
            health: "/health",
            test: "/test",
            apartments: "/apartments",
            process_pending: "/process-pending",
            process_id: "/process/:id",
            results: "/results",
            export: "/export"
        },
        timestamp: new Date().toISOString()
    });
});

// HEALTH
app.get('/health', (req, res) => {
    res.json({
        status: "OK",
        message: "API is working!",
        timestamp: new Date().toISOString()
    });
});

// TEST
app.get('/test', (req, res) => {
    res.json({ message: "Hello World! Test successful!" });
});

// APARTMENTS — przykładowy endpoint
app.get('/apartments', (req, res) => {
    res.json([{ id: 1, title: "Test apartment" }]);
});

// ====== PENDING PROCESSING ======
app.get('/process-pending', async (req, res) => {
    try {
        const items = await db.collection('pending').find().toArray();
        res.json(items);
    } catch (err) {
        res.status(500).json({ error: err.toString() });
    }
});

// ====== PROCESS ITEM BY ID ======
app.get('/process/:id', async (req, res) => {
    const id = req.params.id;

    try {
        const item = await db.collection('pending').findOne({ id: id });

        if (!item) {
            return res.status(404).json({ error: "Item not found" });
        }

        res.json(item);
    } catch (err) {
        res.status(500).json({ error: err.toString() });
    }
});

// ====== RESULTS ======
app.get('/results', async (req, res) => {
    try {
        const results = await db.collection('results').find().toArray();
        res.json(results);
    } catch (err) {
        res.status(500).json({ error: err.toString() });
    }
});

// ====== EXPORT (TRIGGER PYTHON) ======
app.get('/export', (req, res) => {
    exec("python3 /python/export_data.py", (error, stdout, stderr) => {
        if (error) {
            return res.status(500).json({ error: error.toString() });
        }
        res.json({ stdout, stderr });
    });
});

// ===== Start server =====
const PORT = process.env.PORT || 3000;

app.listen(PORT, '0.0.0.0', async () => {
    console.log(`🚀 API SERVER RUNNING ON PORT ${PORT}`);
    await connectToDB();
});
