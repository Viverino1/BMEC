// Simple Node.js Express server to accept image uploads and return JSON
const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const os = require("os");
const { spawn } = require("child_process");
const cors = require("cors");
const app = express();
app.use(cors());
const PORT = 3007;

// Set up storage for uploaded files (in memory for demo)
const storage = multer.memoryStorage();
const upload = multer({ storage });

// Endpoint to accept image upload
app.post("/upload", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image uploaded" });
  }
  // Save the uploaded image to a temp file
  const tempDir = os.tmpdir();
  const ext = path.extname(req.file.originalname) || ".jpg";
  const tempPath = path.join(tempDir, `upload_${Date.now()}${ext}`);
  fs.writeFileSync(tempPath, req.file.buffer);

  // Run detection.py with the temp image
  const pythonProcess = spawn("python3", [
    path.join(__dirname, "src", "detection.py"),
    tempPath,
  ]);

  let stdout = "";
  let stderr = "";
  pythonProcess.stdout.on("data", (data) => {
    stdout += data.toString();
  });
  pythonProcess.stderr.on("data", (data) => {
    stderr += data.toString();
  });
  pythonProcess.on("close", (code) => {
    fs.unlink(tempPath, () => {}); // Clean up temp file
    if (code !== 0) {
      return res
        .status(500)
        .json({ error: "Python script failed", details: stderr });
    }
    try {
      const result = JSON.parse(stdout);
      res.json(result);
    } catch (e) {
      res
        .status(500)
        .json({ error: "Invalid JSON from detection.py", details: stdout });
    }
  });
});

app.get("/", (req, res) => {
  res.send(
    '<h2>Node Image Upload Server is running.</h2><p>POST an image to /upload with form-data key "image".</p>'
  );
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(
    `Server running on port ${PORT} and accessible on your local network.`
  );
});
