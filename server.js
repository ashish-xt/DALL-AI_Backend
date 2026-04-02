require("dotenv").config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const pdfExtract = require("pdf-extraction");
const { EdgeTTS } = require("node-edge-tts");
const fs = require("fs");
const path = require("path");
const os = require("os");

const app = express();
app.use(cors());

// HEALTH CHECK
app.get("/", (req, res) => {
  res.json({
    status: "ok",
    message: "Backend service is alive",
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
  });
});

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// ==========================================
// ROUTE 1: Transcribe Audio (Whisper Turbo)
// ==========================================
app.post("/api/transcribe", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file)
      return res.status(400).json({ error: "No audio file provided." });

    const response = await fetch(
      "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo",
      {
        headers: {
          Authorization: `Bearer ${process.env.HF_TOKEN}`,
          "Content-Type": "audio/webm",
        },
        method: "POST",
        body: req.file.buffer,
      },
    );

    const result = await response.json();
    if (result.error) throw new Error(result.error);

    res.json({ text: result.text.trim() });
  } catch (error) {
    console.error("Transcription error:", error);
    res.status(500).json({ error: "Failed to transcribe audio." });
  }
});

// ==========================================
// ROUTE 2: Analyze Resume & Generate Questions
// ==========================================
app.post("/api/analyze-resume", upload.single("resume"), async (req, res) => {
  try {
    if (!req.file)
      return res.status(400).json({ error: "No resume provided." });

    const pdfData = await pdfExtract(req.file.buffer);
    const resumeText = pdfData.text.substring(0, 4000);

    const questionCount = req.body.questionCount || 5;
    const difficulty = req.body.difficulty || "Medium";

    console.log(`Generating ${questionCount} ${difficulty}-level questions...`);

    const prompt = `
    You are an expert technical interviewer. I will provide you with a candidate's parsed resume text. 
    Generate exactly ${questionCount} customized interview questions based on their resume.
    
    DIFFICULTY LEVEL: ${difficulty}
    - If Easy: Focus on foundational concepts, definitions, and basic explanations of their skills.
    - If Medium: Focus on practical application, problem-solving, and how they built their projects.
    - If Hard: Focus on edge cases, system design, performance optimization, and deep technical architecture.

    Return ONLY a raw JSON array of strings. Do not include any other conversational text.
    Example: ["Question 1?", "Question 2?"]

    CANDIDATE RESUME:
    ${resumeText}
    `;

    const response = await fetch(
      "https://router.huggingface.co/v1/chat/completions",
      {
        headers: {
          Authorization: `Bearer ${process.env.HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        method: "POST",
        body: JSON.stringify({
          model: "Qwen/Qwen2.5-7B-Instruct",
          messages: [
            {
              role: "system",
              content:
                "You are an expert technical interviewer. You must output only raw JSON arrays.",
            },
            { role: "user", content: prompt },
          ],
          max_tokens: 800,
          temperature: 0.7,
        }),
      },
    );

    const result = await response.json();
    if (result.error) throw new Error(result.error);

    let rawOutput = result.choices[0].message.content.trim();
    const jsonMatch = rawOutput.match(/\[[\s\S]*\]/);
    if (!jsonMatch) throw new Error("AI did not return a valid JSON array.");

    const questionsArray = JSON.parse(jsonMatch[0]);
    res.json({ questions: questionsArray });
  } catch (error) {
    console.error("Resume analysis error:", error);
    res.status(500).json({ error: "Failed to generate questions." });
  }
});

// ==========================================
// ROUTE 3: Evaluate Interview Answers
// ==========================================
app.post("/api/evaluate", express.json(), async (req, res) => {
  try {
    // Notice we completely removed faceLostCount from the prompt!
    // It will no longer affect the AI's score.
    const { interviewData } = req.body;

    if (!interviewData || interviewData.length === 0) {
      return res.status(400).json({ error: "No interview data provided." });
    }

    const formattedTranscript = interviewData
      .map(
        (item, index) =>
          `Q${index + 1}: ${item.question}\nCandidate Answer: ${item.answer}`,
      )
      .join("\n\n");

    const prompt = `
    You are DALL-AI, an expert technical mentor. You are speaking DIRECTLY to the user face-to-face to give them feedback on their interview.
    Review the following transcript of questions and the user's answers.
    
    CRITICAL SCORING RULES (STRICT GRADING):
    1. You MUST evaluate the technical accuracy, depth, and relevance of the user's answers.
    2. If the user provides NO answer, says "I don't know", or provides empty/gibberish text, you MUST give a score of 0 for that question.
    3. Be a brutal, strict grader. An overall score of 80+ should ONLY be given for flawless, deep, highly technical answers.
    
    CRITICAL PRONOUN RULES:
    1. You MUST address the user as "you" and "your".
    2. You are STRICTLY FORBIDDEN from using the words "the candidate", "they", "he", or "she".

    Evaluate their performance and return ONLY a raw JSON object with the following exact keys:
    - "overallScore": An integer from 0 to 100 based STRICTLY on the technical quality of their answers.
    - "strengths": An array of 3 strings detailing what they answered well. (Must start with "You...")
    - "weaknesses": An array of 3 strings detailing where they lacked knowledge. (Must start with "You...")
    - "improvementTips": A detailed paragraph giving direct, technical advice on how to improve.

    Do not include any markdown formatting outside the JSON. Output ONLY valid JSON.

    INTERVIEW TRANSCRIPT:
    ${formattedTranscript}
    `;

    const response = await fetch(
      "https://router.huggingface.co/v1/chat/completions",
      {
        headers: {
          Authorization: `Bearer ${process.env.HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        method: "POST",
        body: JSON.stringify({
          model: "Qwen/Qwen2.5-7B-Instruct",
          messages: [
            {
              role: "system",
              content:
                "You are a strict grading system. Output only valid JSON objects.",
            },
            { role: "user", content: prompt },
          ],
          max_tokens: 1000,
          temperature: 0.1, // Lowered temperature to make the AI more analytical and less "creative" with scores
        }),
      },
    );

    const result = await response.json();
    if (result.error) throw new Error(result.error);

    let rawOutput = result.choices[0].message.content.trim();
    const jsonMatch = rawOutput.match(/\{[\s\S]*\}/);
    if (!jsonMatch)
      throw new Error("AI did not return a valid JSON evaluation.");

    const evaluationData = JSON.parse(jsonMatch[0]);
    res.json(evaluationData);
  } catch (error) {
    console.error("Evaluation error:", error);
    res.status(500).json({ error: "Failed to evaluate interview." });
  }
});

// ==========================================
// ROUTE 4: Generate Realistic AI Voice (100% FREE)
// ==========================================
app.post("/api/generate-speech", express.json(), async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: "No text provided." });

    const tts = new EdgeTTS({
      voice: "en-IN-NeerjaNeural",
      lang: "en-IN",
      outputFormat: "audio-24khz-48kbitrate-mono-mp3",
      rate: "-10%",
      pitch: "-5%",
    });

    const tempFilePath = path.join(os.tmpdir(), `tts-${Date.now()}.mp3`);
    await tts.ttsPromise(text, tempFilePath);

    const audioBuffer = fs.readFileSync(tempFilePath);
    res.set("Content-Type", "audio/mpeg");
    res.send(audioBuffer);

    fs.unlinkSync(tempFilePath);
  } catch (error) {
    console.error("Speech generation error:", error);
    res.status(500).json({ error: "Failed to generate audio." });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`DALL-AI Backend running on http://localhost:${PORT}`);
});
