const { GoogleAIFileManager } = require("@google/generative-ai/server");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const fs = require('fs');
const path = require('path');
const os = require('os');

const generateCourse = async (req, res) => {
    console.log(req.files); // Log the uploaded files
    console.log(req.body); // Log the request body

    if (!req.files || !req.files['bookPdf']) {
        console.log("Missing required book PDF file.");
        return res.status(400).json({ error: "Missing required book PDF file." });
    }

    const tempFiles = [];

    try {
        const { prompt } = req.body;
        const bookPdfFile = req.files['bookPdf'][0];
        const learningListFile = req.files['learningList']?.[0];
        
        // Create temporary file for book PDF
        const bookTempPath = path.join(os.tmpdir(), `book-${Date.now()}.pdf`);
        fs.writeFileSync(bookTempPath, bookPdfFile.buffer);
        tempFiles.push(bookTempPath);

        // Create temporary file for learning list if it exists
        let learningListTempPath;
        if (learningListFile) {
            learningListTempPath = path.join(os.tmpdir(), `learning-${Date.now()}.pdf`);
            fs.writeFileSync(learningListTempPath, learningListFile.buffer);
            tempFiles.push(learningListTempPath);
        }

        const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
        const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);
        const model = genAI.getGenerativeModel({ model: 'models/gemini-1.5-flash' });

        // Upload files and prepare content parts
        const bookUpload = await fileManager.uploadFile(bookTempPath, {
            mimeType: "application/pdf",
            displayName: bookPdfFile.originalname,
        });

        const contentParts = [
            {
                fileData: {
                    fileUri: bookUpload.file.uri,
                    mimeType: bookUpload.file.mimeType,
                },
            }
        ];

        if (learningListFile) {
            const learningUpload = await fileManager.uploadFile(learningListTempPath, {
                mimeType: "application/pdf",
                displayName: learningListFile.originalname,
            });
            contentParts.push({
                fileData: {
                    fileUri: learningUpload.file.uri,
                    mimeType: learningUpload.file.mimeType,
                },
            });
        }

        contentParts.push(prompt || 'Generate a course outline based on this document');

        // Generate content
        const result = await model.generateContent(contentParts);

        res.json({ 
            google: result.response.text(), 
            result: result 
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Internal Server Error' });
    } finally {
        // Cleanup temporary files
        tempFiles.forEach(tempFile => {
            try {
                fs.unlinkSync(tempFile);
            } catch (err) {
                console.error('Error deleting temporary file:', err);
            }
        });
    }
};

module.exports = { generateCourse };