var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
const multer = require('multer');
const axios = require('axios');
const cors = require('cors');
var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');
const courseRoutes = require('./routes/courseRoutes');
require('dotenv').config(); // Load environment variables
var app = express();

// view engine setup

app.use(cors()); // Enable CORS for all routes
app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());

app.use('/', indexRouter);
app.use('/users', usersRouter);
app.use('/courses', courseRoutes); // Use course routes

const upload = multer({ dest: 'uploads/' }); // Directory for uploaded files

// Route to generate course
app.post('/generate-course', upload.fields([{ name: 'bookPdf' }, { name: 'learningList' }]), async (req, res) => {
    try {
        const { prompt } = req.body;
        const bookPdf = req.files['bookPdf'][0];
        const learningList = req.files['learningList'][0];

        // Prepare data for Gemini and OpenAI APIs
        const data = {
            bookPdf: bookPdf.path,
            learningList: learningList.path,
            prompt
        };

        // Call Gemini API
        const geminiResponse = await axios.post('GEMINI_API_URL', data, {
            headers: { 'Authorization': `Bearer ${process.env.GEMINI_API_KEY}` }
        });

        // Call OpenAI API
        const openaiResponse = await axios.post('OPENAI_API_URL', data, {
            headers: { 'Authorization': `Bearer ${process.env.OPENAI_API_KEY}` }
        });

        // Combine responses and send back
        res.json({ gemini: geminiResponse.data, openai: openaiResponse.data });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  res.status(404).json({ error: 'Not Found' });
});

// Error handler
app.use(function(err, req, res, next) {
  console.error(err);
  res.status(err.status || 500).json({ error: 'Internal Server Error' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

module.exports = app;
