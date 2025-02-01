const express = require('express');
const multer = require('multer');
const { generateCourse } = require('../controllers/courseController');

const router = express.Router();
const upload = multer(); // Initialize multer

// Route to generate course
router.post('/generate-course', upload.fields([{ name: 'bookPdf' }, { name: 'learningList' }]), generateCourse);

module.exports = router;