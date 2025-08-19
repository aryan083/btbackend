# Booktube — Personalized AI-Powered Learning Platform

Booktube is a generative AI-powered web platform that transforms traditional textbooks and syllabi into interactive, bite-sized articles tailored to a learner’s style, preferences, and skill level. Built as part of a Human-Centered Design project, Booktube bridges the gap between bulky academic material and modern, personalized learning methods.

---

## 🔗 Live Preview

👉 [Visit Booktube Here](https://booktube-opal.vercel.app/)

---

## ✨ Key Features

- 📄 **PDF to Topic-wise Articles**: Upload a syllabus and textbook PDF to get structured, easy-to-understand content.
- 🧠 **Personalized Content Generation**: Uses a RAG (Retrieval-Augmented Generation) pipeline powered by Gemini 2.0 Flash to generate articles tailored to user inputs like learning style, skill level, and prompts.
- 🎯 **Recommendation Engine**: Keeps learners engaged by suggesting relevant articles and topics based on learning patterns.
- 🗂️ **Playlists & Read Later**: Save articles to custom playlists, mark them to read later, and keep track of learning history.
- 🖼️ **AI-Picked Thumbnails**: Dynamically selected visuals from Pexels/Unsplash using Gemini for visually appealing article thumbnails.
- 📈 **Progress Tracking**: Monitor learning progress across chapters and subjects.

---

## 🧪 Tech Stack

- **Frontend**: React.js, Tailwind CSS, Bootstrap  
- **Backend/AI**: Gemini 2.0 Flash API, RAG Pipeline, Python (PDF Parsing)  
- **APIs & Libraries**: Unstructured, Unsplash API, Pexels API, Google Cloud Platform

---


## ⚙️ System Design

Booktube's architecture is designed to transform raw educational content into a personalized learning experience. The system follows a multi-stage pipeline, from content ingestion to personalized delivery.

1.  **Content Ingestion**:
    *   The user uploads a textbook (PDF) and a syllabus via the React-based frontend.
    *   The backend, powered by Flask, receives these files.

2.  **Processing & Chunking**:
    *   The uploaded PDF is parsed using the `Unstructured` library to extract raw text and structure.
    *   The content is then segmented into meaningful, topic-wise chunks based on the syllabus.

3.  **Personalized Content Generation (RAG Pipeline)**:
    *   When a user requests an article on a topic, the system uses a Retrieval-Augmented Generation (RAG) pipeline.
    *   **Retrieval**: Relevant chunks of the textbook are retrieved based on the topic and user's learning preferences.
    *   **Augmentation**: The retrieved context is combined with the user's prompt (e.g., "explain this like I'm a beginner").
    *   **Generation**: The augmented prompt is sent to the **Gemini 2.0 Flash API**, which generates a tailored, easy-to-digest article.

4.  **Engagement & Personalization**:
    *   **AI Thumbnails**: Gemini is used to generate relevant search queries for **Pexels/Unsplash APIs** to fetch and assign visually appealing thumbnails to each article.
    *   **Recommendation Engine**: The system tracks user activity and suggests related articles to create a continuous learning path.
    *   **User Features**: Users can save articles to playlists, mark them for later, and track their progress, with all data managed by the backend.

---

## 🖼️ Screenshots

> **Home Page**
> ![Home Page Sample](https://github.com/user-attachments/assets/a4d80816-2c5f-4f0f-98c7-31e7d55dd29f)


> **Generated Article View**
>  
![Article Page Sample](https://github.com/user-attachments/assets/a2271b0d-850d-4d90-9eed-74d9fe0ad27d)

## 👨‍💻 Contributors

- [Aryan Mahida](https://github.com/aryanmahida)
- [Umang Hirani](https://github.com/ukhirani)
- [Jay Mangukiya](https://github.com/jaymangukiya)
