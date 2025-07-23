# Advanced Movie Recommendation System

A comprehensive movie recommendation system implementing multiple machine learning algorithms with an interactive React frontend.

## 🎬 Overview

This project demonstrates three state-of-the-art recommendation algorithms:
- **SVD Matrix Factorization**: Decomposes user-item matrix using Singular Value Decomposition
- **NMF (Non-negative Matrix Factorization)**: Finds interpretable patterns in non-negative data
- **Neural Collaborative Filtering**: Deep learning approach with embeddings for complex interactions

## 🚀 Quick Start

### Prerequisites
- Node.js (14+ recommended)
- npm or yarn

### Installation

1. **Create React App:**
   ```bash
   npx create-react-app movie-recommender
   cd movie-recommender
   ```

2. **Install Dependencies:**
   ```bash
   npm install lucide-react
   ```

3. **Add Tailwind CSS:**
   Add this to `public/index.html` in the `<head>` section:
   ```html
   <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
   ```

4. **Replace App.js:**
   Replace the contents of `src/App.js` with the React component code

5. **Start the Application:**
   ```bash
   npm start
   ```

## 🐳 Run with Docker

You can also run the application using Docker.

Build the Docker image (from the root directory):

```bash
docker build -t movie-recommender .
```
Run the container:

```bash
docker run -p 3000:3000 movie-recommender
```

Access the app:

Open your browser and go to `http://localhost:3000`

## 💻 Usage

1. **Select Algorithm:** Choose between SVD, NMF, or Neural CF
2. **Pick User:** Select a user ID (1-20) 
3. **Set Recommendations:** Choose number of movies to recommend (3-15)
4. **Train Models:** Click "Train All Models" to see realistic training simulation
5. **Compare Results:** Switch between algorithms to see different recommendations

## 🔧 Features

- **Interactive Training Simulation**: Watch models train with realistic progress bars
- **Algorithm Comparison**: See how different methods produce different recommendations
- **Performance Metrics**: View RMSE and MAE for each trained model
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Dynamic recommendations based on selected parameters

## 📊 Algorithms Explained

### SVD Matrix Factorization
- Decomposes the user-item rating matrix into lower-dimensional factors
- Best for: General collaborative filtering with good accuracy
- Time Complexity: O(k * iterations * non-zero entries)

### Non-negative Matrix Factorization (NMF)  
- Constrains factors to be non-negative for interpretability
- Best for: When you need explainable recommendations
- Time Complexity: O(k * iterations * matrix size)

### Neural Collaborative Filtering
- Uses deep learning with user/item embeddings
- Best for: Capturing complex, non-linear user-item interactions
- Time Complexity: O(epochs * batch_size * network depth)

## 📁 File Structure

```
movie-recommender/
├── public/
│   └── index.html          # Tailwind CSS link added here
├── src/
│   ├── App.js              # Main React component
│   └── index.js            # Entry point
└── README.md               # This file
```

## 🎯 Demo Data

The application uses simulated MovieLens-style data:
- **Users**: 20 simulated users
- **Movies**: 10 popular movies with ratings
- **Recommendations**: Algorithm-specific suggestions with confidence scores

## 🔄 Real Implementation

For a production system using actual MovieLens data, see the included Python implementation (`paste.txt`) which:
- Downloads real MovieLens 100K dataset
- Implements actual SVD, NMF, and Neural CF models
- Provides real training, evaluation, and recommendations
- Requires: pandas, scikit-learn, tensorflow, numpy

## 🤝 Contributing

Feel free to:
- Add new recommendation algorithms
- Improve the UI/UX
- Add more evaluation metrics
- Implement real data integration

## 📄 License

MIT License - Feel free to use and modify as needed.

---

*Built with React, Tailwind CSS, and ❤️ for machine learning*
