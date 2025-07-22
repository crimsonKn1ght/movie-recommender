import React, { useState, useEffect } from 'react';
import { Star, Play, TrendingUp, Brain, BarChart3, Users, Film } from 'lucide-react';

const AdvancedMovieRecommender = () => {
  const [currentMethod, setCurrentMethod] = useState('SVD');
  const [selectedUser, setSelectedUser] = useState(1);
  const [numRecommendations, setNumRecommendations] = useState(5);
  const [isTraining, setIsTraining] = useState(false);
  const [modelAccuracy, setModelAccuracy] = useState({});
  const [trainingProgress, setTrainingProgress] = useState({});
  const [modelsReady, setModelsReady] = useState(false);

  // Simulated movie data and recommendations
  const sampleMovies = [
    { id: 1, title: "The Shawshank Redemption", genre: "Drama", year: 1994, rating: 9.3 },
    { id: 2, title: "Pulp Fiction", genre: "Crime", year: 1994, rating: 8.9 },
    { id: 3, title: "The Dark Knight", genre: "Action", year: 2008, rating: 9.0 },
    { id: 4, title: "Forrest Gump", genre: "Drama", year: 1994, rating: 8.8 },
    { id: 5, title: "Inception", genre: "Sci-Fi", year: 2010, rating: 8.8 },
    { id: 6, title: "The Matrix", genre: "Sci-Fi", year: 1999, rating: 8.7 },
    { id: 7, title: "Goodfellas", genre: "Crime", year: 1990, rating: 8.7 },
    { id: 8, title: "The Godfather", genre: "Crime", year: 1972, rating: 9.2 },
    { id: 9, title: "Interstellar", genre: "Sci-Fi", year: 2014, rating: 8.6 },
    { id: 10, title: "Fight Club", genre: "Drama", year: 1999, rating: 8.8 }
  ];

  const methods = {
    'SVD': {
      name: 'SVD Matrix Factorization',
      description: 'Decomposes user-item matrix using Singular Value Decomposition',
      icon: BarChart3,
      color: 'bg-blue-500',
      accuracy: { rmse: 0.874, mae: 0.693 }
    },
    'NMF': {
      name: 'Non-negative Matrix Factorization',
      description: 'Finds interpretable patterns in non-negative data',
      icon: TrendingUp,
      color: 'bg-green-500',
      accuracy: { rmse: 0.891, mae: 0.712 }
    },
    'Neural': {
      name: 'Neural Collaborative Filtering',
      description: 'Deep learning approach to capture complex user-item interactions',
      icon: Brain,
      color: 'bg-purple-500',
      accuracy: { rmse: 0.856, mae: 0.681 }
    }
  };

  // Simulated recommendations based on method
  const getRecommendations = (method, userId, count) => {
    const baseRecommendations = {
      'SVD': [
        { movie: sampleMovies[0], predictedRating: 4.7, confidence: 0.92 },
        { movie: sampleMovies[2], predictedRating: 4.6, confidence: 0.89 },
        { movie: sampleMovies[4], predictedRating: 4.5, confidence: 0.87 },
        { movie: sampleMovies[7], predictedRating: 4.4, confidence: 0.85 },
        { movie: sampleMovies[9], predictedRating: 4.3, confidence: 0.83 }
      ],
      'NMF': [
        { movie: sampleMovies[1], predictedRating: 4.8, confidence: 0.91 },
        { movie: sampleMovies[3], predictedRating: 4.7, confidence: 0.88 },
        { movie: sampleMovies[5], predictedRating: 4.5, confidence: 0.86 },
        { movie: sampleMovies[6], predictedRating: 4.4, confidence: 0.84 },
        { movie: sampleMovies[8], predictedRating: 4.2, confidence: 0.82 }
      ],
      'Neural': [
        { movie: sampleMovies[7], predictedRating: 4.9, confidence: 0.94 },
        { movie: sampleMovies[0], predictedRating: 4.8, confidence: 0.92 },
        { movie: sampleMovies[2], predictedRating: 4.7, confidence: 0.90 },
        { movie: sampleMovies[4], predictedRating: 4.6, confidence: 0.88 },
        { movie: sampleMovies[9], predictedRating: 4.5, confidence: 0.86 }
      ]
    };

    return baseRecommendations[method].slice(0, count);
  };

  const simulateTraining = () => {
    setIsTraining(true);
    setModelsReady(false);
    setTrainingProgress({});
    setModelAccuracy({});

    // Simulate training each model with realistic timing
    const models = ['SVD', 'NMF', 'Neural'];
    
    models.forEach((model, index) => {
      // Stagger the start times
      setTimeout(() => {
        // Start training this model
        setTrainingProgress(prev => ({
          ...prev,
          [model]: { status: 'training', progress: 0 }
        }));

        // Simulate progress updates
        let progress = 0;
        const progressInterval = setInterval(() => {
          progress += Math.random() * 25 + 10; // Random progress increments
          if (progress >= 100) {
            progress = 100;
            clearInterval(progressInterval);
            
            // Mark model as completed
            setTrainingProgress(prev => ({
              ...prev,
              [model]: { status: 'completed', progress: 100 }
            }));

            // Add model accuracy
            const accuracies = {
              SVD: { rmse: (0.85 + Math.random() * 0.05).toFixed(3), mae: (0.68 + Math.random() * 0.03).toFixed(3) },
              NMF: { rmse: (0.88 + Math.random() * 0.05).toFixed(3), mae: (0.70 + Math.random() * 0.03).toFixed(3) },
              Neural: { rmse: (0.82 + Math.random() * 0.05).toFixed(3), mae: (0.65 + Math.random() * 0.03).toFixed(3) }
            };

            setModelAccuracy(prev => ({
              ...prev,
              [model]: { ...accuracies[model], trained: true }
            }));

            // Check if all models are done
            setTimeout(() => {
              setTrainingProgress(current => {
                const allCompleted = models.every(m => 
                  current[m] && current[m].status === 'completed'
                );
                if (allCompleted) {
                  setIsTraining(false);
                  setModelsReady(true);
                }
                return current;
              });
            }, 100);
            
          } else {
            setTrainingProgress(prev => ({
              ...prev,
              [model]: { status: 'training', progress: Math.round(progress) }
            }));
          }
        }, 200 + index * 100); // Different speeds for different models
        
      }, index * 1000); // Start each model 1 second apart
    });
  };

  const recommendations = getRecommendations(currentMethod, selectedUser, numRecommendations);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 text-white">
      <div className="container mx-auto px-6 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <Film className="w-12 h-12 text-purple-400 mr-4" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Advanced Movie Recommender
            </h1>
          </div>
          <p className="text-gray-300 text-lg">
            Powered by Matrix Factorization & Deep Learning
          </p>
        </div>

        {/* Method Selection */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          {Object.entries(methods).map(([key, method]) => {
            const IconComponent = method.icon;
            return (
              <div
                key={key}
                className={`p-6 rounded-xl border-2 cursor-pointer transition-all duration-300 ${
                  currentMethod === key
                    ? 'border-purple-400 bg-purple-800/30 shadow-lg shadow-purple-400/20'
                    : 'border-gray-600 bg-gray-800/50 hover:border-gray-500'
                }`}
                onClick={() => setCurrentMethod(key)}
              >
                <div className="flex items-center mb-3">
                  <div className={`p-2 rounded-lg ${method.color} mr-3`}>
                    <IconComponent className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-semibold">{method.name}</h3>
                </div>
                <p className="text-gray-400 text-sm mb-3">{method.description}</p>
                {modelAccuracy[key] && (
                  <div className="text-xs text-green-400">
                    RMSE: {method.accuracy.rmse} | MAE: {method.accuracy.mae}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Controls */}
        <div className="bg-gray-800/50 rounded-xl p-6 mb-8">
          <div className="grid md:grid-cols-3 gap-6 items-end">
            <div>
              <label className="block text-sm font-medium mb-2">User ID</label>
              <select
                value={selectedUser}
                onChange={(e) => setSelectedUser(Number(e.target.value))}
                className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-400 focus:border-transparent"
              >
                {[...Array(20)].map((_, i) => (
                  <option key={i + 1} value={i + 1}>User {i + 1}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Recommendations</label>
              <select
                value={numRecommendations}
                onChange={(e) => setNumRecommendations(Number(e.target.value))}
                className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-400 focus:border-transparent"
              >
                {[3, 5, 10, 15].map(num => (
                  <option key={num} value={num}>{num} movies</option>
                ))}
              </select>
            </div>

            <button
              onClick={simulateTraining}
              disabled={isTraining}
              className={`px-6 py-3 rounded-lg font-medium transition-all duration-300 ${
                isTraining 
                  ? 'bg-gray-600 cursor-not-allowed opacity-50' 
                  : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700'
              }`}
            >
              {isTraining ? (
                <div className="flex items-center">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                  Training Models...
                </div>
              ) : modelsReady ? (
                'Retrain Models'
              ) : (
                'Train All Models'
              )}
            </button>
          </div>
        </div>

        {/* Training Progress */}
        {isTraining && (
          <div className="bg-gray-800/50 rounded-xl p-6 mb-8">
            <div className="flex items-center mb-4">
              <Brain className="w-6 h-6 text-purple-400 mr-2 animate-pulse" />
              <h3 className="text-lg font-semibold">Training in Progress</h3>
            </div>
            <div className="space-y-4">
              {Object.entries(methods).map(([key, method]) => {
                const progress = trainingProgress[key] || { status: 'waiting', progress: 0 };
                const IconComponent = method.icon;
                
                return (
                  <div key={key} className="bg-gray-700/30 rounded-lg p-4">
                    <div className="flex items-center mb-3">
                      <div className={`p-2 rounded-lg ${method.color} mr-3`}>
                        <IconComponent className="w-4 h-4" />
                      </div>
                      <span className="font-medium flex-1">{method.name}</span>
                      <span className={`text-sm px-2 py-1 rounded-full ${
                        progress.status === 'completed' ? 'bg-green-600 text-white' :
                        progress.status === 'training' ? 'bg-blue-600 text-white animate-pulse' :
                        'bg-gray-600 text-gray-300'
                      }`}>
                        {progress.status === 'completed' ? 'Completed' :
                         progress.status === 'training' ? `${progress.progress}%` :
                         'Waiting...'}
                      </span>
                    </div>
                    
                    <div className="relative">
                      <div className="w-full bg-gray-600 rounded-full h-3">
                        <div 
                          className={`h-3 rounded-full transition-all duration-500 ${
                            progress.status === 'completed' ? 'bg-green-500' :
                            progress.status === 'training' ? method.color : 'bg-gray-600'
                          }`}
                          style={{ width: `${progress.progress}%` }}
                        ></div>
                      </div>
                      {progress.status === 'training' && (
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse rounded-full"></div>
                      )}
                    </div>
                    
                    {modelAccuracy[key] && (
                      <div className="mt-3 text-sm text-green-400 flex justify-between">
                        <span>RMSE: {modelAccuracy[key].rmse}</span>
                        <span>MAE: {modelAccuracy[key].mae}</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            
            {modelsReady && (
              <div className="mt-4 p-3 bg-green-800/30 border border-green-600 rounded-lg">
                <div className="flex items-center text-green-400">
                  <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></div>
                  <span className="font-medium">All models trained successfully! Try different recommendation methods now.</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Recommendations */}
        <div className="bg-gray-800/50 rounded-xl p-6">
          <div className="flex items-center mb-6">
            <Play className="w-6 h-6 text-purple-400 mr-3" />
            <h3 className="text-xl font-semibold">
              {methods[currentMethod].name} Recommendations for User {selectedUser}
            </h3>
          </div>

          <div className="grid gap-4">
            {recommendations.map((rec, index) => (
              <div
                key={rec.movie.id}
                className="flex items-center p-4 bg-gray-700/50 rounded-lg hover:bg-gray-700/70 transition-all duration-300"
              >
                <div className="w-12 h-12 bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg flex items-center justify-center font-bold text-lg mr-4">
                  {index + 1}
                </div>
                
                <div className="flex-1">
                  <h4 className="text-lg font-semibold mb-1">{rec.movie.title}</h4>
                  <div className="flex items-center text-sm text-gray-400">
                    <span className="mr-4">{rec.movie.genre}</span>
                    <span className="mr-4">{rec.movie.year}</span>
                    <div className="flex items-center">
                      <Star className="w-4 h-4 text-yellow-400 mr-1" />
                      {rec.movie.rating}
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-lg font-semibold text-purple-400 mb-1">
                    {rec.predictedRating.toFixed(1)}/5.0
                  </div>
                  <div className="text-xs text-gray-500">
                    {(rec.confidence * 100).toFixed(0)}% confidence
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Model Stats */}
          <div className="mt-8 p-4 bg-gray-700/30 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Users className="w-5 h-5 text-gray-400 mr-2" />
                <span className="text-sm text-gray-400">
                  Dataset: MovieLens 100K (100,000 ratings, 943 users, 1,682 movies)
                </span>
              </div>
              <div className="text-sm text-gray-400">
                Method: {methods[currentMethod].description}
              </div>
            </div>
          </div>
        </div>

        {/* Performance Comparison */}
        {modelsReady && Object.keys(modelAccuracy).length > 0 && (
          <div className="mt-8 bg-gray-800/50 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-6 flex items-center">
              <BarChart3 className="w-6 h-6 text-purple-400 mr-3" />
              Model Performance Comparison
            </h3>
            <div className="grid md:grid-cols-3 gap-4">
              {Object.entries(methods).map(([key, method]) => (
                <div key={key} className="bg-gray-700/50 rounded-lg p-4">
                  <div className="flex items-center mb-3">
                    <div className={`w-3 h-3 rounded-full ${method.color} mr-2`}></div>
                    <span className="font-medium">{method.name}</span>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">RMSE:</span>
                      <span className="font-mono">{method.accuracy.rmse}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">MAE:</span>
                      <span className="font-mono">{method.accuracy.mae}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 text-xs text-gray-500">
              Lower RMSE and MAE values indicate better prediction accuracy
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AdvancedMovieRecommender;