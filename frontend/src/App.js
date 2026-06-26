import React, { useCallback, useEffect, useState } from 'react';
import { Star, Play, TrendingUp, Brain, BarChart3, Users, Film, AlertCircle } from 'lucide-react';
import { api } from './api/client';

// Static presentation for each algorithm key. Names, descriptions and metrics
// come from the backend; icons/colors are a frontend concern.
const METHOD_STYLE = {
  SVD: { icon: BarChart3, color: 'bg-blue-500' },
  NMF: { icon: TrendingUp, color: 'bg-green-500' },
  Neural: { icon: Brain, color: 'bg-purple-500' },
};

const STATUS_LABEL = {
  ready: 'Ready',
  training: 'Training',
  not_started: 'Not trained',
  failed: 'Failed',
  unavailable: 'Unavailable',
};

const AdvancedMovieRecommender = () => {
  const [algorithms, setAlgorithms] = useState([]);
  const [users, setUsers] = useState([]);
  const [status, setStatus] = useState(null);

  const [currentMethod, setCurrentMethod] = useState('SVD');
  const [selectedUser, setSelectedUser] = useState(1);
  const [numRecommendations, setNumRecommendations] = useState(5);

  const [recommendations, setRecommendations] = useState([]);
  const [recError, setRecError] = useState(null);
  const [loadingRecs, setLoadingRecs] = useState(false);
  const [error, setError] = useState(null);

  const models = status?.models || {};
  const isTraining = Object.values(models).some((m) => m.status === 'training');
  const currentModelStatus = models[currentMethod]?.status;

  // --- Initial load: algorithms, users, status -----------------------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [algos, usersResp, st] = await Promise.all([
          api.getAlgorithms(),
          api.getUsers(),
          api.getStatus(),
        ]);
        if (cancelled) return;
        setAlgorithms(algos);
        setUsers(usersResp.users);
        setStatus(st);
        if (usersResp.users.length) setSelectedUser(usersResp.users[0]);
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // --- Poll training status while any model is training --------------------
  useEffect(() => {
    if (!isTraining) return undefined;
    const id = setInterval(async () => {
      try {
        const st = await api.getStatus();
        setStatus(st);
        setAlgorithms(await api.getAlgorithms());
      } catch (e) {
        setError(e.message);
      }
    }, 1500);
    return () => clearInterval(id);
  }, [isTraining]);

  // --- Fetch recommendations when inputs change (and model is ready) -------
  const fetchRecommendations = useCallback(async () => {
    if (currentModelStatus !== 'ready') {
      setRecommendations([]);
      return;
    }
    setLoadingRecs(true);
    setRecError(null);
    try {
      const recs = await api.getRecommendations(selectedUser, currentMethod, numRecommendations);
      setRecommendations(recs);
    } catch (e) {
      setRecError(e.message);
      setRecommendations([]);
    } finally {
      setLoadingRecs(false);
    }
  }, [selectedUser, currentMethod, numRecommendations, currentModelStatus]);

  useEffect(() => {
    fetchRecommendations();
  }, [fetchRecommendations]);

  const handleTrain = async () => {
    try {
      const st = await api.train('all');
      setStatus(st.status);
    } catch (e) {
      setError(e.message);
    }
  };

  const methodMeta = (key) => algorithms.find((a) => a.key === key);
  const anyReady = Object.values(models).some((m) => m.status === 'ready');

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
            Real recommendations from SVD, NMF &amp; Neural CF on MovieLens 100K
          </p>
        </div>

        {error && (
          <div className="mb-8 p-4 bg-red-900/40 border border-red-600 rounded-lg flex items-center text-red-200">
            <AlertCircle className="w-5 h-5 mr-2" />
            <span>Backend error: {error}</span>
          </div>
        )}

        {/* Method Selection */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          {algorithms.map((algo) => {
            const style = METHOD_STYLE[algo.key] || METHOD_STYLE.SVD;
            const IconComponent = style.icon;
            const mStatus = models[algo.key]?.status;
            const disabled = mStatus === 'unavailable';
            return (
              <div
                key={algo.key}
                className={`p-6 rounded-xl border-2 transition-all duration-300 ${
                  disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                } ${
                  currentMethod === algo.key
                    ? 'border-purple-400 bg-purple-800/30 shadow-lg shadow-purple-400/20'
                    : 'border-gray-600 bg-gray-800/50 hover:border-gray-500'
                }`}
                onClick={() => !disabled && setCurrentMethod(algo.key)}
              >
                <div className="flex items-center mb-3">
                  <div className={`p-2 rounded-lg ${style.color} mr-3`}>
                    <IconComponent className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-semibold flex-1">{algo.name}</h3>
                </div>
                <p className="text-gray-400 text-sm mb-3">{algo.description}</p>
                <div className="flex items-center justify-between text-xs">
                  <span
                    className={`px-2 py-1 rounded-full ${
                      mStatus === 'ready'
                        ? 'bg-green-600'
                        : mStatus === 'training'
                        ? 'bg-blue-600 animate-pulse'
                        : 'bg-gray-600'
                    }`}
                  >
                    {STATUS_LABEL[mStatus] || mStatus}
                  </span>
                  {algo.metrics && (
                    <span className="text-green-400">
                      RMSE {algo.metrics.rmse} · MAE {algo.metrics.mae}
                    </span>
                  )}
                </div>
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
                {users.map((u) => (
                  <option key={u} value={u}>
                    User {u}
                  </option>
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
                {[3, 5, 10, 15].map((num) => (
                  <option key={num} value={num}>
                    {num} movies
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={handleTrain}
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
              ) : anyReady ? (
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
              {algorithms.map((algo) => {
                const mStatus = models[algo.key]?.status || 'not_started';
                const style = METHOD_STYLE[algo.key] || METHOD_STYLE.SVD;
                const IconComponent = style.icon;
                return (
                  <div key={algo.key} className="bg-gray-700/30 rounded-lg p-4">
                    <div className="flex items-center">
                      <div className={`p-2 rounded-lg ${style.color} mr-3`}>
                        <IconComponent className="w-4 h-4" />
                      </div>
                      <span className="font-medium flex-1">{algo.name}</span>
                      <span
                        className={`text-sm px-2 py-1 rounded-full ${
                          mStatus === 'ready'
                            ? 'bg-green-600 text-white'
                            : mStatus === 'training'
                            ? 'bg-blue-600 text-white animate-pulse'
                            : 'bg-gray-600 text-gray-300'
                        }`}
                      >
                        {STATUS_LABEL[mStatus] || mStatus}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Recommendations */}
        <div className="bg-gray-800/50 rounded-xl p-6">
          <div className="flex items-center mb-6">
            <Play className="w-6 h-6 text-purple-400 mr-3" />
            <h3 className="text-xl font-semibold">
              {methodMeta(currentMethod)?.name || currentMethod} Recommendations for User {selectedUser}
            </h3>
          </div>

          {currentModelStatus !== 'ready' ? (
            <div className="p-6 text-center text-gray-400">
              {currentModelStatus === 'unavailable'
                ? 'This model is unavailable (TensorFlow not installed on the backend).'
                : currentModelStatus === 'training'
                ? 'Model is training — recommendations will appear when it is ready.'
                : 'Model not trained yet. Click "Train All Models" to get started.'}
            </div>
          ) : loadingRecs ? (
            <div className="p-6 text-center text-gray-400">Loading recommendations…</div>
          ) : recError ? (
            <div className="p-4 bg-red-900/40 border border-red-600 rounded-lg text-red-200">
              {recError}
            </div>
          ) : (
            <div className="grid gap-4">
              {recommendations.map((rec, index) => (
                <div
                  key={rec.movie_id}
                  className="flex items-center p-4 bg-gray-700/50 rounded-lg hover:bg-gray-700/70 transition-all duration-300"
                >
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg flex items-center justify-center font-bold text-lg mr-4">
                    {index + 1}
                  </div>

                  <div className="flex-1">
                    <h4 className="text-lg font-semibold mb-1">{rec.title}</h4>
                    <div className="flex items-center text-sm text-gray-400 flex-wrap gap-x-4">
                      {rec.genres?.length > 0 && <span>{rec.genres.join(', ')}</span>}
                      {rec.year && <span>{rec.year}</span>}
                      {rec.avg_rating != null && (
                        <div className="flex items-center">
                          <Star className="w-4 h-4 text-yellow-400 mr-1" />
                          {rec.avg_rating} avg
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-lg font-semibold text-purple-400 mb-1">
                      {rec.predicted_rating.toFixed(2)}/5.0
                    </div>
                    <div className="text-xs text-gray-500">predicted</div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Dataset footer */}
          <div className="mt-8 p-4 bg-gray-700/30 rounded-lg">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center">
                <Users className="w-5 h-5 text-gray-400 mr-2" />
                <span className="text-sm text-gray-400">
                  Dataset: MovieLens 100K (100,000 ratings, {users.length || 943} users, 1,682 movies)
                </span>
              </div>
              <div className="text-sm text-gray-400">
                {methodMeta(currentMethod)?.description}
              </div>
            </div>
          </div>
        </div>

        {/* Performance Comparison */}
        {anyReady && (
          <div className="mt-8 bg-gray-800/50 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-6 flex items-center">
              <BarChart3 className="w-6 h-6 text-purple-400 mr-3" />
              Model Performance Comparison
            </h3>
            <div className="grid md:grid-cols-3 gap-4">
              {algorithms.map((algo) => {
                const style = METHOD_STYLE[algo.key] || METHOD_STYLE.SVD;
                return (
                  <div key={algo.key} className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center mb-3">
                      <div className={`w-3 h-3 rounded-full ${style.color} mr-2`}></div>
                      <span className="font-medium">{algo.name}</span>
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">RMSE:</span>
                        <span className="font-mono">{algo.metrics ? algo.metrics.rmse : '—'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">MAE:</span>
                        <span className="font-mono">{algo.metrics ? algo.metrics.mae : '—'}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
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
