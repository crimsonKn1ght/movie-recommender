// Thin API client for the Movie Recommender backend.
//
// Calls are same-origin by default: in development, package.json "proxy"
// forwards /api/* to the FastAPI backend on :8000; in production, the nginx
// container proxies /api/* to the backend service. Override with
// REACT_APP_API_URL (baked at build time) if you need a cross-origin backend.

const API_BASE = process.env.REACT_APP_API_URL || '';

async function http(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body && body.detail) detail = body.detail;
    } catch (_) {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  getAlgorithms: () => http('/api/algorithms'),
  getUsers: () => http('/api/users'),
  getStatus: () => http('/api/train/status'),
  train: (algorithm = 'all') =>
    http('/api/train', { method: 'POST', body: JSON.stringify({ algorithm }) }),
  getRecommendations: (userId, algorithm, n) =>
    http(`/api/recommend?user_id=${userId}&algorithm=${encodeURIComponent(algorithm)}&n=${n}`),
};

export default api;
