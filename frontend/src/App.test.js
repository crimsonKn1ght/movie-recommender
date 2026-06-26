import { render, screen } from '@testing-library/react';
import App from './App';

// The component talks to the backend on mount; mock the client so the test is
// hermetic (jsdom has no fetch and we don't want a live backend in CI).
jest.mock('./api/client', () => ({
  api: {
    getAlgorithms: () => Promise.resolve([]),
    getUsers: () => Promise.resolve({ count: 0, users: [] }),
    getStatus: () => Promise.resolve({ models: {} }),
    train: () => Promise.resolve({ status: { models: {} } }),
    getRecommendations: () => Promise.resolve([]),
  },
}));

test('renders movie recommender title', async () => {
  render(<App />);
  const titleElement = await screen.findByText(/Advanced Movie Recommender/i);
  expect(titleElement).toBeInTheDocument();
});
