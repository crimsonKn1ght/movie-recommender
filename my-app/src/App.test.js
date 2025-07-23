import { render, screen } from '@testing-library/react';
import App from './App';

test('renders movie recommender title', () => {
  render(<App />);
  const titleElement = screen.getByText(/Advanced Movie Recommender/i);
  expect(titleElement).toBeInTheDocument();
});